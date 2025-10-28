from logging import getLogger
from pydantic import BaseModel
from rdflib import OWL, Graph, URIRef, RDFS, RDF, SKOS
from pathlib import Path
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document
from kgpipe.evaluation.cluster import MatchCluster
from kgpipe.common.models import KgTask, DataFormat, Data
from typing import Dict, List, Optional
import json
from kgpipe.common.registry import Registry
import os
from kgcore.model.ontology import OntologyUtil
from kgpipe.execution.config import TARGET_ONTOLOGY_NAMESPACE
from kgpipe_tasks.entity_resolution.fusion.util import load_matches_from_file

logger = getLogger(__name__)

class TrackRecord(BaseModel):
    original_subject: str
    subject: str
    original_predicate: str
    predicate: str
    original_object: str
    object: str

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Merge RDF entities using first value fusion",
    category=["EntityResolution", "Fusion"]
)
def select_first_value(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    For two KGs A and B, merge A into B where for each s_p and 
    1) p is fusable and B does not have any s_p_o or 
    2) p is not fusable erge all s_p_o
    """
    ontology_path = os.environ.get("ONTOLOGY_PATH", "false")
    if ontology_path == "false":
        raise ValueError("ONTOLOGY_PATH is not set")

    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    allowed_predicates = set[str]([str(p.uri) for p in ontology.properties]+[str(RDFS.label), str(RDF.type), str(SKOS.altLabel)])
    fusable_properties = set[str]([str(p.uri) for p in ontology.properties if p.max_cardinality == 1]+[str(RDFS.label), str(RDF.type)])  

    def is_fusable(p):
        return str(p) in fusable_properties

    source_graph = Graph()
    source_graph.parse(inputs["source"].path, format="nt")
    seed_graph = Graph() # seed graph
    seed_graph.parse(inputs["target"].path, format="nt")

    current_subjects = set[str]([str(s) for s in seed_graph.subjects(unique=True)])

    selected: List[TrackRecord] = []
    discarded: List[TrackRecord] = []
    
    for s, p, o in source_graph:
        s_can = s
        p_can = p
        o_can = o 

        if not isinstance(p_can, URIRef) or str(p_can) not in allowed_predicates:
            continue
        
        if p_can == RDF.type and not str(o_can).startswith(TARGET_ONTOLOGY_NAMESPACE):
            continue

        if is_fusable(p_can):
            # Add exactly one value if none exists yet
            if not any(seed_graph.objects(s_can, p_can)):
                seed_graph.add((s_can, p_can, o_can))
                selected.append(
                    TrackRecord(subject=s_can,predicate=p_can,object=o,original_subject=s,original_predicate=p,original_object=o))
                # keep subjects set fresh for subsequent matches
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))
            else:
                discarded.append(
                    TrackRecord(subject=s_can,predicate=p_can,object=o,original_subject=s,original_predicate=p,original_object=o))
        else:
            # Non-fusable: copy if not already present (avoid dupes)
            if (s_can, p_can, o_can) not in seed_graph:
                seed_graph.add((s_can, p_can, o_can))
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))

    # sel(ected)
    selected_file_path = outputs["output"].path.parent / (outputs["output"].path.stem + ".selected.json")
    with open(selected_file_path, "w") as f:
        json.dump(selected, f, default=lambda x: x.model_dump())
    # dis(carded)
    discarded_file_path = outputs["output"].path.parent / (outputs["output"].path.stem + ".discarded.json")
    with open(discarded_file_path, "w") as f:
        json.dump(discarded, f, default=lambda x: x.model_dump())

    # prov graph is skipped here as no uris are replaced (is done in previouse steps)
    seed_graph.serialize(outputs["output"].path, format="nt")

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES, "matches1": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Fuse RDF entities using first value fusion",
    category=["EntityResolution", "Fusion"]
)
def fusion_first_value(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    Fuse RDF entities
    - replacing ids of target graph with ids of source graph based on matches
    - only fusable properties are fused
    - selects the first value from source graph if no target value exists (does not add values from target graph)
    - also if target graph has multiple values for a property, the first value is selected (for new entities)
    """
    ontology_path = os.environ.get("ONTOLOGY_PATH", "false")
    if ontology_path == "false":
        raise ValueError("ONTOLOGY_PATH is not set")
    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))

    entity_matching_threshold = float(os.environ.get("ENTITY_MATCHING_THRESHOLD", 0.5))
    relation_matching_threshold = float(os.environ.get("RELATION_MATCHING_THRESHOLD", 0.5))
    allowed_predicates = set[str]([str(p.uri) for p in ontology.properties]+[str(RDFS.label), str(RDF.type), str(SKOS.altLabel)])
    fusable_properties = set[str]([str(p.uri) for p in ontology.properties if p.max_cardinality == 1]+[str(RDFS.label), str(RDF.type)])

    def is_fusable(p):
        return str(p) in fusable_properties

    entity_matches = load_matches_from_file(inputs["matches1"].path, entity_matching_threshold, "entity")
    relation_matches = load_matches_from_file(inputs["matches1"].path, relation_matching_threshold, "relation")

    source_graph = Graph()
    source_graph.parse(inputs["source"].path, format="nt")
    seed_graph = Graph() # seed graph
    seed_graph.parse(inputs["kg"].path, format="nt")

    current_subjects = set[str]([str(s) for s in seed_graph.subjects(unique=True)])

    sameAsProv = {}

    def canonicalize_entity_term(term):
        """Map a URI from the target graph to the matching source URI, if any."""
        if isinstance(term, URIRef):
            t_str = str(term)
            cluster = entity_matches.get_cluster(t_str)
            if cluster:
                right_candidates = [c for c in cluster if not c == t_str]
                if len(right_candidates) > 2:
                    raise ValueError(f"Multiple matches found for {t_str}")
                else:
                    for m in right_candidates:
                        # if not m == t_str:
                        sameAsProv[str(term)] = str(m)
                        return URIRef(m)
                    return term
            else:
                return term
        return term

    def canonicalize_property_term(term):
        """Map a URI from the target graph to the matching source URI, if any."""
        if isinstance(term, URIRef):
            t_str = str(term)
            mapped = relation_matches.has_match_to_namespace(t_str, TARGET_ONTOLOGY_NAMESPACE)
            if mapped:
                return URIRef(mapped)
        return term

    selected: List[TrackRecord] = []
    discarded: List[TrackRecord] = []

    for s, p, o in source_graph:
        # Canonicalize
        logger.debug(f"Canonicalizing {s}, {p}, {o}")
        s_can = canonicalize_entity_term(s)
        p_can = canonicalize_property_term(p)
        o_can = canonicalize_entity_term(o) if isinstance(o, URIRef) else o  # keep literals/bnodes as-is

        # Only work with properties that are in our ontology (after canonicalization)
        if not isinstance(p_can, URIRef) or str(p_can) not in allowed_predicates:
            logger.debug(f"Skipping {s}, {p}, {o} because it is not in the allowed predicates")
            continue

        if p_can == RDF.type and not str(o_can).startswith(TARGET_ONTOLOGY_NAMESPACE):
            continue

        if is_fusable(p_can):
            # Add exactly one value if none exists yet
            if not any(seed_graph.objects(s_can, p_can)):
                seed_graph.add((s_can, p_can, o_can))
                selected.append(
                    TrackRecord(subject=s_can,predicate=p_can,object=o,original_subject=s,original_predicate=p,original_object=o))
                # keep subjects set fresh for subsequent matches
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))
            else:
                discarded.append(
                    TrackRecord(subject=s_can,predicate=p_can,object=o,original_subject=s,original_predicate=p,original_object=o))
        else:
            # Non-fusable: copy if not already present (avoid dupes)
            if (s_can, p_can, o_can) not in seed_graph:
                seed_graph.add((s_can, p_can, o_can))
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))

    # sel(ected)
    selected_file_path = outputs["output"].path.parent / (outputs["output"].path.stem + ".selected.json")
    with open(selected_file_path, "w") as f:
        json.dump(selected, f, default=lambda x: x.model_dump())
    # dis(carded)
    discarded_file_path = outputs["output"].path.parent / (outputs["output"].path.stem + ".discarded.json")
    with open(discarded_file_path, "w") as f:
        json.dump(discarded, f, default=lambda x: x.model_dump())

    prov_graph = Graph()
    for sid,gid in sameAsProv.items():
        prov_graph.add((URIRef(gid), OWL.sameAs, URIRef(sid)))
    prov_graph.serialize(outputs["output"].path.as_posix() + ".prov", format="nt")
    seed_graph.serialize(outputs["output"].path, format="nt")

@DeprecationWarning
def preference_fusion(values_with_scores, preferred_source='target_kg'):
    """
    Implements preference-based fusion strategy.
    
    Args:
        values_with_scores: List of tuples (value, score, source)
        preferred_source: The source to prefer when scores are similar
    
    Returns:
        The value from the preferred source if available, otherwise highest scoring
    """
    if not values_with_scores:
        return None
    
    # Find values from preferred source
    preferred_values = [(v, s, src) for v, s, src in values_with_scores if src == preferred_source]
    
    if preferred_values:
        # Return highest scoring value from preferred source
        return max(preferred_values, key=lambda x: x[1])[0]
    
    # Fall back to highest scoring value from any source
    return max(values_with_scores, key=lambda x: x[1])[0]