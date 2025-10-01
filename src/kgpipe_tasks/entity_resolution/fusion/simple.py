from rdflib import Graph, URIRef, RDFS, RDF, SKOS
from pathlib import Path
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document
from kgpipe.evaluation.cluster import MatchCluster
from kgpipe.common.models import KgTask, DataFormat, Data
from typing import Dict, Optional
import json
from kgpipe.common.registry import Registry
import os
from kgpipe_tasks.common.ontology import OntologyUtil

# TODO read from global state
# TODO multiple target namespaces possible
SOURCE_NAMESPACE = "http://kg.org/rdf/"
TARGET_RESOURCE_NAMESPACE = "http://kg.org/resource/"
TARGET_ONTOLOGY_NAMESPACE = "http://kg.org/ontology/"

from logging import getLogger
logger = getLogger(__name__)


def load_matches_from_file(file_path, threshold, type_filter: Optional[str] = None) -> MatchCluster:

    er = ER_Document(**json.load(open(file_path)))

    matches = MatchCluster()

    for match in er.matches:
        if match.score > threshold:
            if type_filter and match.id_type != type_filter:
                continue
            id_1 = str(match.id_1) # .split("-")[0] # paris workaround
            id_2 = str(match.id_2) # .split("-")[0] # paris workaround
            if id_1.endswith("-") or id_2.endswith("-"):
                continue
            if id_1.endswith("_uri") or id_1.endswith("_literal"):
                id_1 = id_1.replace("_uri", "").replace("_literal", "")
            if id_2.endswith("_uri") or id_2.endswith("_literal"):
                id_2 = id_2.replace("_uri", "").replace("_literal", "")
            if match.id_type == "relation":
                print(f"Adding relation match: {id_1} {id_2}")
            matches.add_match(id_1, id_2)

    return matches

def fuse_rdf_files(f1,f2,er):
    g = Graph()
    g.parse(f1)
    g.parse(f2)

    ng = Graph()

    entity_matches = {}
    relation_matches = {}
    for em in [em for em in er.matches if em.score > 0.5]:
        if em.id_type == "entity":
            if em.id_1.startswith(SOURCE_NAMESPACE) and em.id_2.startswith(TARGET_RESOURCE_NAMESPACE):
                entity_matches[em.id_1] = em.id_2
            if em.id_1.startswith(TARGET_RESOURCE_NAMESPACE) and em.id_2.startswith(SOURCE_NAMESPACE):
                entity_matches[em.id_2] = em.id_1
            else:
                # print("not merged", em.id_1, em.id_2)
                continue
        else:
            if em.id_1.endswith("-") or em.id_2.endswith("-"):
                continue
            if em.id_1.startswith(SOURCE_NAMESPACE) and em.id_2.startswith(TARGET_RESOURCE_NAMESPACE):
                relation_matches[em.id_1] = em.id_2
            if em.id_1.startswith(TARGET_RESOURCE_NAMESPACE) and em.id_2.startswith(SOURCE_NAMESPACE):
                relation_matches[em.id_2] = em.id_1
            else:
                # print("not merged", em.id_1, em.id_2)
                continue

    for s,p,o in g:
        sub = s
        if str(s) in entity_matches:
            sub = URIRef(entity_matches[str(s)])
        pred = p
        if str(p) in relation_matches:#
            # print("merged", str(p), relation_matches[str(p)])
            pred = URIRef(relation_matches[str(p)])
        obj = o
        if isinstance(o, URIRef) and str(o) in entity_matches:
            obj = URIRef(entity_matches[str(o)])

        ng.add((sub,pred,obj))
    
    return ng

#######################################

# === TASK DEFINITIONS ===

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Union RDF files without entity matching",
    category=["EntityResolution", "Fusion"]
)
def fusion_union_rdf(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    Path(outputs["output"].path).touch()
    er = ER_Document()
    # TODO
    graph = fuse_rdf_files(inputs["source"].path, inputs["target"].path, er)
    graph.serialize(outputs["output"].path, format="nt")

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES, "matches": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Union RDF files with entity matching using provided matches",
    category=["EntityResolution", "Fusion"]
)
def union_matched_rdf(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    Union RDF files with entity matching using provided matches.
    
    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """
    Path(outputs["output"].path).touch()
    er = ER_Document(**json.load(open(inputs["matches"].path)))
    graph = fuse_rdf_files(inputs["source"].path, inputs["target"].path, er)
    graph.serialize(outputs["output"].path, format="nt")

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES, "matches1": DataFormat.ER_JSON, "matches2": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Union RDF files with entity matching using provided matches",
    category=["EntityResolution", "Fusion"]
)
def union_matched_rdf_combined(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    Path(outputs["output"].path).touch()
    er1  = ER_Document(**json.load(open(inputs["matches1"].path)))
    er2  = ER_Document(**json.load(open(inputs["matches2"].path)))
    er_comb = ER_Document(matches=er1.matches + er2.matches)

    graph = fuse_rdf_files(inputs["source"].path, inputs["target"].path, er_comb)
    graph.serialize(outputs["output"].path, format="nt")






# union_matched_rdf_combined_task = KgTask(
#     name = "union",
#     inputDict={"source": RDF_NT, "target": RDF_NT, "matches1": ER_JSON, "matches2": ER_JSON},
#     outputDict={"output": RDF_NT},
#     function=union_matched_rdf_combined
# )


def majority_fusion(values_with_scores):
    """
    Implements majority voting fusion strategy.
    
    Args:
        values_with_scores: List of tuples (value, score, source) where:
            - value: The actual value
            - score: Confidence score for this value
            - source: Source identifier (e.g., 'source_kg', 'target_kg')
    
    Returns:
        The value that appears most frequently, with ties broken by highest score
    """
    if not values_with_scores:
        return None
    
    # Count occurrences of each value
    value_counts = {}
    value_scores = {}
    
    for value, score, source in values_with_scores:
        if value not in value_counts:
            value_counts[value] = 0
            value_scores[value] = []
        
        value_counts[value] += 1
        value_scores[value].append(score)
    
    # Find the most frequent value(s)
    max_count = max(value_counts.values())
    most_frequent = [v for v, count in value_counts.items() if count == max_count]
    
    if len(most_frequent) == 1:
        return most_frequent[0]
    
    # Break ties by highest average score
    best_value = most_frequent[0]
    best_avg_score = sum(value_scores[best_value]) / len(value_scores[best_value])
    
    for value in most_frequent[1:]:
        avg_score = sum(value_scores[value]) / len(value_scores[value])
        if avg_score > best_avg_score:
            best_value = value
            best_avg_score = avg_score
    
    return best_value

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

# def first_value_fusion(values_with_scores, target_source='target_kg'):
#     """
#     Implements first-value fusion strategy.
#     Takes values from target KG first, only from source if no target value exists.
    
#     Args:
#         values_with_scores: List of tuples (value, score, source)
#         target_source: The source to prioritize (default: 'target_kg')
    
#     Returns:
#         The first available value from target source, or first from any source if none available
#     """
#     if not values_with_scores:
#         return None
    
#     # First try to get value from target source
#     target_values = [(v, s, src) for v, s, src in values_with_scores if src == target_source]
#     if target_values:
#         return target_values[0][0]  # Return first target value
    
#     # If no target values, return first available value
#     return values_with_scores[0][0]

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Merge RDF entities using first value fusion",
    category=["EntityResolution", "Fusion"]
)
def select_first_value(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    ontology_path = os.environ.get("ONTOLOGY_PATH", "false")
    if ontology_path == "false":
        raise ValueError("ONTOLOGY_PATH is not set")

    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    allowed_predicates = set([str(p.uri) for p in ontology.properties]+[str(RDFS.label), str(RDF.type), str(SKOS.altLabel)])
    fusable_properties = set([str(p.uri) for p in ontology.properties if p.max_cardinality == 1]+[str(RDFS.label), str(RDF.type)])  

    def is_fusable(p):
        return str(p) in fusable_properties

    source_graph = Graph()
    source_graph.parse(inputs["source"].path, format="nt")
    seed_graph = Graph() # seed graph
    seed_graph.parse(inputs["target"].path, format="nt")

    current_subjects = set([str(s) for s in seed_graph.subjects(unique=True)])
    # allowed_predicates = set([str(p.uri) for p in ontology.properties] + [str(RDFS.label), str(RDF.type)])
    
    for s, p, o in source_graph:

        # Canonicalize
        s_can = s
        p_can = p
        o_can = o 

        # Only work with properties that are in our ontology (after canonicalization)
        if not isinstance(p_can, URIRef) or str(p_can) not in allowed_predicates:
            continue
        
        if p_can == RDF.type and not str(o_can).startswith(TARGET_ONTOLOGY_NAMESPACE):
            continue

        if is_fusable(p_can):
            # Add exactly one value if none exists yet
            if not any(seed_graph.objects(s_can, p_can)):
                seed_graph.add((s_can, p_can, o_can))
                # keep subjects set fresh for subsequent matches
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))
        else:
            # Non-fusable: copy if not already present (avoid dupes)
            if (s_can, p_can, o_can) not in seed_graph:
                seed_graph.add((s_can, p_can, o_can))
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))
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
    entity_matching_threshold = float(os.environ.get("ENTITY_MATCHING_THRESHOLD", 0.5))
    relation_matching_threshold = float(os.environ.get("RELATION_MATCHING_THRESHOLD", 0.5))

    print(f"[CONFIG] Entity matching threshold: {entity_matching_threshold}")
    print(f"[CONFIG] Relation matching threshold: {relation_matching_threshold}")

    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    allowed_predicates = set([str(p.uri) for p in ontology.properties]+[str(RDFS.label), str(RDF.type), str(SKOS.altLabel)])
    fusable_properties = set([str(p.uri) for p in ontology.properties if p.max_cardinality == 1]+[str(RDFS.label), str(RDF.type)])

    print("### fusable predicates ###")
    print(fusable_properties)
    print("### allowed predicates ###")
    print(allowed_predicates)

    def is_fusable(p):
        return str(p) in fusable_properties

    entity_matches = load_matches_from_file(inputs["matches1"].path, entity_matching_threshold, "entity")
    relation_matches = load_matches_from_file(inputs["matches1"].path, relation_matching_threshold, "relation")
    # matches.is_match(uri1, uri2)
    # matches.has_match_to_namespace(target_uri, src_namespace)

    source_graph = Graph()
    source_graph.parse(inputs["source"].path, format="nt")
    seed_graph = Graph() # seed graph
    seed_graph.parse(inputs["kg"].path, format="nt")

    current_subjects = set([str(s) for s in seed_graph.subjects(unique=True)])

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
            # print(f"Fusing {s_can}, {p_can}, {o_can}")
            # Add exactly one value if none exists yet
            if not any(seed_graph.objects(s_can, p_can)):
                seed_graph.add((s_can, p_can, o_can))
                # keep subjects set fresh for subsequent matches
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))
        else:
            # Non-fusable: copy if not already present (avoid dupes)
            if (s_can, p_can, o_can) not in seed_graph:
                seed_graph.add((s_can, p_can, o_can))
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))
        
    seed_graph.serialize(outputs["output"].path, format="nt")

# @flextask({"source": "rdf", "target": "rdf", "matches": "em"}, {"output": "rdf"})
# def custom_rdf_fusion_union(i,o):

#     def wrapper():
#         return fusion_union_rdf(i["source"].location, i["target"].location, i["matches"].location, o["output"].location, "turtle")

#     return wrapper
#     # pass


# def fusion_union(left_graph: Graph, right_graph: Graph, matches):
#     output_graph = Graph()
#     output_graph += left_graph
#     output_graph += right_graph

#     for match in matches:
#         left_uri = URIRef(match['left'])
#         right_uri = URIRef(match['right'])

#         triples_to_replace = list(output_graph.triples((right_uri, None, None))) + \
#                              list(output_graph.triples((None, None, right_uri)))

#         for s, p, o in triples_to_replace:
#             output_graph.remove((s, p, o))
#             if s == right_uri:
#                 s = left_uri
#             if o == right_uri:
#                 o = left_uri
#             output_graph.add((s, p, o))
#     return output_graph


# def fusion_union_rdf(rdf_left_path: str, rdf_right_path: str, em_json_path: str, output_rdf_path: str, format: str):
#     left_graph = Graph()
#     right_graph = Graph()

#     left_graph.parse(rdf_left_path, format=format)
#     right_graph.parse(rdf_right_path, format=format)

#     with open(em_json_path, 'r') as f:
#         matches = json.load(f)

#     output_graph = fusion_union(left_graph, right_graph, matches)

#     output_graph.serialize(destination=output_rdf_path, format=format)

#     return f"Fused RDF saved to {output_rdf_path}".encode('utf-8')



# if __name__ == "__main__":
#     g = fuse_rdf_files("/data/bench/sources/nt/person11.nt",
#                    "/data/bench/sources/nt/person12.nt",
#                    "/data/bench/results/person_11+12.er.json"
#                    )
#     g.serialize("/data/bench/results/person11+12.nt", format="ntriples")

@Registry.task(
    input_spec={"json1": DataFormat.ER_JSON, "json2": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.ER_JSON},
    description="Fuse RDF entities using first value fusion",
    category=["EntityResolution", "Fusion"]
)
def aggregate_2matches(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    er1 = ER_Document(**json.load(open(inputs["json1"].path)))
    er2 = ER_Document(**json.load(open(inputs["json2"].path)))
    
    n_er1 = []
    for er in er1.matches:
        if er.id_type:
            ne = ER_Match(id_1=er.id_1, id_2=er.id_2, id_type=er.id_type.replace("str","relation"), score=er.score)
        else:
            ne = ER_Match(id_1=er.id_1, id_2=er.id_2, score=er.score)
        n_er1.append(ne)
    n_er2 = []
    for er in er2.matches:
        if er.id_type:
            ne = ER_Match(id_1=er.id_1, id_2=er.id_2, id_type=er.id_type.replace("str","relation"), score=er.score)
        else:
            ne = ER_Match(id_1=er.id_1, id_2=er.id_2, score=er.score)
        n_er2.append(ne)
    er_comb = ER_Document(matches=n_er1 + n_er2)
    
    with open(outputs["output"].path, "w") as f:
        json.dump(er_comb.model_dump(), f, indent=4)

from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Match

@Registry.task(
    input_spec={"json1": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.ER_JSON},
    description="Fuse RDF entities using first value fusion",
    category=["EntityResolution", "Fusion"]
)
def reduce_to_best_match_per_entity(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    er1 = ER_Document(**json.load(open(inputs["json1"].path)))

    # Sort all candidate matches by score (high to low)
    sorted_matches = sorted(er1.matches, key=lambda m: m.score, reverse=True)

    selected_matches = []
    matched_ids = set()  # entities already committed to a match

    for match in sorted_matches:
        id1 = match.id_1
        id2 = match.id_2
        # Only accept if both endpoints are currently unmatched
        if id1 not in matched_ids and id2 not in matched_ids:
            selected_matches.append(match)
            matched_ids.add(id1)
            matched_ids.add(id2)

    er_comb = ER_Document(matches=selected_matches)
    with open(outputs["output"].path, "w") as f:
        json.dump(er_comb.model_dump(), f, indent=4)

# def reduce_to_best_match_per_entity(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    
#     current_per_id : Dict[str, ER_Match] = {}
    
#     er1 = ER_Document(**json.load(open(inputs["json1"].path)))

#     for match in er1.matches:
#         score = match.score
#         id1 = match.id_1
#         id2 = match.id_2
#         if id1 in current_per_id:
#             if current_per_id[id1].score < score:
#                 current_per_id[id1] = match
#         else:
#             current_per_id[id1] = match
#         if id2 in current_per_id:
#             if current_per_id[id2].score < score:
#                 current_per_id[id2] = match
#         else:
#             current_per_id[id2] = match
    
#     er_comb = ER_Document(matches=list(current_per_id.values()))
#     with open(outputs["output"].path, "w") as f:
#         json.dump(er_comb.model_dump(), f, indent=4)