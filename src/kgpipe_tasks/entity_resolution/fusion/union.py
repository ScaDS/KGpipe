from rdflib import OWL, Graph, URIRef, RDFS, RDF, SKOS
from pathlib import Path
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document
from kgpipe.evaluation.cluster import MatchCluster
from kgpipe.common.models import KgTask, DataFormat, Data
from typing import Dict, Optional
import json
from kgpipe.common.registry import Registry
import os
from kgcore.model.ontology import OntologyUtil
from kgpipe.execution.config import SOURCE_NAMESPACE, TARGET_ONTOLOGY_NAMESPACE, TARGET_RESOURCE_NAMESPACE


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