from typing import Dict
from rdflib import Graph, URIRef, RDF
from kgpipe_tasks.common.ontology import Ontology, OntologyUtil
import os
from kgpipe.common import Registry, Data, DataFormat
from pathlib import Path

def enrich_type_information(graph: Graph, ontology: Ontology, type_property: URIRef = RDF.type) -> Graph:
    type_dict = {}

    new_graph = Graph()

    for s, p, o in graph:
        domain, range = ontology.get_domain_range(str(p))
        if domain and isinstance(s, URIRef):
            if str(s) not in type_dict:
                type_dict[str(s)] = []
            type_dict[str(s)].append(str(domain))   
        if range and isinstance(o, URIRef):
            if str(o) not in type_dict:
                type_dict[str(o)] = []
            type_dict[str(o)].append(str(range))
        new_graph.add((s, p, o))

    for uri, types in type_dict.items():
        for type in types:
            new_graph.add((URIRef(uri), type_property, URIRef(type)))
    return new_graph

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES},
    output_spec={ "result": DataFormat.RDF_NTRIPLES}
)
def type_inference_ontology_simple(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    
    ontology_path = os.getenv("ONTOLOGY_PATH")
    if not ontology_path:
        raise ValueError("ONTOLOGY_PATH is not set")
    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))

    source_graph = Graph().parse(inputs["source"].path)
    result_graph = enrich_type_information(source_graph, ontology)
    result_graph.serialize(destination=outputs["result"].path, format="nt")