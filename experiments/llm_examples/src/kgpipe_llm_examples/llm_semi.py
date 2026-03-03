# Generating RDF triples

from kgcore.api.ontology import Ontology, OntologyUtil
from kgpipe.common import Registry, Data, DataFormat
from kgpipe_llm.common.snippets import generate_ontology_snippet_v3
from kgpipe_llm.common.core import get_client_from_env
from kgpipe_llm.json_mapping import JSON_Mapping_v1
import os
from pathlib import Path
from typing import Dict
from kgpipe_llm_examples.struct_out_schema import RDFTriples
from rdflib import Graph, URIRef, Literal
import json
ONTOLOGY = None
def get_ontology() -> Ontology:
    global ONTOLOGY
    if ONTOLOGY is None:
        ontology_path = os.getenv("ONTOLOGY_PATH")
        if ontology_path is None:
            raise ValueError("ONTOLOGY is not set")
        ONTOLOGY = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    return ONTOLOGY

def triples_to_graph(triples: RDFTriples) -> Graph:
    """
    Convert RDFTriples into an rdflib.Graph.
    Handles IRIs, literals, datatypes, and language tags.
    """
    g = Graph()

    for t in triples.triples:
        subj = URIRef(str(t.subject.iri))
        pred = URIRef(str(t.predicate.iri))

        if t.object.kind == "iri":
            obj = URIRef(str(t.object.value))
        else:
            # Literal
            obj = Literal(
                t.object.value or "",
                datatype=URIRef(str(t.object.datatype)) if t.object.datatype else None,
                lang=t.object.language if t.object.language else None
            )
        g.add((subj, pred, obj))
    return g

@Registry.task(input_spec={"input": DataFormat.JSON}, output_spec={"output": DataFormat.RDF_NTRIPLES})
def test_mapping_direct(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    Generate RDF triples from a JSON file without providing an ontology.
    """
    json_file = inputs["input"].path
    json_data = json.dumps(json.load(open(json_file)), indent=4)
    prompt = f"""
    Generate RDF triples from the following JSON file:

    {json_data}

    """

    system_prompt = "You are a KG engineer. Generating RDF triples from a JSON file."
    llm = get_client_from_env()

    print(prompt)

    response = llm.send_prompt(prompt, RDFTriples, system_prompt)
    
    print(response)
    try:
        response = RDFTriples(**response)
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(response)
        raise e

    graph = triples_to_graph(response)
    graph.serialize(outputs["output"].path, format="nt")
    pass

def test_mapping_target(): # already implemented in kgpipe_llm.tasks.llm_task_map_and_construct
    pass

# Generating RML mappings

def test_mapping_rml_direct():
    pass

def test_mapping_rml_target():
    ONTOLOGY=""
    pass

# Generating Python code to construct RDF triples

def test_mapping_python_direct():
    pass

def test_mapping_python_target():
    ONTOLOGY=""
    pass