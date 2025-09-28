from kgpipe.common import Registry, Data, DataFormat
from kgpipe_llm.common import LLMClient
from kgpipe_tasks.common import OntologyUtil, Ontology
from kgpipe_llm.common.models import OntologyMappings
from kgpipe_llm.common.snippets import generate_ontology_snippet

from kgpipe_llm.common.core import get_client_from_env

import json
import jsonpath_ng
from typing import Optional, Dict
from pathlib import Path


def map_json_to_ontology(json_data: dict, ontology: Ontology) -> Optional[OntologyMappings]:
    """
    Map a JSON object to an ontology.
    """

    system_prompt = f"""
    You are a data‑integration engineer. Constructing a json_path to ontology mapping in JSON format.
    """
    
    task_part = f"""
    * create jsonpath to ontology mapping
    """

    prompt = f"""
    Ontology glossary
    -----------------
    {generate_ontology_snippet(ontology)}

    JSON sample
    -----------
    {json.dumps(json_data, indent=4)}
    
    Task
    ----
    {task_part}
    """

    # Write an RML mapping in Turtle that:

    response =get_client_from_env().send_prompt(prompt, OntologyMappings, system_prompt=system_prompt)
    return OntologyMappings(**response)

def execute_json_ontology_mapping(json_path: Path, ontology_path: Path) -> Optional[OntologyMappings]:
    """
    Execute the JSON to Ontology mapping.
    """
    json_data = json.load(json_path.open())
    ontology = OntologyUtil.load_ontology_from_file(ontology_path)
    return map_json_to_ontology(json_data, ontology)

def apply_jsonpath_mapping(ontology_mapping : OntologyMappings, json_data: dict) -> Optional[dict]:
    """
    Apply the ontology mapping to the JSON data.
    """
    for mapping in ontology_mapping.mappings:
        jsonpath = mapping.source_field
        ontology_class = mapping.target_class
        ontology_property = mapping.target_property

        jsonpath_expr = jsonpath_ng.parse(jsonpath)
        matches = jsonpath_expr.find(json_data)

        for match in matches:
            print(match.value, ontology_class, ontology_property)

# task definitons

@Registry.task(
    description="Map a JSON object to an ontology (json_path statement -> property).",
    input_spec={"json_data": DataFormat.JSON, "ontology": DataFormat.RDF_TTL},
    output_spec={"ontology_mapping": DataFormat.JSON_ONTO_MAPPING_JSON},
    category=["Construction", "LLM"]
)
def json_ontology_mapping(input: Dict[str, Data], output: Dict[str, Data]):
    """
    Map a JSON object to an ontology.
    """
    return execute_json_ontology_mapping(input["json_data"].path, input["ontology"].path)


@Registry.task(
    description="Align relations of a source KG with a target KG.",
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.JSON},
    category=["Construction", "LLM"]
)
def align_kg(input: Dict[str, Data], output: Dict[str, Data]):
    """
    Align relations of a source KG with a target KG.
    """
    llm = LLMClient()
    prompt = f"""
    Align relations of a source KG with a target KG.

    === Source KG ===
    {open(input["source"].path, "r").read()}

    === Target KG ===
    {open(input["target"].path, "r").read()}

    === OUTPUT ===
    {{
        "mappings": [
            {{
                "source_field": "...",
                "target_class": "...",
                "target_property": "...",
                "mapping_type": "...",
                "confidence": 0.0
            }}
        ]
    }}

    === Task ===
    * align relations of the source KG with the target KG
    * return the aligned KG in the provided output format
    * only return the aligned relations as JSON, no other text
    """
    
    response_dict = llm.send_prompt(prompt, OntologyMappings)
    mappings = OntologyMappings(**response_dict)

    with open(output["output"].path, "w") as f:
        f.write(mappings.model_dump_json(indent=4))