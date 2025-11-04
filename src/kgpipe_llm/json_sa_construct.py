
from kgpipe.common import Registry, KG, Data, DataFormat
from kgpipe_llm.common.core import get_client_from_env
from kgpipe_llm.common.snippets import generate_ontology_snippet
from kgpipe_llm.common.models import OntologyMappings
from kgcore.model.ontology import Ontology, OntologyUtil
from pydantic import BaseModel, Field
from typing import Optional, Dict, Callable, List
import jsonpath_ng
import json
from pathlib import Path
import os
from rdflib import Graph

ONTOLOGY = None
def get_ontology() -> Ontology:
    global ONTOLOGY
    if ONTOLOGY is None:
        ontology_path = os.getenv("ONTOLOGY_PATH")
        if ontology_path is None:
            raise ValueError("ONTOLOGY_PATH is not set")
        ONTOLOGY = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    return ONTOLOGY


def apply_to_file_or_files_in_dir(func: Callable, input_path: Path, output_path: Path, parallel: int = 1) -> None:
    from multiprocessing import Pool

    if os.path.isfile(input_path):
        func(input_path, output_path)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        if parallel > 1:
            file_pairs = [
                (os.path.join(input_path, file), os.path.join(output_path, file))
                for file in os.listdir(input_path)
            ]

            with Pool(parallel) as p:
                p.starmap(func, file_pairs)
            p.join()
            p.close()
        else:
            for file in os.listdir(input_path):
                func(os.path.join(input_path, file), os.path.join(output_path, file))


# Ontology matching models
class OntologyMapping_v1(BaseModel):
    """Mapping between source schema and target ontology."""
    source_path: str = Field(..., description="Field in source schema")
    # target_class: str = Field(..., description="Target class in ontology")
    target_property: str = Field(..., description="Target property in ontology")
    mapping_type: str = Field(..., description="Type of mapping (direct, transform, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the mapping")

class OntologyMappings_v1(BaseModel):
    """Container for ontology mappings."""
    mappings: List[OntologyMapping_v1] = Field(..., description="List of ontology mappings")

class JSON_SA_Construct_v1:

    @staticmethod
    def map_json_to_ontology(json_data: dict, ontology: Ontology) -> Optional[OntologyMappings_v1]:
        """
        Map a JSON object to an ontology.
        """

        system_prompt = f"""
        You are a data‑integration engineer. Constructing a json_path to ontology mapping in JSON format.
        """

        prompt = f"""
        Ontology glossary
        -----------------
        {generate_ontology_snippet(ontology)}

        JSON sample
        -----------
        {json.dumps(json_data, indent=4)}
        """

        # Write an RML mapping in Turtle that:

        response_dict = get_client_from_env().send_prompt(prompt, OntologyMappings_v1, system_prompt=system_prompt)

        return OntologyMappings_v1(**response_dict)


    @staticmethod
    def construct_rdf_with_path_mapping(ontology_mapping : OntologyMappings_v1, json_data: dict):
        """
        Build an RDF graph with the json_path to ontology_relation alignment.
        The json data is a dict or a list of dicts, where each is constructed as an entity.
        For nested json_data decide on the matching ontology_property for each nested json_data.
        """

        ONTOLOGY = get_ontology()

        for mapping in ontology_mapping.mappings:
            jsonpath = mapping.source_path
            # ontology_class = mapping.target_class
            ontology_property = mapping.target_property

            jsonpath_expr = jsonpath_ng.parse(jsonpath)
            matches = jsonpath_expr.find(json_data)



    @staticmethod
    def sa_and_mapping_from_json_file_to_json_file(input_path: Path, output_path: Path):
        """
        Construct a knowledge graph from a JSON object.
        """
        json_data = json.load(open(input_path))
        ontology_mapping = JSON_SA_Construct_v1.map_json_to_ontology(json_data, get_ontology())
        if ontology_mapping is not None:
            JSON_SA_Construct_v1.construct_rdf_with_path_mapping(ontology_mapping, json_data)
        json.dump(json_data, open(output_path, "w"), indent=4)

@Registry.task(
    description="Map JSON keys to JSON file.",
    input_spec={"json": DataFormat.JSON},
    output_spec={"rdf": DataFormat.RDF_NTRIPLES},
    category=["JSON", "RDF Construct"]
)
def json_sa_and_mapping_v1(inputs: Dict[str, Data], outputs : Dict[str, Data]):
    
    input_path = inputs["json"].path
    output_path = outputs["json"].path

    apply_to_file_or_files_in_dir(JSON_SA_Construct_v1.sa_and_mapping_from_json_file_to_json_file, input_path, output_path, 8)


# TODO create the sample method
