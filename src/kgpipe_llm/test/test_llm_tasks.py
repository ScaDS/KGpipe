from kgpipe_llm.text_triple_extract import TripleExtract_v1
from kgpipe_llm.rdf_om import RDF_OM_v1
from kgpipe_llm.json_sa_construct import JSON_SA_Construct_v1
from kgpipe_llm.json_mapping import JSON_Mapping_v1
# from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document
from pathlib import Path
from datetime import datetime
import json

def test_llm_text_triple_extract_v1():
    """
    Test the Text Triple Extract task
    """
    text = "John Smith is a software engineer at Google."
    te_document = TripleExtract_v1.extract_te_document(text)
    
    print(te_document.model_dump_json(indent=2))

def test_llm_json_path_schema_align():
    """
    Test the JSON Path Schema Align task
    """
    input_path = Path("/home/marvin/project/data/final/film_100/split_1/sources/json/data/")
    output_file = datetime.now().strftime("%Y%m%d_%H%M%S") + "_json_path_schema_align.json"
    output_path = Path("/home/marvin/project/data/out/testing") / output_file

    JSON_SA_Construct_v1.sa_and_mapping_from_json_file_to_json_file(input_path, output_path)

def test_llm_rdf_ontology_matching_v1():
    """
    Test the RDF Ontology Matching task
    """
    input_path = Path("/home/marvin/project/data/final/film_100/split_1/sources/rdf/data.nt")
    target_path = Path("/home/marvin/project/data/final/film_100/split_0/kg/seed/data.nt")
    output_file = datetime.now().strftime("%Y%m%d_%H%M%S") + "_rdf_om.json"
    output_path = Path("/home/marvin/project/data/out/testing") / output_file
    RDF_OM_v1.rdf_om_from_files_to_er_json_file(input_path, target_path, output_path)


def test_llm_json_construct():
    """
    Test the JSON Path Schema Align task
    """
    input_path = Path("/home/marvin/project/data/final/film_100/split_1/sources/json/data/")
    output_file = datetime.now().strftime("%Y%m%d_%H%M%S") + "_json_path_schema_align.json"
    output_path = Path("/home/marvin/project/data/out/testing") / output_file

    if input_path.is_dir:
        input_path = next(input_path.iterdir())

    JSON_Mapping_v1.map_and_construct_json_file_to_rdf_file(input_path, output_path)