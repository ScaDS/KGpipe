
from kgpipe.common import Data, DataFormat
from kgpipe_llm.construct.construct import (
    construct_from_any_input_as_triples, 
    construct_from_any_input_with_kg_as_triples,
    construct_from_any_input_with_kg_and_ontology_as_triples,
    construct_from_any_input_with_ontology_as_triples,
    construct_from_any_input_as_json_ld,
    construct_from_any_input_with_ontology_as_json_ld,
    construct_from_any_input_with_kg_as_json_ld,
    construct_from_any_input_with_kg_and_ontology_as_json_ld
)
from kgpipe_tasks.test import get_test_data_path

import pytest
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import os
import datetime

load_dotenv()

KG_PATH = "/home/marvin/project/data/old/acquisiton/film100_bundle/split_0/kg/seed/data.nt"
INPUT_PATH = "/home/marvin/project/data/old/acquisiton/film100_bundle/split_0/sources/json/data/"

data_kg = Data(KG_PATH, DataFormat.RDF_NTRIPLES)
data_input = Data(INPUT_PATH, DataFormat.ANY)

DEV_OUT_PATH = "/home/marvin/project/data/llm_responses"

MODEL_NAME = "gpt-5-mini"

RUN_DATE = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def get_temp_file(suffix=""):
    if DEV_OUT_PATH:
        os.makedirs(DEV_OUT_PATH, exist_ok=True)
        return Path(DEV_OUT_PATH) / f"{RUN_DATE}_{suffix}.json"
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.close()
    return Path(temp_file.name)

def clean_temp_files(output_file, temp_file):
    if not DEV_OUT_PATH:
        Path(output_file.name).unlink()
        Path(temp_file.name).unlink()

def test_construct_from_any_json_as_triples():
    
    output_file = get_temp_file("as_triples")
    data_output = Data(output_file.name , DataFormat.RDF_NTRIPLES)

    os.environ["DEFAULT_LLM_MODEL_NAME"] = MODEL_NAME

    report = construct_from_any_input_as_triples.run([data_input], [data_output], stable_files_override=True)
    print(report.model_dump_json(indent=4))

    clean_temp_files(output_file, data_output)

def test_construct_from_any_json_with_ontology_as_triples():
    
    output_file = get_temp_file("with_ontology_as_triples")
    data_output = Data(output_file.as_posix() , DataFormat.RDF_NTRIPLES)

    os.environ["DEFAULT_LLM_MODEL_NAME"] = MODEL_NAME

    report = construct_from_any_input_with_ontology_as_triples.run([data_input, data_kg], [data_output], stable_files_override=True)
    print(report.model_dump_json(indent=4))

    clean_temp_files(output_file, data_output)

def test_construct_from_any_json_with_kg_as_triples():

    output_file = get_temp_file("with_kg_as_triples")
    data_output = Data(output_file.as_posix() , DataFormat.RDF_NTRIPLES)

    os.environ["DEFAULT_LLM_MODEL_NAME"] = MODEL_NAME

    report = construct_from_any_input_with_kg_as_triples.run([data_input, data_kg], [data_output], stable_files_override=True)
    print(report.model_dump_json(indent=4))

@pytest.mark.skip(reason="Not implemented yet")
def test_construct_from_any_json_with_kg_and_ontology_as_triples():
    pass

def test_construct_from_any_json_as_json_ld():
    output_file = get_temp_file("as_json_ld")
    data_output = Data(output_file.as_posix() , DataFormat.RDF_NTRIPLES)

    os.environ["DEFAULT_LLM_MODEL_NAME"] = MODEL_NAME

    report = construct_from_any_input_as_json_ld.run([data_input, data_kg], [data_output], stable_files_override=True)
    print(report.model_dump_json(indent=4))

    clean_temp_files(output_file, data_output)

def test_construct_from_any_json_with_ontology_as_json_ld():

    output_file = get_temp_file("with_ontology_as_json_ld")
    data_output = Data(output_file.as_posix() , DataFormat.RDF_NTRIPLES)

    os.environ["DEFAULT_LLM_MODEL_NAME"] = MODEL_NAME

    report = construct_from_any_input_with_ontology_as_json_ld.run([data_input, data_kg], [data_output], stable_files_override=True)
    print(report.model_dump_json(indent=4))

def test_construct_from_any_json_with_kg_as_json_ld():

    output_file = get_temp_file("with_kg_as_json_ld")
    data_output = Data(output_file.as_posix() , DataFormat.RDF_NTRIPLES)

    os.environ["DEFAULT_LLM_MODEL_NAME"] = MODEL_NAME

    report = construct_from_any_input_with_kg_as_json_ld.run([data_input, data_kg], [data_output], stable_files_override=True)
    print(report.model_dump_json(indent=4))


@pytest.mark.skip(reason="Not implemented yet")
def test_construct_from_any_json_with_kg_and_ontology_as_json_ld():

    output_file = get_temp_file("kg_and_ontology_as_json_ld")
    data_output = Data(output_file.as_posix() , DataFormat.RDF_NTRIPLES)

    os.environ["DEFAULT_LLM_MODEL_NAME"] = MODEL_NAME

    report = construct_from_any_input_with_kg_and_ontology_as_triples.run([data_input, data_kg], [data_output], stable_files_override=True)
    print(report.model_dump_json(indent=4))

    clean_temp_files(output_file, data_output)

def test_parse_triples_to_graph():
    from kgpipe_llm.construct.construct import __parse_response_to_graph, Triples
    triples_json_path = "/home/marvin/project/data/llm_responses/20250825_223515_with_ontology_as_triples.json_response.json"
    triples = Triples(**json.load(open(triples_json_path)))

    graph = __parse_response_to_graph(triples, Triples)
    print(graph.serialize(format="nt"))
    
    