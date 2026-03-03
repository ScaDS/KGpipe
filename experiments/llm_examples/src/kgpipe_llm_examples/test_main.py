
# TODO current impls

from kgpipe.common import KgPipe, Data, DataFormat
from kgpipe_tasks.text_processing import label_alias_embedding_rl, dbpedia_spotlight_ner_nel, dbpedia_spotlight_exchange
from kgpipe_tasks.transform_interop import aggregate3_te_json
# from text_pipelines.text_tasks import openie6_task_docker, graphene_nt_exchange, graphene_task_docker, \
#     minie_task_docker, minie_exchange, imojie_task_docker, imojie_exchange, genie_task_docker, genie_exchange

import tempfile
import shutil
from typing import List
from kgpipe.common.model.data import DataFormat


# TODO collect prompts
# TODO e2e eval
# TODO add OR to kgpipe_llm
# https://openrouter.ai/docs/guides/features/structured-outputs
import hashlib
import os
from rdflib import Graph
from kgpipe_llm_examples.llm_text import text_extraction_direct
from kgpipe_llm_examples.llm_semi import test_mapping_direct

INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"

models = [
    "openai/gpt-oss-120b:nitro",
    "openai/gpt-5-mini"
    "openai/gpt-4o-mini",
    # # "llama-3.3-70b-versatile",
    "meta-llama/llama-3-70b-instruct"
    "qwen/qwen3-30b-a3b-instruct-2507"
]


def test_mapping():
    JSON_INPUT_DIR = os.path.join(INPUTS_DIR, "json")
    JSON_TEST_FILE = os.path.join(JSON_INPUT_DIR, "6eda025127bbacf5ed81d858c7c246d4.json")
    inputs: List[Data] = [Data(path=JSON_TEST_FILE, format=DataFormat.JSON)]
    RDF_OUTPUT_FILE = os.path.join(OUTPUTS_DIR, "6eda025127bbacf5ed81d858c7c246d4.nt")
    outputs: List[Data] = [Data(path=RDF_OUTPUT_FILE, format=DataFormat.RDF_NTRIPLES)]
    test_mapping_direct.run(inputs=inputs, outputs=outputs)

def test_text():
    inputs: List[Data] = [Data(path="input.txt", format=DataFormat.TEXT)]
    outputs: List[Data] = [Data(path="output.json", format=DataFormat.TE_JSON)]
    text_extraction_direct.run(inputs=inputs, outputs=outputs)

from kgpipe_llm_examples.llm_python import mapping_python_target

def test_mapping_python():
    SAMPLE_FILE_NAME="d941fd0b3826e9b284c3901a86d1f945.json"
    JSON_INPUT_DIR = os.path.join(INPUTS_DIR, "json")
    JSON_TEST_FILE = os.path.join(JSON_INPUT_DIR, SAMPLE_FILE_NAME)
    inputs: List[Data] = [Data(path=JSON_TEST_FILE, format=DataFormat.JSON)]
    RDF_OUTPUT_FILE = os.path.join(OUTPUTS_DIR, f"{SAMPLE_FILE_NAME}_1.nt")
    outputs: List[Data] = [Data(path=RDF_OUTPUT_FILE, format=DataFormat.RDF_NTRIPLES)]
    mapping_python_target.run(inputs=inputs, outputs=outputs)