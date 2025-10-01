from kgpipe.common.discovery import discover_entry_points
from kgpipe.generation.loaders import load_pipeline_catalog
from pyodibel.datasets.mp_mf.multipart_multisource import load_dataset, Dataset
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
discover_entry_points()

PIPELINE_CONFIG=os.getenv("PIPELINE_CONFIG")
ONTOLOGY_PATH=os.getenv("ONTOLOGY_PATH")
OUTPUT_DIR=os.getenv("OUTPUT_DIR")

DATASET_SMALL_DIR=os.getenv("DATASET_SMALL")
DATASET_MEDIUM_DIR=os.getenv("DATASET_MEDIUM")
DATASET_LARGE_DIR=os.getenv("DATASET_LARGE")
DATASET_SELECT=os.getenv("DATASET_SELECT")

if not PIPELINE_CONFIG:
    raise ValueError("MISSING PIPELINE CONFIG")
catalog = load_pipeline_catalog(Path(PIPELINE_CONFIG)) # TODO os.getenv

if not ONTOLOGY_PATH:
    raise ValueError("MISSING ONTOLOGY PATH")

if not DATASET_SELECT:
    raise ValueError("MISSING DATASET SELECT")

if DATASET_SELECT == "small" and DATASET_SMALL_DIR:
    dataset = load_dataset(Path(DATASET_SMALL_DIR))
elif DATASET_SELECT == "medium" and DATASET_MEDIUM_DIR:
    dataset = load_dataset(Path(DATASET_MEDIUM_DIR))
elif DATASET_SELECT == "large" and DATASET_LARGE_DIR:
    dataset = load_dataset(Path(DATASET_LARGE_DIR))
else:
    raise ValueError("INVALID DATASET SELECT")

if not OUTPUT_DIR:
    raise ValueError("MISSING OUTPUT DIRECTORY")
OUTPUT_ROOT = Path(OUTPUT_DIR) / DATASET_SELECT


pipeline_types = {
    "rdf_a": "rdf",
    "rdf_b": "rdf",
    "text_a": "text",
    "text_b": "text",
    "json_a": "json",
    "json_b": "json"
}

llm_pipeline_types = {
    "json_llm_mapping_v1": "json",
    "rdf_llm_schema_align_v1": "rdf",
    "text_llm_triple_extract_v1": "text",
}

ssp = {
    "rdf": "rdf_a",
    "json": "json_b",
    "text": "text_a"
}




# load and override from env or set here
