import os
from pathlib import Path
from dotenv import load_dotenv

from kgpipe.common.discovery import discover_entry_points
from kgpipe.generation.loaders import load_pipeline_catalog
from kgpipe.datasets.multipart_multisource import load_dataset, Dataset

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
    "rdf_base": "rdf",
    "rdf_alt": "rdf",
    "text_base": "text",
    "text_alt": "text",
    "json_base": "json",
    "json_alt": "json",
}

llm_pipeline_types = {
    "json_llm": "json",
    "rdf_llm": "rdf",
    "text_llm": "text",
}
