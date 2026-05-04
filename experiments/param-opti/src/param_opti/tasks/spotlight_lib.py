"""
DBpedia Spotlight Entity Linking

This module provides entity linking using DBpedia Spotlight.
"""

import json
import os
import requests
from pathlib import Path
from typing import Dict, Any

from kgpipe.common import KgTask, Data, DataFormat, Registry
from kgpipe.common.io import get_docker_volume_bindings
from kgpipe.execution import docker_client
from tqdm import tqdm

import os


CONFIDENCE = 0.35
HEADERS = {
    "Accept": "application/json"
}


def api_request(url: str, text: str) -> Dict[str, Any]:
    """Make API request to DBpedia Spotlight."""
    data = {
        "text": text,
        "confidence": str(CONFIDENCE)
    }
    response = requests.post(url, data=data, headers=HEADERS, verify=False)

    if response.status_code == 200:
        result = response.json()
    else:
        result = {
            "error": f"Request failed with status code {response.status_code}",
            "text": text
        }
    return result


@Registry.task(
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.SPOTLIGHT_JSON},
    description="Link entities using DBpedia Spotlight API",
    category=["TextProcessing", "EntityLinking"]
)
def dbpedia_spotlight_ner_nel(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Link entities using DBpedia Spotlight API."""
    input_data = inputs["input"]
    output_data = outputs["output"]

    DBPEDIA_ANNOTATE_URL = os.getenv("DBPEDIA_ANNOTATE_URL")
    if not DBPEDIA_ANNOTATE_URL:
        raise ValueError("Missing DBpedia ANnotate URL")
    
    dir_or_file = input_data.path
    if os.path.isdir(dir_or_file):
        os.makedirs(output_data.path, exist_ok=True)
        for file in tqdm(os.listdir(dir_or_file)):
            with open(os.path.join(dir_or_file, file), encoding='utf-8') as f:
                input_text = f.read()

            results = api_request(DBPEDIA_ANNOTATE_URL, input_text)

            with open(os.path.join(output_data.path, file+".json"), 'w', encoding='utf-8') as f:
                f.write(json.dumps(results))
            # print(f"Converted {file} to {os.path.join(output_data.path, file)}")
    else:
        with open(input_data.path, encoding='utf-8') as f:
            input_text = f.read()

        results = api_request(DBPEDIA_ANNOTATE_URL, input_text)

        with open(output_data.path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results))


@Registry.task(
    input_spec={"source": DataFormat.SPOTLIGHT_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Convert Spotlight JSON to TE JSON format",
    category=["TextProcessing", "EntityLinking"]
)
def dbpedia_spotlight_exchange_filtered(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Convert Spotlight JSON to TE JSON format."""
    input_path = inputs["source"].path
    output_path = outputs["output"].path

    

    # create output folder
    os.makedirs(os.path.normpath(output_path), exist_ok=True)

    def __spotlightjson2tejson(data) -> Dict[str, Any]:
        """Convert Spotlight JSON to TE Document format."""
        links = []

        for result in data.get('Resources', []):
            link = {
                "span": result.get('@surfaceForm', ''),
                "mapping": result.get('@URI', ''),
                "score": float(result.get('@similarityScore', 0.0)),
                "link_type": "entity"
            }
            links.append(link)

        text = data.get('@text', '')
        return {"text": text, "links": links}

    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            # Read input json
            with open(os.path.join(input_path, file), 'r') as f:
                data = json.load(f)
                te_doc = __spotlightjson2tejson(data)
                outfile = os.path.join(output_path, file)
            
                with open(outfile, 'w') as of:
                    json.dump(te_doc, of)
                # print(f"Converted {file} to {outfile}")
                
    else:
        # Read input json
        with open(input_path, 'r') as f:
            data = json.load(f)
            te_doc = __spotlightjson2tejson(data)
            outfile = os.path.join(output_path, 'output.te.json')
            with open(outfile, 'w') as of:
                json.dump(te_doc, of)
            # print(f"Converted {input_path} to {output_path}")


@Registry.task(
    input_spec={"source": DataFormat.SPOTLIGHT_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Convert Spotlight JSON to TE JSON format, with seed filter",
    category=["TextProcessing", "EntityLinking"]
)
def dbpedia_spotlight_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Convert Spotlight JSON to TE JSON format."""
    input_path = inputs["source"].path
    output_path = outputs["output"].path

    # create output folder
    os.makedirs(os.path.normpath(output_path), exist_ok=True)

    def __spotlightjson2tejson(data) -> Dict[str, Any]:
        """Convert Spotlight JSON to TE Document format."""
        links = []

        for result in data.get('Resources', []):
            link = {
                "span": result.get('@surfaceForm', ''),
                "mapping": result.get('@URI', ''),
                "score": float(result.get('@similarityScore', 0.0)),
                "link_type": "entity"
            }
            links.append(link)

        text = data.get('@text', '')
        return {"text": text, "links": links}

    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            # Read input json
            with open(os.path.join(input_path, file), 'r') as f:
                data = json.load(f)
                te_doc = __spotlightjson2tejson(data)
                outfile = os.path.join(output_path, file)
            
                with open(outfile, 'w') as of:
                    json.dump(te_doc, of)
                # print(f"Converted {file} to {outfile}")
                
    else:
        # Read input json
        with open(input_path, 'r') as f:
            data = json.load(f)
            te_doc = __spotlightjson2tejson(data)
            outfile = os.path.join(output_path, 'output.te.json')
            with open(outfile, 'w') as of:
                json.dump(te_doc, of)
                