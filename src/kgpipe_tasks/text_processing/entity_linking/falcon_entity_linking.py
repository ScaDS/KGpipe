"""
FALCON Entity Linking

This module provides entity linking using FALCON.
"""

import json
import os
import requests
from pathlib import Path
from typing import Dict, Any

from kgpipe.common import KgTask, Data, DataFormat, Registry
from kgpipe.execution import docker_client


# Constants
FALCON_URL = 'http://localhost:5000/process'
HEADERS = {
    "Accept": "application/json"
}


def api_request(url: str, text: str) -> Dict[str, Any]:
    """Make API request to FALCON."""
    data = {
        "text": text
    }
    response = requests.post(url, json=data, headers=HEADERS)
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
    output_spec={"output": DataFormat.FALCON_JSON},
    description="Link entities using FALCON API",
    category=["TextProcessing", "EntityLinking", "RelationLinking"]
)
def falcon_ner_nel_rl(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Link entities using FALCON API."""
    input_data = inputs["input"]
    output_data = outputs["output"]
    
    dir_or_file = input_data.path
    if os.path.isdir(dir_or_file):
        for file in os.listdir(dir_or_file):
            with open(os.path.join(dir_or_file, file), encoding='utf-8') as f:
                input_text = f.read()

            results = api_request(FALCON_URL, input_text)

            with open(os.path.join(output_data.path, file), 'w', encoding='utf-8') as f:
                f.write(str(results))
            print(f"Converted {file} to {os.path.join(output_data.path, file)}")
    else:
        with open(input_data.path, encoding='utf-8') as f:
            input_text = f.read()

        results = api_request(FALCON_URL, input_text)

        with open(output_data.path, 'w', encoding='utf-8') as f:
            f.write(str(results))


@Registry.task(
    input_spec={"source": DataFormat.FALCON_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Convert FALCON JSON to TE JSON format",
    category=["Interopability"]
)
def falcon_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Convert FALCON JSON to TE JSON format."""
    input_path = inputs["source"].path
    output_path = outputs["output"].path

    os.makedirs(os.path.normpath(output_path), exist_ok=True)

    def __falconjson2tejson(data) -> Dict[str, Any]:
        """Convert FALCON JSON to TE Document format."""
        links = []

        for e in data.get("entities_wikidata", []):
            link = {
                "span": e.get("surface form", ""),
                "mapping": e.get("URI", ""),
                "score": 1.0,
                "link_type": "entity"
            }
            links.append(link)

        for r in data.get("relations_wikidata", []):
            link = {
                "span": r.get("surface form", ""),
                "mapping": r.get("URI", ""),
                "score": 1.0,
                "link_type": "relation"
            }
            links.append(link)

        return {"text": "", "links": links}

    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            print(file)
            with open(os.path.join(input_path, file), 'r') as f:
                data = json.load(f)
                te_doc = __falconjson2tejson(data)
                outfile = os.path.join(output_path, file)

                with open(outfile, 'w') as of:
                    json.dump(te_doc, of)
                print(f"Converted {file} to {outfile}")

    else:
        with open(input_path, 'r') as f:
            data = json.load(f)
            te_doc = __falconjson2tejson(data)
            outfile = os.path.join(output_path, 'output.te.json')
            with open(outfile, 'w') as of:
                json.dump(te_doc, of)
            print(f"Converted {input_path} to {outfile}")

