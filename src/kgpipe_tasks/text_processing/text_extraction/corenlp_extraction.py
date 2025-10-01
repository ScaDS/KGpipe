"""
Stanford CoreNLP Information Extraction

This module provides information extraction using Stanford CoreNLP.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

from kgpipe.common import KgTask, Data, DataFormat, Registry
from kgpipe.common.io import get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.execution import docker_client


CORENLP_ENTRYPOINT = ["java", "-cp", "*", "edu.stanford.nlp.pipeline.StanfordCoreNLP"]


@Registry.task(
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.OPENIE_JSON},
    description="Extract OpenIE triples using Stanford CoreNLP",
    category=["TextProcessing", "TextExtraction"]
)
def corenlp_openie_extraction(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Extract OpenIE triples using Stanford CoreNLP."""
    # input_data = inputs["input"]
    # output_data = outputs["output"]
    
    # Setup Docker
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)
    
    print(inputs["input"])
    print(outputs["output"])
    # Remap paths for container
    input_path = remap_data_path_for_container(inputs["input"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)
    
    # Create command
    command = ["bash", "openie.sh", str(input_path.path), str(output_path.path)]
    # CORENLP_ENTRYPOINT + [
    #     "-annotators", "tokenize,pos,lemma,ner,parse,coref,openie",
    #     "-file", str(input_path.path),
    #     "-outputFormat", "json",
    #     "-outputDirectory", str(output_path.path)
    # ]
    
    # Run container
    client = docker_client(
        image="kgt/corenlp:latest",
        command=command,
        volumes=volumes
    )
    client()


@Registry.task(
    input_spec={"input": DataFormat.OPENIE_JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Convert OpenIE JSON to IE JSON format",
    category=["Interopability"]
)
def corenlp_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Convert OpenIE JSON to IE JSON format."""
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    # create output folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def __openiejson2tejson(openiedata) -> Dict[str, Any]:
        """Convert OpenIE JSON to TE Document format."""
        doc = {"triples": [], "chains": []}

        # Convert to triples
        triplets = []
        for sentence in openiedata.get('sentences', []):
            for triple_span in sentence.get('openie', []):
                triplet = {
                    "subject": {"surface_form": triple_span.get('subject', '')},
                    "predicate": {"surface_form": triple_span.get('relation', '')},
                    "object": {"surface_form": triple_span.get('object', '')}
                }
                triplets.append(triplet)

        # Get chains (simplified)
        chains = get_coreference_chains(openiedata)

        doc["triples"] = triplets
        doc["chains"] = chains
        return doc

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for file in os.listdir(input_path):
            # Read input json
            with open(os.path.join(input_path, file), 'r') as f:
                data = json.load(f)
                te_doc = __openiejson2tejson(data)
                outfile = os.path.join(output_path, file)
            
                with open(outfile, 'w') as of:
                    json.dump(te_doc, of)
                # print(f"Converted {input_path} to {outfile}")
                
    else:
        # Read input json
        with open(input_path, 'r') as f:
            data = json.load(f)
            te_doc = __openiejson2tejson(data)
            with open(output_path, 'w') as of:
                json.dump(te_doc, of)
            # print(f"Converted {input_path} to {output_path}")


def get_coreference_chains(response: dict) -> List[Dict[str, Any]]:
    """Extract coreference chains from CoreNLP response."""
    result = []
    for _, coref in response.get('corefs', {}).items():
        if len(coref) > 1:
            chain = {"main": coref[0].get('text', '')}
            alias = []
            for chunk in coref[1:]:
                sentence = response.get('sentences', [])[chunk.get('sentNum', 1) - 1]
                start = sentence.get('tokens', [])[chunk.get('startIndex', 1) - 1].get('characterOffsetBegin', 0)
                end = sentence.get('tokens', [])[chunk.get('endIndex', 2) - 2].get('characterOffsetEnd', 0)
                alias.append({
                    "surface_form": chunk.get('text', ''),
                    "text": chunk.get('text', ''),
                    "start": start,
                    "end": end
                })
            chain["aliases"] = alias
            result.append(chain)
    return result

