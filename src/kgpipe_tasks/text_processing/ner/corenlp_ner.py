"""
Stanford CoreNLP Named Entity Recognition

This module provides NER using Stanford CoreNLP.
"""

from pathlib import Path
from typing import Dict, Any

from kgpipe.common.models import KgTask, Data, DataFormat
from kgpipe.common.io import get_docker_volume_bindings, remap_data_paths_for_container
from kgpipe.execution import docker_client


CORENLP_ENTRYPOINT = ["java", "-cp", "*", "edu.stanford.nlp.pipeline.StanfordCoreNLP"]


def corenlp_kbp_extraction(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Extract KBP (Knowledge Base Population) using Stanford CoreNLP."""
    input_data = inputs["input"]
    output_data = outputs["output"]
    
    # Setup Docker
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)
    
    # Remap paths for container
    input_path = remap_data_paths_for_container([input_data], host_to_container)[0]
    output_path = remap_data_paths_for_container([output_data], host_to_container)[0]
    
    # Create command
    command = CORENLP_ENTRYPOINT + [
        "-annotators", "tokenize,pos,lemma,ner,parse,coref,kbp",
        "-file", str(input_path),
        "-outputFormat", "json",
        "-outputDirectory", str(output_path)
    ]
    
    # Run container
    client = docker_client(
        image="kgtool/corenlp:latest",
        command=command,
        volumes=volumes
    )
    client()


# Create task
corenlp_kbp_extraction_task = KgTask(
    name="corenlp_kbp_extraction",
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.CORENLP_JSON},
    function=corenlp_kbp_extraction,
    description="Extract KBP (Knowledge Base Population) using Stanford CoreNLP"
) 