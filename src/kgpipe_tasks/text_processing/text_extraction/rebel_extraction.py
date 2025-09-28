"""
REBEL Information Extraction

This module provides information extraction using REBEL.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

from kgpipe.common.models import KgTask, Data, DataFormat
from kgpipe.common.io import get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.execution import docker_client


def rebel_extraction(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Extract triples using REBEL."""

    # Setup Docker
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)
    
    # Remap paths for container
    input_path = remap_data_path_for_container(inputs["input"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)
    
    # Create command
    command = [
        "python", "rebel_extract.py",
        "--input", str(input_path),
        "--output", str(output_path)
    ]
    
    # Run container
    client = docker_client(
        image="kgtool/rebel:latest",
        command=command,
        volumes=volumes
    )
    client()


# Create task
rebel_extraction_task = KgTask(
    name="rebel_extraction",
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.TE_JSON},
    function=rebel_extraction,
    description="Extract triples using REBEL"
) 