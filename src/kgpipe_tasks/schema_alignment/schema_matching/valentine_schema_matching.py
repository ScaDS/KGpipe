"""
Valentine Schema Matching

This module provides schema matching using Valentine.
"""

from pathlib import Path
from typing import Dict, Any

from kgpipe.common import KgTask, Data, DataFormat, Registry
from kgpipe.common.io import get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.execution import docker_client

@Registry.task(
    input_spec={"source": DataFormat.CSV, "target": DataFormat.CSV},
    output_spec={"output": DataFormat.ER_JSON},
    description="Perform schema matching using Valentine",
    category=["SchemaAlignment", "SchemaMatching"]
)
def valentine_csv_matching(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Perform schema matching using Valentine."""
    # source_data = inputs["source"]
    # target_data = inputs["target"]
    # output_data = outputs["output"]
    
    # Setup Docker
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)
    
    # Remap paths for container
    source_path = remap_data_path_for_container(inputs["source"], host_to_container)
    target_path = remap_data_path_for_container(inputs["target"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)
    
    # Create command
    command = [
        "bash", "valentine.sh",
        str(source_path.path), str(target_path.path), str(output_path.path)
    ]
    
    # Run container
    client = docker_client(
        image="kgt/valentine:latest",
        command=command,
        volumes=volumes
    )
    client()


@Registry.task(
    input_spec={"source": DataFormat.CSV, "target": DataFormat.CSV},
    output_spec={"output": DataFormat.ER_JSON},
    description="Perform schema matching using Valentine",
    category=["SchemaAlignment", "SchemaMatching"]
)
def valentine_csv_matching_v2(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Perform schema matching using Valentine."""
    # source_data = inputs["source"]
    # target_data = inputs["target"]
    # output_data = outputs["output"]
    
    # Setup Docker
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)
    
    # Remap paths for container
    source_path = remap_data_path_for_container(inputs["source"], host_to_container)
    target_path = remap_data_path_for_container(inputs["target"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)
    
    # Create command
    command = [
        "bash", "valentine_v2.sh",
        str(source_path.path), str(target_path.path), str(output_path.path), "500"
    ]
    
    # Run container
    client = docker_client(
        image="kgt/valentine:latest",
        command=command,
        volumes=volumes
    )
    client()