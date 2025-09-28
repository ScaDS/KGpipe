"""
LIMES Ontology Matching

This module provides ontology matching using LIMES.
"""

from pathlib import Path
from typing import Dict, Any

from kgpipe.common.models import KgTask, Data, DataFormat
from kgpipe.common.io import get_docker_volume_bindings, remap_data_paths_for_container
from kgpipe.execution import docker_client


def limes_rdf_matching(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Perform RDF matching using LIMES."""
    input1_data = inputs["input1"]
    input2_data = inputs["input2"]
    output_data = outputs["output"]
    
    # Setup Docker
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)
    
    # Remap paths for container
    input1_path = remap_data_paths_for_container([input1_data], host_to_container)[0]
    input2_path = remap_data_paths_for_container([input2_data], host_to_container)[0]
    output_path = remap_data_paths_for_container([output_data], host_to_container)[0]
    
    # Create command
    command = [
        "limes", "rdf-matching",
        "--input1", str(input1_path),
        "--input2", str(input2_path),
        "--output", str(output_path)
    ]
    
    # Run container
    client = docker_client(
        image="kgtool/limes:latest",
        command=command,
        volumes=volumes
    )
    client()


# Create task
limes_rdf_matching_task = KgTask(
    name="limes_rdf_matching",
    input_spec={"input1": DataFormat.RDF, "input2": DataFormat.RDF},
    output_spec={"output": DataFormat.LIMES_XML},
    function=limes_rdf_matching,
    description="Perform RDF matching using LIMES"
) 