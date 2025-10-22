"""
AgreementMaker Ontology Matching

This module provides ontology matching using AgreementMaker.
"""

from pathlib import Path
from typing import Dict, Any

from kgpipe.common import KgTask, Data, DataFormat, Registry
from kgpipe.common.io import get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.execution import docker_client

@Registry.task(
    input_spec={"source": DataFormat.RDF, "target": DataFormat.RDF},
    output_spec={"output": DataFormat.AGREEMENTMAKER_RDF},
    description="Perform ontology matching using Agreementmaker",
    category=["SchemaAlignment", "OntologyMatching"]
)
def agreementmaker_ontology_matching(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Perform ontology matching using AgreementMaker."""
    source_data = inputs["source"]
    target_data = inputs["target"]
    output_data = outputs["output"]
    
    # Setup Docker
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)
    
    # Remap paths for container
    source_path = remap_data_path_for_container(source_data, host_to_container)
    target_path = remap_data_path_for_container(target_data, host_to_container)
    output_path = remap_data_path_for_container(output_data, host_to_container)
    
    # Create command
    command = [
        "java", "-jar", "AgreementMakerLight.jar",
        "-a", "-s", str(source_path.path), "-t", str(target_path.path), "-o", str(output_path.path)
    ]
    
    # Run container
    client = docker_client(
        image="kgt/agreementmaker:latest",
        command=command,
        volumes=volumes
    )
    client()
