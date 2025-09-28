"""
Docker execution utilities for KGbench.

This module provides utilities for running Docker containers in KGbench tasks.
"""

import docker
from docker.errors import ContainerError, ImageNotFound
from pathlib import Path
from typing import Dict, Any, List, Optional


def docker_client(
    image: str, 
    command: List[str], 
    volumes: Optional[Dict[str, Dict[str, str]]] = None,
    environment: Optional[Dict[str, str]] = None,
    working_dir: str = "/data",
    timeout: int = 3600,
    entrypoint: Optional[str] = None
) -> Any:
    """
    Create and return a Docker client for running containerized tasks.
    
    Args:
        image: Docker image name
        command: Command to run in the container
        volumes: Volume mappings for the container
        environment: Environment variables for the container
        working_dir: Working directory in the container
        timeout: Timeout in seconds for container execution
    
    Returns:
        A function that when called executes the Docker container
        
    Raises:
        RuntimeError: If Docker execution fails
    """
    def run_container():
        """Run the Docker container with the specified configuration."""
        try:
            client = docker.from_env()
            
            # Prepare container configuration
            container_config = {
                "image": image,
                "command": command,
                # "working_dir": working_dir,
                "detach": False,
                "remove": True,
                # "timeout": timeout
            }
            
            if entrypoint:
                container_config["entrypoint"] = entrypoint

            if volumes:
                container_config["volumes"] = volumes
            
            if environment:
                container_config["environment"] = environment
            
            # Run the container
            result = client.containers.run(**container_config)
            
            # Return the output
            if hasattr(result, 'decode'):
                return result.decode('utf-8')
            else:
                return str(result)
                
        except ImageNotFound:
            print(f"Docker image '{image}' not found. Please build or pull the image.")
            raise RuntimeError(f"Docker image '{image}' not found. Please build or pull the image.")
        except ContainerError as e:
            print(f"Container execution failed: {e.stderr}")
            raise RuntimeError(f"Container execution failed: {e}")
        except Exception as e:
            print(f"Docker execution failed: {e}")
            raise RuntimeError(f"Docker execution failed: {e}")
    
    print(f"Running container: {image} with volumes: {volumes} and command: {" ".join(command)}")
    return run_container


def run_docker_task(
    image: str,
    command: List[str],
    inputs: List[Path],
    outputs: List[Path],
    volumes: Optional[Dict[str, Dict[str, str]]] = None,
    environment: Optional[Dict[str, str]] = None,
    working_dir: str = "/data",
    timeout: int = 3600
) -> str:
    """
    Run a Docker task with input and output file handling.
    
    Args:
        image: Docker image name
        command: Command to run in the container
        inputs: List of input file paths
        outputs: List of output file paths
        volumes: Additional volume mappings
        environment: Environment variables
        working_dir: Working directory in the container
        timeout: Timeout in seconds
    
    Returns:
        Container output as string
    """
    # Create volume bindings for all files
    all_files = inputs + outputs
    file_volumes = {}
    
    for file_path in all_files:
        if file_path.exists():
            # Bind the parent directory to allow the container to access the file
            parent_dir = str(file_path.parent.absolute())
            file_volumes[parent_dir] = {
                "bind": parent_dir,
                "mode": "rw"
            }
    
    # Merge with additional volumes
    if volumes:
        file_volumes.update(volumes)
    
    # Create and run the Docker client
    client = docker_client(
        image=image,
        command=command,
        volumes=file_volumes,
        environment=environment,
        working_dir=working_dir,
        timeout=timeout
    )
    
    return client() 