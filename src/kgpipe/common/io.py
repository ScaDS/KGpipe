# load/save helpers, graph formats

from pathlib import Path
from typing import List, Dict, Tuple
from .models import Data
import os

def get_docker_volume_bindings(data_list: List[Data], container_base: str = "/data") -> Tuple[Dict[str, Dict[str, str]], Dict[Path, Path]]:
    """
    Given a list of Data objects, return a Docker volumes dict and a mapping from host path to container path.
    - All unique parent directories are mounted to the container under /data.
    - Returns (volumes_dict, host_to_container_path_map)
    """
    volumes = {}
    host_to_container = {}
    container_base_path = Path(container_base)


    DOCKER_HOST_BIND_MAP = os.getenv("DOCKER_HOST_BIND_MAP")
    if DOCKER_HOST_BIND_MAP: 
        host_source, cont_target = DOCKER_HOST_BIND_MAP.split(":",1)
        
    def substiute_real_host(path: Path): # TODO replaces to broadly
        if DOCKER_HOST_BIND_MAP:
            print(f"SUBSTITUTING REAL DOCKER HOST PATH {cont_target} -> {host_source} for path {path}")
            return path.as_posix().replace(cont_target, host_source)
        else:
            return path
    
    # Collect all unique parent directories
    parents = {Path(d.path).resolve().parent for d in data_list}
    
    for idx, parent in enumerate(parents):
        container_mount = container_base_path / f"mount{idx}"
        volumes[substiute_real_host(parent)] = {"bind": str(container_mount), "mode": "rw"}
        # Map all files in this parent to their container path
        for d in data_list:
            if Path(d.path).resolve().parent == parent:
                host_to_container[Path(d.path).resolve()] = container_mount / Path(d.path).name

    print(f"VOLUMES {volumes}")
    print(f"HOST_TO_CONTAINER {host_to_container}")
    
    return volumes, host_to_container


# def remap_data_paths_for_container(data_list: List[Data], host_to_container: Dict[Path, Path]) -> List[Data]:
#     """
#     Given a list of Data objects and a host-to-container path map, return new Data objects with .path set to the container path.
#     """

#     DOCKER_HOST_BIND_MAP = os.getenv("DOCKER_HOST_BIND_MAP")
#     if DOCKER_HOST_BIND_MAP: 
#         host_source, cont_target = DOCKER_HOST_BIND_MAP.split(":",1)
#     def substiute_real_host(path: Path): # TODO replaces to broadly
#         if DOCKER_HOST_BIND_MAP:
#             #     print(f"SUBSTITUTING REAL DOCKER HOST PATH {cont_target} -> {host_source}")
#             return path.as_posix().replace(cont_target, host_source)
#         else:
#             return path

#     remapped = []
#     for d in data_list:
#         print(f"printing d {d}")
#         container_path = host_to_container.get(substiute_real_host(Path(d.path).resolve()), d.path)
#         remapped.append(Data(container_path, d.format))
#     return remapped

def remap_data_path_for_container(data: Data, host_to_container: Dict[Path, Path]) -> Data:
    """
    Given a Data object and a host-to-container path map, return a new Data object with .path set to the container path.
    """
    container_path = host_to_container.get(Path(data.path).resolve(), data.path)
    return Data(container_path, data.format)

class DataUtils():
    """
    Utility class for handling data formats.
    """

from pathlib import Path
from typing import Dict, List
from kgpipe.common import Data
from kgpipe.execution import docker_client


def run_docker_container_task(
    inputs: Dict[str, Data],
    outputs: Dict[str, Data],
    image: str,
    command_template: List[str],
    ensure_output_dirs: bool = True,
    **docker_kwargs
) -> str:
    """
    Generic helper function to run Docker container tasks with automatic path remapping.
    
    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
        image: Docker image name (e.g., "kgt/paris:latest")
        command_template: Command template as a list of strings. Use placeholders like
            "{source}", "{kg}", "{output}" that correspond to keys in inputs/outputs.
            These will be replaced with remapped container paths.
        ensure_output_dirs: If True, ensure output directories exist before execution
        **docker_kwargs: Additional arguments to pass to docker_client (e.g., environment, timeout)
    
    Returns:
        Container execution result
    
    Example:
        run_docker_container_task(
            inputs={"source": source_data, "kg": kg_data},
            outputs={"output": output_data},
            image="kgt/paris:latest",
            command_template=["bash", "paris.sh", "{source}", "{kg}", "{output}"]
        )
    """
    # Get all data for Docker volume bindings
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)
    
    # Remap all input paths
    remapped_inputs = {
        key: remap_data_path_for_container(data, host_to_container)
        for key, data in inputs.items()
    }
    
    # Remap all output paths
    remapped_outputs = {
        key: remap_data_path_for_container(data, host_to_container)
        for key, data in outputs.items()
    }
    
    # Ensure output directories exist
    if ensure_output_dirs:
        for output_data in outputs.values():
            output_data.path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build command by replacing placeholders
    command = []
    for part in command_template:
        # Check if it's a placeholder (e.g., "{source}", "{output}")
        if part.startswith("{") and part.endswith("}"):
            param_name = part[1:-1]  # Remove braces
            # Try inputs first, then outputs
            if param_name in remapped_inputs:
                command.append(str(remapped_inputs[param_name].path))
            elif param_name in remapped_outputs:
                command.append(str(remapped_outputs[param_name].path))
            else:
                raise ValueError(
                    f"Placeholder '{part}' not found in inputs or outputs. "
                    f"Available: inputs={list(inputs.keys())}, outputs={list(outputs.keys())}"
                )
        else:
            # Regular command part, use as-is
            command.append(part)
    
    # Create and execute Docker client
    client = docker_client(
        image=image,
        command=command,
        volumes=volumes,
        **docker_kwargs
    )
    
    result = client()
    return result