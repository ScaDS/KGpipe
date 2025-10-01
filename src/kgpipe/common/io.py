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
            #     print(f"SUBSTITUTING REAL DOCKER HOST PATH {cont_target} -> {host_source}")
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
