# from kgflex.framework.model import *
# from kgflex.framework.kgflex import *
# from framework.core.util import generate_docker_sdk_function
# from kgflex.resources.mainspec import *
from kgpipe.common import DataFormat, KgTask, Data, Registry
from kgpipe.common.io import get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.execution import docker_client
from typing import Dict

@Registry.task(
    input_spec={"source": DataFormat.CSV, "target": DataFormat.CSV},
    output_spec={"output": DataFormat.ER_JSON},
    description="Jedai entity matching",
    category=["EntityResolution", "Matching"]
)
def pyjedai_entity_matching(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    PyJedAI entity matching task that runs in a Docker container.
    
    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """
    print(f"Jedai entity matching with inputs: {inputs}")
    
    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    # Extract input paths
    source_path = remap_data_path_for_container(inputs["source"], host_to_container)
    target_path = remap_data_path_for_container(inputs["target"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)
    
    # Ensure output directory exists
    outputs["output"].path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all data for Docker volume bindings


    
    # Create Docker client with proper volume bindings
    client = docker_client(
        image="kgt/pyjedai:latest",
        # command=["ls", "-la"],
        command=["bash", "pyjedai.sh",
                 str(source_path.path),
                 str(target_path.path),
                 str(output_path.path)],
        volumes=volumes,
    )
    
    # Execute the container
    result = client()
    print(f"Jedai entity matching completed: {result}")


# def test_pyjedai():
#     """
#     Simple PyJedai Test
#     """

#     dm = DataMgr()
#     source = dm.fileData("/kg-testdata/_snippets/pyjedai/Abt.csv", "csv")
#     target = dm.fileData("/kg-testdata/_snippets/pyjedai/Buy.csv", "csv")

#     output = dm.fileData("/kg-testdata/pyjedai/output.json", "em_json")

#     tl = ToolLoader()
#     tools = tl.load_tools_package("kgflexflow_resources")

#     print("==doing matching==")
#     matching_task = tools["pyjedai_entity_matching"]["func"]([source, target],[output])
#     log = matching_task()
#     print(log.decode("utf-8"))