from kgpipe.common import TaskInput, TaskOutput, DataFormat, get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.common.registry import Registry
from kgpipe.execution import docker_client


@Registry.task(
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.TEXT},
)
def openie6_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    Openie6 information extraction task that runs in a Docker container.

    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """

    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    source_path = remap_data_path_for_container(inputs["input"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)

    outputs["output"].path.touch()

    client = docker_client(
        image="openie6:latest",
        command=["openie6.sh",
                 str(source_path.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"Openie6 completed: {result}")



