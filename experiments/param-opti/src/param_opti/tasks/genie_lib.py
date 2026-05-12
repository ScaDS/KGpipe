
import re
import json
import os
from typing import Dict
from kgpipe.common import Data, TaskInput, TaskOutput
from kgpipe.common import KgTask, DataFormat, Data, Registry, TaskInput, TaskOutput
from kgpipe.common.io import get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.execution import docker_client
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Match, ER_Document


def genie_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    GenIE information extraction task that runs in a Docker container.

    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """

    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    source_path = remap_data_path_for_container(inputs["input"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)

    client = docker_client(
        image="genie:latest",
        command=["genie.sh",
                 str(source_path.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"GenIE completed: {result}")

def process_io(input_path, output_path, process_file_fn, extension):
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)

        for filename in os.listdir(input_path):
            input_file = os.path.join(input_path, filename)

            if not os.path.isfile(input_file):
                continue

            output_file = os.path.join(
                output_path,
                os.path.splitext(filename)[0] + extension
            )

            process_file_fn(input_file, output_file)

    else:
        process_file_fn(input_path, output_path)

def genie_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    triple_pattern = re.compile(
        r"<sub>\s*(.*?)\s*<rel>\s*(.*?)\s*<obj>\s*(.*?)\s*<et>"
    )

    def exchange_file(input_file, output_file):
        triples = []
        chains = []

        with open(input_file, "r", encoding="utf-8") as f:
            genie_output = json.load(f)

        for sentence in genie_output:
            for beam in sentence:
                text = beam.get("text", "")
                matches = triple_pattern.findall(text)

                for subj, pred, obj in matches:
                    triples.append({
                        "subject": {"surface_form": subj.strip()},
                        "predicate": {"surface_form": pred.strip()},
                        "object": {"surface_form": obj.strip()}
                    })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"triples": triples, "chains": chains}, f, indent=2)

    process_io(input_path, output_path, exchange_file, ".te.json")