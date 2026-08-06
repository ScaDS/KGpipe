import csv
import json
import os
import re
from typing import Dict, Any

from rdflib import Graph, URIRef, Literal, BNode

from kgpipe.common import TaskInput, TaskOutput, DataFormat, get_docker_volume_bindings, remap_data_path_for_container, \
    Data
from kgpipe.common.registry import Registry
from kgpipe.execution import docker_client


def process_io(input_path, output_path, process_file_fn, extension):
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)

        for filename in os.listdir(input_path):
            input_file = os.path.join(input_path, filename)

            if not os.path.isfile(input_file):
                continue

            output_file = os.path.join(
                output_path,
                filename + extension
            )

            process_file_fn(input_file, output_file)

    else:
        process_file_fn(input_path, output_path)


@Registry.task(
    input_spec={"input1": DataFormat.CSV, "input2": DataFormat.CSV},
    output_spec={"output": DataFormat.ER_JSON},
)
def valentine_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    Valentine task that runs in a Docker container.

    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """

    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    source_path1 = remap_data_path_for_container(inputs["input1"], host_to_container)
    source_path2 = remap_data_path_for_container(inputs["input2"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)

    client = docker_client(
        image="kgt/valentine",
        command=["sh",
                 "valentine.sh",
                 str(source_path1.path),
                 str(source_path2.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"Valentine completed: {result}")

@Registry.task(
    input_spec={"input1": DataFormat.RDF, "input2": DataFormat.RDF},
    output_spec={"output": DataFormat.ER_JSON},
)
def paris_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    PARIS task that runs in a Docker container.

    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """

    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    source_path1 = remap_data_path_for_container(inputs["input1"], host_to_container)
    source_path2 = remap_data_path_for_container(inputs["input2"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)

    client = docker_client(
        image="kgt/paris",
        command=["sh",
                 "paris.sh",
                 str(source_path1.path),
                 str(source_path2.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"PARIS completed: {result}")

@Registry.task(
    input_spec={"input1": DataFormat.CSV, "input2": DataFormat.CSV, "gt": DataFormat.CSV, "config": DataFormat.ANY},
    output_spec={"output": DataFormat.ER_JSON},
)
def pyjedai_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    PyJedAI task that runs in a Docker container.

    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """

    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    source_path1 = remap_data_path_for_container(inputs["input1"], host_to_container)
    source_path2 = remap_data_path_for_container(inputs["input2"], host_to_container)
    gt_path = remap_data_path_for_container(inputs["gt"], host_to_container)
    config_path = remap_data_path_for_container(inputs["config"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)

    client = docker_client(
        image="kgt/pyjedai",
        command=["sh",
                 "pyjedai.sh",
                 str(source_path1.path),
                 str(source_path2.path),
                 str(output_path.path),
                 str(config_path.path),
                 str(gt_path.path),
                 ",",],
        volumes=volumes,
    )

    result = client()
    print(f"pyjedai completed: {result}")


@Registry.task(
    input_spec={"input1": DataFormat.CSV, "input2": DataFormat.CSV},
    output_spec={"output": DataFormat.ER_JSON},
)
def deepmatcher_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    DeepMatcher task that runs in a Docker container.

    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """

    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    source_path1 = remap_data_path_for_container(inputs["input1"], host_to_container)
    source_path2 = remap_data_path_for_container(inputs["input2"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)

    client = docker_client(
        image="kgt/deepmatcher",
        command=["sh",
                 "deepmatcher.sh",
                 str(source_path1.path),
                 str(source_path2.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"deepmatcher completed: {result}")

@Registry.task(
    input_spec={"input1": DataFormat.RDF, "input2": DataFormat.RDF},
    output_spec={"output": DataFormat.ER_JSON},
)
def agreementmaker_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    Openie6 information extraction task that runs in a Docker container.

    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """

    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    source_path1 = remap_data_path_for_container(inputs["input1"], host_to_container)
    source_path2 = remap_data_path_for_container(inputs["input2"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)

    client = docker_client(
        image="kgt/agreementmaker",
        command=["sh",
                 "agreementmaker.sh",
                 str(source_path1.path),
                 str(source_path2.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"agreementmaker completed: {result}")

@Registry.task(
    input_spec={"input1": DataFormat.RDF_NTRIPLES, "input2": DataFormat.ANY},
    output_spec={"output": DataFormat.ER_JSON},
)
def ontoea_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    OntoEA task that runs in a Docker container.

    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """

    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    source_path1 = remap_data_path_for_container(inputs["input1"], host_to_container)
    source_path2 = remap_data_path_for_container(inputs["input2"], host_to_container)

    client = docker_client(
        image="kgt/ontoea",
        command=["sh",
                 "ontoea.sh",
                 str(source_path1.path),
                 str(source_path2.path)],
        volumes=volumes,
    )

    result = client()
    print(f"ontoea completed: {result}")

@Registry.task(
    input_spec={"input1": DataFormat.RDF, "input2": DataFormat.RDF},
    output_spec={"output": DataFormat.ER_JSON},
)
def ontoaligner_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    OntoAligner information extraction task that runs in a Docker container.

    Args:
        inputs: Dictionary mapping input names to Data objects
        outputs: Dictionary mapping output names to Data objects
    """

    all_data = list(inputs.values()) + list(outputs.values())
    volumes, host_to_container = get_docker_volume_bindings(all_data)

    source_path1 = remap_data_path_for_container(inputs["input1"], host_to_container)
    source_path2 = remap_data_path_for_container(inputs["input2"], host_to_container)
    output_path = remap_data_path_for_container(outputs["output"], host_to_container)

    client = docker_client(
        image="kgt/ontoaligner",
        command=["sh",
                 "ontoaligner.sh",
                 str(source_path1.path),
                 str(source_path2.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"ontoaligner completed: {result}")



def _get_sf(triple, key):
    return triple.get(key, {}).get("surface_form", "").strip()


def _is_valid_triple(triple):
    return all([
        _get_sf(triple, "subject"),
        _get_sf(triple, "predicate"),
        _get_sf(triple, "object"),
    ])


def _extract_row(triple):
    if not _is_valid_triple(triple):
        return None
    return [
        _get_sf(triple, "subject"),
        _get_sf(triple, "predicate"),
        _get_sf(triple, "object"),
    ]


def _sort_key(triple):
    return (
        _get_sf(triple, "subject").lower(),
        _get_sf(triple, "predicate").lower(),
        _get_sf(triple, "object").lower(),
    )

def _load_te_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_sorted_triples(te_json):
    triples = te_json.get("triples", [])
    return sorted(triples, key=_sort_key)