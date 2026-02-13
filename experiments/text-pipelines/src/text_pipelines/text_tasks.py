import json
import os
import re
from typing import Dict, Any

from rdflib import Graph, URIRef, Literal, BNode

from kgpipe.common import TaskInput, TaskOutput, DataFormat, get_docker_volume_bindings, remap_data_path_for_container, \
    Data
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


@Registry.task(
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
)
def graphene_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    Graphene information extraction task that runs in a Docker container.

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
        image="graphene:latest",
        command=["graphene.sh",
                 str(source_path.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"Graphene completed: {result}")


@Registry.task(
    input_spec={"input": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.TE_JSON},
    description="Convert RDF NTriples to TE JSON format",
    category=["Interopability"]
)
def graphene_nt_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """Convert Graphene RDF NTriples to TE JSON format."""
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    # create output folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def __graphenent2tejson(rdfntdata) -> Dict[str, Any]:
        """Convert Graphene RDF NTriples to TE Document format."""

        GRAPHENE_SUBJ = URIRef("http://lambda3.org/graphene/extraction#subject")
        GRAPHENE_PRED = URIRef("http://lambda3.org/graphene/extraction#predicate")
        GRAPHENE_OBJ = URIRef("http://lambda3.org/graphene/extraction#object")

        RDF_VALUE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#value")

        doc = {"triples": [], "chains": []}

        g = Graph()
        g.parse(data=rdfntdata, format="nt")

        value_map = {}
        for s, p, o in g.triples((None, RDF_VALUE, None)):
            if isinstance(o, Literal):
                value_map[s] = str(o)

        extractions = {}
        for s, p, o in g:
            if isinstance(s, BNode):
                extractions.setdefault(s, {})[p] = o

        triples = []
        for extraction, props in extractions.items():
            if (GRAPHENE_SUBJ in props and
                GRAPHENE_PRED in props and
                GRAPHENE_OBJ in props):
                triples.append({
                    "subject": {"surface_form": value_map.get(props[GRAPHENE_SUBJ], "")},
                    "predicate": {"surface_form": value_map.get(props[GRAPHENE_PRED], "")},
                    "object": {"surface_form": value_map.get(props[GRAPHENE_OBJ], "")}
                })

        doc["triples"] = triples
        doc["chains"] = []
        return doc

    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for file in os.listdir(input_path):
            # Read input nt file
            with open(os.path.join(input_path, file), 'r') as f:
                data = f.read()
                te_doc = __graphenent2tejson(data)
                outfile = os.path.join(output_path, file)

                with open(outfile, 'w') as of:
                    json.dump(te_doc, of)
                # print(f"Converted {input_path} to {outfile}")

    else:
        # Read input nt file
        with open(input_path, 'r') as f:
            data = f.read()
            te_doc = __graphenent2tejson(data)
            with open(output_path, 'w') as of:
                json.dump(te_doc, of)
            # print(f"Converted {input_path} to {output_path}")


@Registry.task(
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.ANY},
)
def minie_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    MinIE information extraction task that runs in a Docker container.

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
        image="minie:latest",
        command=["minie.sh",
                 str(source_path.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"MinIE completed: {result}")


@Registry.task(
    input_spec={"input": DataFormat.ANY},
    output_spec={"output": DataFormat.TE_JSON},
    description="Convert MinIE output to TE JSON format",
    category=["Interopability"]
)
def minie_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    triples = []
    chains = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Triple:"):
            parts = line[len("Triple:"):].strip().split(" | ")
            if len(parts) == 3:
                triple = {
                    "subject": {"surface_form": parts[0].strip()},
                    "predicate": {"surface_form": parts[1].strip()},
                    "object": {"surface_form": parts[2].strip()}
                }
                triples.append(triple)

    output_json = {
        "triples": triples,
        "chains": chains
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)


@Registry.task(
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.ANY},
)
def imojie_task_docker(inputs: TaskInput, outputs: TaskOutput):
    """
    Imojie information extraction task that runs in a Docker container.

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
        image="imojie:latest",
        command=["imojie.sh",
                 str(source_path.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"Imojie completed: {result}")


@Registry.task(
    input_spec={"input": DataFormat.ANY},
    output_spec={"output": DataFormat.TE_JSON},
    description="Convert Imojie output to TE JSON format",
    category=["Interopability"]
)
def imojie_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    triples = []
    chains = []

    triple_pattern = re.compile(r'\(\s*(.*?)\s*;\s*(.*?)\s*;\s*(.*?)\s*\)')

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            matches = triple_pattern.findall(line)
            for subj, pred, obj in matches:
                triple = {
                    "subject": {"surface_form": subj.strip()},
                    "predicate": {"surface_form": pred.strip()},
                    "object": {"surface_form": obj.strip()}
                }
                triples.append(triple)

    output_json = {
        "triples": triples,
        "chains": chains
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

@Registry.task(
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.JSON},
)
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

    outputs["output"].path.touch()

    client = docker_client(
        image="genie:latest",
        command=["genie.sh",
                 str(source_path.path),
                 str(output_path.path)],
        volumes=volumes,
    )

    result = client()
    print(f"GenIE completed: {result}")


@Registry.task(
    input_spec={"input": DataFormat.JSON},
    output_spec={"output": DataFormat.TE_JSON},
    description="Convert GenIE output to TE JSON format",
    category=["Interopability"]
)
def genie_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    triples = []
    chains = []

    triple_pattern = re.compile(r"<sub>\s*(.*?)\s*<rel>\s*(.*?)\s*<obj>\s*(.*?)\s*<et>")

    with open(input_path, "r", encoding="utf-8") as f:
        genie_output = json.load(f)

    for sentence_outputs in genie_output:
        for beam in sentence_outputs:
            text = beam.get("text", "")

            matches = triple_pattern.findall(text)

            for subj, pred, obj in matches:
                triple = {
                    "subject": {"surface_form": subj.strip()},
                    "predicate": {"surface_form": pred.strip()},
                    "object": {"surface_form": obj.strip()}
                }
                triples.append(triple)

    output_json = {
        "triples": triples,
        "chains": chains
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)