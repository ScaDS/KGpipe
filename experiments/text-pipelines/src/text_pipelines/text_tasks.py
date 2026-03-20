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
                os.path.splitext(filename)[0] + extension
            )

            process_file_fn(input_file, output_file)

    else:
        process_file_fn(input_path, output_path)


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
                outfile = os.path.join(
                    output_path,
                    os.path.splitext(file)[0] + ".te.json"
                )
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

    def exchange_file(input_file, output_file):
        triples = []
        chains = []

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("Triple:"):
                    continue

                parts = [p.strip() for p in line[len("Triple:"):].split("|")]

                if len(parts) != 3:
                    continue

                subject, predicate, object_ = parts

                triples.append({
                    "subject": {"surface_form": subject},
                    "predicate": {"surface_form": predicate},
                    "object": {"surface_form": object_}
                })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"triples": triples, "chains": chains}, f, indent=2)

    process_io(input_path, output_path, exchange_file, ".te.json")


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

    triple_pattern = re.compile(
        r'\(\s*([^;]+?)\s*;\s*([^;]+?)\s*;\s*([^;]+?)\s*\)'
    )

    def make_triple(subj, pred, obj):
        return {
            "subject": {"surface_form": subj.strip()},
            "predicate": {"surface_form": pred.strip()},
            "object": {"surface_form": obj.strip()}
        }

    def exchange_file(input_file, output_file):
        triples = []
        chains = []

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                matches = triple_pattern.findall(line)

                for subj, pred, obj in matches:
                    triples.append(make_triple(subj, pred, obj))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"triples": triples, "chains": chains}, f, indent=2)

    process_io(input_path, output_path, exchange_file, ".te.json")

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

@Registry.task(
    input_spec={"input": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.CSV},
    description="Convert TE_JSON triples to csv",
    category=["Interopability"]
)
def te_json_triple_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    def sort_key(t):
        return (
            t.get("subject", {}).get("surface_form", "").lower(),
            t.get("predicate", {}).get("surface_form", "").lower(),
            t.get("object", {}).get("surface_form", "").lower(),
        )

    def extract_row(triple):
        subject = triple.get("subject", {}).get("surface_form", "").strip()
        predicate = triple.get("predicate", {}).get("surface_form", "").strip()
        object_ = triple.get("object", {}).get("surface_form", "").strip()

        if not (subject and predicate and object_):
            return None

        return [subject, predicate, object_]

    def exchange_file(input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            te_json = json.load(f)

        triples = te_json.get("triples", [])
        triples_sorted = sorted(triples, key=sort_key)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "predicate", "object"])

            for triple in triples_sorted:
                row = extract_row(triple)
                if row:
                    writer.writerow(row)

    process_io(input_path, output_path, exchange_file, ".csv")

@Registry.task(
    input_spec={"input": DataFormat.TE_JSON},
    output_spec={"output": DataFormat.CSV},
    description="Convert linked TE_JSON triples to csv",
    category=["Interopability"]
)
def linked_te_json_triple_exchange(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    def get_sf(triple, key):
        return triple.get(key, {}).get("surface_form", "").strip()

    def sort_key(t):
        return (
            get_sf(t, "subject").lower(),
            get_sf(t, "predicate").lower(),
            get_sf(t, "object").lower(),
        )

    def exchange_file(input_file, output_file):
        with open(input_file, "r", encoding="utf-8") as f:
            te_json = json.load(f)

        triples = te_json.get("triples", [])
        global_links = te_json.get("links", [])

        span_to_mapping = {
            link.get("span"): link.get("mapping")
            for link in global_links
            if link.get("span") and link.get("mapping")
        }

        triples_sorted = sorted(triples, key=sort_key)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "subject", "predicate", "object",
                "subject_link", "predicate_link", "object_link"
            ])

            for triple in triples_sorted:
                subject_sf = get_sf(triple, "subject")
                predicate_sf = get_sf(triple, "predicate")
                object_sf = get_sf(triple, "object")

                # skip kaputte triples
                if not (subject_sf and predicate_sf and object_sf):
                    continue

                writer.writerow([
                    subject_sf,
                    predicate_sf,
                    object_sf,
                    span_to_mapping.get(subject_sf, ""),
                    span_to_mapping.get(predicate_sf, ""),
                    span_to_mapping.get(object_sf, "")
                ])

    process_io(input_path, output_path, exchange_file, ".csv")