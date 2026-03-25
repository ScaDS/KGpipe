from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from rdflib import Graph
from rdflib.namespace import OWL, RDF, RDFS


def _local_name(uri: object) -> str:
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", maxsplit=1)[-1]
    if "/" in text:
        return text.rsplit("/", maxsplit=1)[-1]
    return text


def _load_graph(ttl_path: Path) -> Graph:
    graph = Graph()
    graph.parse(ttl_path, format="turtle")
    return graph


def get_available_layers(ttl_path: Path) -> list[str]:
    graph = _load_graph(ttl_path)
    layer_names: set[str] = set()
    for class_node in graph.subjects(RDF.type, OWL.Class):
        for class_type in graph.objects(class_node, RDF.type):
            if class_type != OWL.Class:
                layer_names.add(_local_name(class_type))
    return sorted(layer_names)


def _normalize_layer_filter(layer_filter: Optional[str | Iterable[str]]) -> Optional[set[str]]:
    if layer_filter is None:
        return None
    if isinstance(layer_filter, str):
        return {layer_filter}
    normalized = {layer for layer in layer_filter if layer}
    return normalized or None


def _filtered_class_names(
    graph: Graph, layer_filter: Optional[str | Iterable[str]]
) -> set[str]:
    all_classes = {_local_name(node) for node in graph.subjects(RDF.type, OWL.Class)}
    selected_layers = _normalize_layer_filter(layer_filter)
    if not selected_layers:
        return all_classes

    selected: set[str] = set()
    for class_node in graph.subjects(RDF.type, OWL.Class):
        class_types = {_local_name(node) for node in graph.objects(class_node, RDF.type)}
        if class_types.intersection(selected_layers):
            selected.add(_local_name(class_node))
    return selected


def convert_owl_ttl_to_mermaid(
    ttl_path: Path, layer_filter: Optional[str | Iterable[str]] = None
) -> str:
    graph = _load_graph(ttl_path)
    selected_classes = _filtered_class_names(graph, layer_filter)

    classes = sorted(selected_classes)
    object_property_nodes = sorted(
        set(graph.subjects(RDF.type, OWL.ObjectProperty)), key=lambda node: _local_name(node)
    )
    datatype_property_nodes = sorted(
        set(graph.subjects(RDF.type, OWL.DatatypeProperty)),
        key=lambda node: _local_name(node),
    )

    domain_map: dict[str, list[str]] = defaultdict(list)
    range_map: dict[str, list[str]] = defaultdict(list)
    for prop_node in object_property_nodes + datatype_property_nodes:
        prop_name = _local_name(prop_node)
        for domain in graph.objects(prop_node, RDFS.domain):
            domain_map[prop_name].append(_local_name(domain))
        for value_range in graph.objects(prop_node, RDFS.range):
            range_map[prop_name].append(_local_name(value_range))

    lines: list[str] = ["classDiagram", "direction LR", ""]

    for class_name in classes:
        lines.append(f"class {class_name}")

    subclass_lines: list[str] = []
    for child, _, parent in graph.triples((None, RDFS.subClassOf, None)):
        child_name = _local_name(child)
        parent_name = _local_name(parent)
        if child_name not in selected_classes or parent_name not in selected_classes:
            continue
        subclass_lines.append(f"{parent_name} <|-- {child_name}")
    if subclass_lines:
        lines.extend(["", "%% Inheritance", *sorted(set(subclass_lines))])

    relation_lines: list[str] = []
    for prop in sorted(_local_name(node) for node in object_property_nodes):
        for domain in domain_map.get(prop, []):
            for value_range in range_map.get(prop, []):
                if domain not in selected_classes or value_range not in selected_classes:
                    continue
                relation_lines.append(
                    f'{domain} "0..*" --> "0..*" {value_range} : {prop}'
                )
    if relation_lines:
        lines.extend(["", "%% Object properties", *sorted(set(relation_lines))])

    datatype_map: dict[str, list[str]] = defaultdict(list)
    for prop in sorted(_local_name(node) for node in datatype_property_nodes):
        for domain in domain_map.get(prop, []):
            if domain not in selected_classes:
                continue
            value_ranges = range_map.get(prop, ["string"])
            for value_range in value_ranges:
                datatype_map[domain].append(f"  +{value_range} {prop}")

    if datatype_map:
        lines.extend(["", "%% Datatype properties"])
        for domain in sorted(datatype_map):
            lines.append(f"class {domain} {{")
            lines.extend(sorted(set(datatype_map[domain])))
            lines.append("}")

    return "\n".join(lines) + "\n"


def convert_and_write_mermaid(
    ttl_path: Path,
    output_path: Path,
    layer_filter: Optional[str | Iterable[str]] = None,
) -> str:
    mermaid = convert_owl_ttl_to_mermaid(ttl_path, layer_filter=layer_filter)
    output_path.write_text(mermaid, encoding="utf-8")
    return mermaid
