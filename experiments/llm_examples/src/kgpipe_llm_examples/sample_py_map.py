#!/usr/bin/env python3
"""
converter.py

Command line tool to convert a JSON representation of a film into RDF
using the supplied KG ontology.  The script is invoked as:

    python converter.py INPUT.json OUTPUT.rdf

The output format is chosen by the file extension of OUTPUT.
Only rdflib and the Python standard library are used.
"""
import argparse
import json
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

from rdflib import Graph, Literal, Namespace, RDF, RDFS, XSD, URIRef

# ---------------------------------------------------------------------------
# Namespace declarations (deterministic URIs are built under EX)
# ---------------------------------------------------------------------------
KG = Namespace("http://kg.org/ontology/")
EX = Namespace("http://example.org/resource/")

# ---------------------------------------------------------------------------
# Mapping configuration
# ---------------------------------------------------------------------------
# Each entry maps a JSON key (or dot‑separated path for nested objects) to a
# tuple: (ontology URI, mapping type)
# mapping type: "type" (rdf:type), "literal", "object", "repeat_literal",
# "repeat_object"
MAPPING: Mapping[str, tuple[URIRef, str]] = {
    # Film level
    "label": (RDFS.label, "literal"),
    "runtime": (KG.runtime, "literal"),
    "budget": (KG.budget, "literal"),
    "gross": (KG.gross, "literal"),
    # Relationships
    "writer": (KG.writer, "object"),
    "director": (KG.director, "object"),
    "producer": (KG.producer, "object"),
    "starring": (KG.starring, "repeat_object"),
    "productionCompany": (KG.production, "repeat_object"),
    "distributor": (KG.distribution, "repeat_object"),
    "musicComposer": (KG.composer, "repeat_object"),
    # Person attributes
    "birthDate": (KG.birthDate, "literal"),
    "birthPlace": (KG.birthPlace, "literal"),
    "occupation": (KG.occupation, "literal"),
    # Company attributes
    "foundingDate": (KG.foundingDate, "literal"),
    "industry": (KG.industry, "literal"),
    "headquarter": (KG.headquarter, "literal"),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load JSON from *path*.
    Accepts either a single object or a list of objects.
    Returns a list of film dictionaries.
    """
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logging.error("Failed to read JSON file %s: %s", path, exc)
        sys.exit(1)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    logging.error("JSON root must be an object or a list of objects.")
    sys.exit(1)


def deterministic_uri(base: Namespace, seed: str) -> URIRef:
    """Generate a deterministic URI under *base* by hashing *seed*."""
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return URIRef(f"{base}{h}")


def is_iso_date(s: str) -> bool:
    """Return True if *s* looks like an ISO‑8601 date (YYYY-MM-DD)."""
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s))


def literal_from_value(val: Any) -> Literal:
    """Convert a Python value to an rdflib Literal with appropriate XSD type."""
    if isinstance(val, bool):
        return Literal(val, datatype=XSD.boolean)
    if isinstance(val, int):
        return Literal(val, datatype=XSD.integer)
    if isinstance(val, float):
        return Literal(val, datatype=XSD.decimal)
    if isinstance(val, str):
        if is_iso_date(val):
            return Literal(val, datatype=XSD.date)
        # Accept scientific notation as decimal
        if re.fullmatch(r"[-+]?\d+(\.\d+)?[eE][-+]?\d+", val):
            try:
                f = float(val)
                return Literal(f, datatype=XSD.decimal)
            except ValueError:
                pass
        return Literal(val)  # plain string
    # Fallback
    return Literal(str(val))


def ensure_uri_for_entity(name: str, class_uri: URIRef) -> URIRef:
    """Create or retrieve a deterministic URI for an entity identified by *name*."""
    # Using the name string as seed gives stable URIs across runs.
    return deterministic_uri(EX, f"{class_uri}_{name}")


def add_person(graph: Graph, data: Dict[str, Any]) -> URIRef:
    """Add a Person node from *data* and return its URI."""
    label = data.get("label")
    if not label:
        # Fallback to hash of the whole dict if no label is present.
        seed = json.dumps(data, sort_keys=True, separators=(",", ":"))
        person_uri = deterministic_uri(EX, seed)
    else:
        person_uri = ensure_uri_for_entity(label, KG.Person)

    graph.add((person_uri, RDF.type, KG.Person))
    if label:
        graph.add((person_uri, RDFS.label, Literal(label)))

    # Add known person attributes
    for key in ("birthDate", "birthPlace", "occupation"):
        if key in data:
            pred, _ = MAPPING[key]
            graph.add((person_uri, pred, literal_from_value(data[key])))
    return person_uri


def add_company(graph: Graph, data: Dict[str, Any]) -> URIRef:
    """Add a Company node from *data* and return its URI."""
    label = data.get("label")
    if not label:
        seed = json.dumps(data, sort_keys=True, separators=(",", ":"))
        comp_uri = deterministic_uri(EX, seed)
    else:
        comp_uri = ensure_uri_for_entity(label, KG.Company)

    graph.add((comp_uri, RDF.type, KG.Company))
    if label:
        graph.add((comp_uri, RDFS.label, Literal(label)))

    for key in ("foundingDate", "industry", "headquarter"):
        if key in data:
            pred, _ = MAPPING[key]
            graph.add((comp_uri, pred, literal_from_value(data[key])))
    return comp_uri


def add_film(graph: Graph, film_data: Dict[str, Any]) -> None:
    """Convert a single film JSON object into RDF triples."""
    # Create deterministic film URI
    film_uri = deterministic_uri(EX, json.dumps(film_data, sort_keys=True, separators=(",", ":")))
    graph.add((film_uri, RDF.type, KG.Film))

    # Process scalar fields
    for key, (pred, mtype) in MAPPING.items():
        if key not in film_data:
            continue

        if mtype == "literal":
            graph.add((film_uri, pred, literal_from_value(film_data[key])))
        elif mtype == "object":
            # Expect a dict or a string (name)
            val = film_data[key]
            if isinstance(val, dict):
                person_uri = add_person(graph, val)
            else:
                person_uri = ensure_uri_for_entity(str(val), KG.Person)
                graph.add((person_uri, RDF.type, KG.Person))
                graph.add((person_uri, RDFS.label, Literal(str(val))))
            graph.add((film_uri, pred, person_uri))
        elif mtype == "repeat_object":
            arr = film_data[key]
            if not isinstance(arr, list):
                logging.warning("Expected list for key %s, got %s", key, type(arr))
                continue
            for item in arr:
                if key in ("starring", "musicComposer"):
                    if isinstance(item, dict):
                        person_uri = add_person(graph, item)
                    else:
                        person_uri = ensure_uri_for_entity(str(item), KG.Person)
                        graph.add((person_uri, RDF.type, KG.Person))
                        graph.add((person_uri, RDFS.label, Literal(str(item))))
                    graph.add((film_uri, pred, person_uri))
                elif key in ("productionCompany", "distributor"):
                    if isinstance(item, dict):
                        comp_uri = add_company(graph, item)
                    else:
                        comp_uri = ensure_uri_for_entity(str(item), KG.Company)
                        graph.add((comp_uri, RDF.type, KG.Company))
                        graph.add((comp_uri, RDFS.label, Literal(str(item))))
                    graph.add((film_uri, pred, comp_uri))
                else:
                    logging.warning("Unhandled repeat_object key: %s", key)

    # Additional scalar fields not covered by MAPPING but present in example.
    for extra_key in ("writer",):
        if extra_key in film_data and extra_key not in MAPPING:
            logging.warning("No mapping for %s; skipping.", extra_key)


def load_ontology() -> Graph:
    """Return a minimal ontology graph that defines needed prefixes and classes."""
    g = Graph()
    # Bind common prefixes
    g.bind("kg", KG)
    g.bind("ex", EX)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("owl", Namespace("http://www.w3.org/2002/07/owl#"))

    # Minimal class and property declarations (optional, kept lightweight)
    for cls in (KG.Film, KG.Person, KG.Company):
        g.add((cls, RDF.type, Namespace("http://www.w3.org/2002/07/owl#")["Class"]))

    for pred in (
        KG.writer,
        KG.starring,
        KG.production,
        KG.director,
        KG.distribution,
        KG.producer,
        KG.composer,
        KG.runtime,
        KG.budget,
        KG.gross,
        KG.birthDate,
        KG.birthPlace,
        KG.occupation,
        KG.foundingDate,
        KG.industry,
        KG.headquarter,
    ):
        g.add((pred, RDF.type, Namespace("http://www.w3.org/2002/07/owl#")["ObjectProperty"]))
    return g


def serialize_graph(graph: Graph, output_path: Path) -> None:
    """Serialize *graph* to *output_path* according to the file extension."""
    ext = output_path.suffix.lower()
    if ext == ".ttl":
        fmt = "turtle"
    elif ext == ".nt":
        fmt = "nt"
    elif ext in {".rdf", ".xml"}:
        fmt = "xml"
    else:
        fmt = "turtle"  # default
    try:
        graph.serialize(destination=str(output_path), format=fmt)
    except Exception as exc:
        logging.error("Failed to serialize graph: %s", exc)
        sys.exit(1)


def validate_required_keys(film: Dict[str, Any]) -> None:
    """Ensure that required top‑level keys exist in *film*."""
    required = ["label"]
    missing = [k for k in required if k not in film]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} in film object.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert film JSON to RDF.")
    parser.add_argument("input", type=Path, help="Input JSON file")
    parser.add_argument("output", type=Path, help="Output RDF file")
    args = parser.parse_args()

    # Load ontology (mostly for prefix binding)
    ontology_graph = load_ontology()

    # Load JSON data
    film_objs = load_json(args.input)

    # Build RDF graph
    g = Graph()
    for prefix, ns in ontology_graph.namespaces():
        g.bind(prefix, ns)

    for film in film_objs:
        try:
            validate_required_keys(film)
            add_film(g, film)
        except Exception as exc:
            logging.error("Error processing film object: %s", exc)
            sys.exit(1)

    # Serialize
    serialize_graph(g, args.output)
    sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)
    main()