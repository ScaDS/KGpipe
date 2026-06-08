import sys
import json
import re
import argparse

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, XSD

DBP  = Namespace("http://dbpedia.org/property/")
DBO  = Namespace("http://kg.org/ontology/")     # currently used namespace in ontology file
DBR  = Namespace("http://dbpedia.org/resource/")
EX   = Namespace("http://example.org/data/")    # fallback for unknown resources

LITERAL_TYPES = {
    XSD.integer, XSD.int, XSD.long, XSD.short, XSD.byte,
    XSD.decimal, XSD.float, XSD.double,
    XSD.boolean,
    XSD.date, XSD.dateTime, XSD.gYear, XSD.gYearMonth,
    XSD.string, RDFS.Literal,
    URIRef("http://www.w3.org/2000/01/rdf-schema#Literal"),
}


def load_ontology(path: str) -> Graph:
    g = Graph()
    formats = ["xml", "turtle", "n3", "nt", "json-ld"]
    last_error = None
    for fmt in formats:
        try:
            g.parse(path, format=fmt)
            print(f"Ontology loaded ({len(g)} triples, format={fmt})", file=sys.stderr)
            return g
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Could not parse ontology '{path}': {last_error}")


def get_range(ontology: Graph, rel: str) -> URIRef | None:
    prop_uri = DBO[rel]
    for obj in ontology.objects(prop_uri, RDFS.range):
        return URIRef(obj)
    return None


def is_literal_range(range_uri: URIRef | None) -> bool:
    if range_uri is None:
        return False
    return range_uri in LITERAL_TYPES or str(range_uri).startswith(str(XSD))


_QUOTED_RE = re.compile(r'^"(.*)"$')


def strip_outer_quotes(value: str) -> str:
    m = _QUOTED_RE.match(value)
    return m.group(1) if m else value


def looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def looks_year(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", value))


def make_label(raw: str) -> str:
    label = strip_outer_quotes(raw).replace("_", " ").strip()
    return label


def resource_uri(name: str) -> URIRef:
    clean = strip_outer_quotes(name).replace(" ", "_")
    return DBR[clean]


def infer_xsd_type(value: str) -> URIRef:
    if looks_year(value):
        return XSD.gYear
    if re.fullmatch(r"\d+", value):
        return XSD.integer
    try:
        float(value)
        return XSD.decimal
    except ValueError:
        pass
    return XSD.string


def convert(data_path: str, ontology_path: str, output_path: str) -> None:
    ontology = load_ontology(ontology_path)

    out_graph = Graph()
    out_graph.bind("dbo",  DBO)
    out_graph.bind("dbr",  DBR)
    out_graph.bind("dbp",  DBP)
    out_graph.bind("rdf",  RDF)
    out_graph.bind("rdfs", RDFS)
    out_graph.bind("xsd",  XSD)

    labelled: set[URIRef] = set()

    def add_label(uri: URIRef, raw_name: str) -> None:
        if uri not in labelled:
            out_graph.add((uri, RDFS.label, Literal(make_label(raw_name), lang="en")))
            labelled.add(uri)

    with open(data_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] Line {lineno}: JSON error – {e}", file=sys.stderr)
                continue

            sent_id = record.get("id", f"unknown_{lineno}")
            sentence = record.get("sent", "")
            triples  = record.get("triples", [])

            for triple in triples:
                sub_raw = triple["sub"]
                rel_raw = triple["rel"]
                obj_raw = triple["obj"]

                sub_uri  = resource_uri(sub_raw)
                prop_uri = DBO[rel_raw]

                range_uri   = get_range(ontology, rel_raw)
                literal_val = is_literal_range(range_uri)

                if range_uri is None:
                    obj_clean = strip_outer_quotes(obj_raw)
                    if looks_numeric(obj_clean) or _QUOTED_RE.match(obj_raw):
                        literal_val = True

                if literal_val:
                    obj_clean = strip_outer_quotes(obj_raw)
                    xsd_type  = (
                        infer_xsd_type(obj_clean)
                        if range_uri is None or range_uri == RDFS.Literal
                        else range_uri
                    )
                    obj_node = Literal(obj_clean, datatype=xsd_type)
                else:
                    obj_node = resource_uri(obj_raw)
                    if range_uri and range_uri not in LITERAL_TYPES:
                        out_graph.add((obj_node, RDF.type, range_uri))
                        add_label(range_uri, str(range_uri).rsplit("/", 1)[-1])
                    add_label(obj_node, obj_raw)

                out_graph.add((sub_uri, prop_uri, obj_node))
                add_label(sub_uri, sub_raw)

    out_graph.serialize(destination=output_path, format="nt")
    print(f"Written {output_path}", file=sys.stderr)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JSON ground truth triple files to N-Triples RDF."
    )
    parser.add_argument("data_file",     help="Path to the JSON-lines ground truth triple file")
    parser.add_argument("ontology_file", help="Path to the RDF/OWL ontology file")
    parser.add_argument("output_file",   help="Output .nt file")
    args = parser.parse_args()
    convert(args.data_file, args.ontology_file, args.output_file)


if __name__ == "__main__":
    main()