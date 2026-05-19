import sys
import json
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS


THIS = Namespace("http://kg.org/ontology/")
WD   = Namespace("http://www.wikidata.org/entity/")
WDT  = Namespace("http://www.wikidata.org/prop/direct/")
DBO  = Namespace("http://dbpedia.org/ontology/")
DBP  = Namespace("http://dbpedia.org/property/")
FOAF = Namespace("http://xmlns.com/foaf/0.1/")


DATE_TYPES = {"Date", "Year"}

def is_datatype_prop(range_str: str) -> bool:
    """True when the range maps to an XSD literal rather than an OWL class."""
    return range_str in DATE_TYPES or range_str.startswith("xsd:")

def range_uri(range_str: str):
    """Map a range string to an rdflib URIRef."""
    if range_str == "Date":
        return XSD.date
    if range_str == "Year":
        return XSD.gYear
    if range_str.startswith("xsd:"):
        return XSD[range_str[4:]]
    # Assume it is a concept defined in this ontology
    return THIS[range_str]


def convert(ontology: dict) -> Graph:
    g = Graph()

    g.bind("this", THIS)
    g.bind("foaf", FOAF)
    g.bind("wd",   WD)
    g.bind("wdt",  WDT)
    g.bind("dbo",  DBO)
    g.bind("dbp",  DBP)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)
    g.bind("rdf",  RDF)
    g.bind("skos", SKOS)
    g.bind("xsd",  XSD)

    g.add((THIS[""],  RDF.type,    OWL.Ontology))
    g.add((THIS[""],  RDFS.label,  Literal(ontology.get("title", ""))))
    if "id" in ontology:
        g.add((THIS[""], RDFS.comment, Literal(f"Ontology ID: {ontology['id']}")))

    for concept in ontology.get("concepts", []):
        uri = THIS[concept["qid"]]
        g.add((uri, RDF.type,        OWL.Class))
        g.add((uri, RDF.type,        RDFS.Class))
        g.add((uri, RDFS.subClassOf, OWL.Thing))
        g.add((uri, RDFS.label,      Literal(concept["label"])))
        g.add((uri, RDFS.comment,    Literal(f"Concept representing a {concept['label']}")))
        g.add((uri, OWL.equivalentClass, DBO[concept["qid"]]))


    for rel in ontology.get("relations", []):
        uri = THIS[rel["pid"]]
        rng = rel["range"]

        if is_datatype_prop(rng):
            g.add((uri, RDF.type, OWL.DatatypeProperty))
        else:
            g.add((uri, RDF.type, OWL.ObjectProperty))

        g.add((uri, RDF.type,     RDF.Property))
        g.add((uri, RDFS.label,   Literal(rel["label"])))
        g.add((uri, RDFS.comment, Literal(f"Relation '{rel['label']}' from {rel['domain']} to {rng}")))
        g.add((uri, RDFS.domain,  THIS[rel["domain"]]))
        g.add((uri, RDFS.range,   range_uri(rng)))
        g.add((uri, OWL.equivalentProperty, DBO[rel["pid"]]))


    return g

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    graph = convert(ontology)
    graph.serialize(destination=output_path, format="turtle")
    print(f"Converted '{input_path}' → '{output_path}' ({len(graph)} triples)")


if __name__ == "__main__":
    main()