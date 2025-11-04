from dataclasses import dataclass
from typing import List, Self, Dict
from enum import Enum
from rdflib import Graph, RDF, OWL, RDFS, SKOS, URIRef
from pathlib import Path
# A util for ontology related tasks

# Common namespaces
RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NAMESPACE = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NAMESPACE = "http://www.w3.org/2002/07/owl#"
XSD_NAMESPACE = "http://www.w3.org/2001/XMLSchema#"
PROV_NAMESPACE = "http://www.w3.org/ns/prov#"
DCTERMS_NAMESPACE = "http://purl.org/dc/terms/"
SCHEMA_NAMESPACE = "http://schema.org/"
DUBLIN_CORE_NAMESPACE = "http://purl.org/dc/elements/1.1/"


class OntologyExtractor:
    """
    A class to extract the ontology from a given rdfgraph
    all entities of type owl:Class, owl:ObjectProperty, owl:DatatypeProperty, owl:AnnotationProperty, owl:NamedIndividual
    """
    pass

@dataclass
class OwlClass:
    uri: str
    label: str
    alias: List[str]
    description: str
    superclasses: List[Self]
    subclases: List[Self]
    equivalent: set[str]
    disjointWith: set[str]

    def __str__(self) -> str:
        return f"OwlClass(\nuri={self.uri},\nlabel={self.label},\nalias={self.alias},\nsuperclasses={self.superclasses},\nsubclases={self.subclases},\nequivalent={self.equivalent})"


class OwlPropertyType(Enum):
    ObjectProperty = "ObjectProperty"
    DatatypeProperty = "DatatypeProperty"
    AnnotationProperty = "AnnotationProperty"


@dataclass
class OwlProperty:
    uri: str
    type: OwlPropertyType
    label: str
    alias: List[str]
    description: str
    domain: OwlClass
    range: OwlClass
    equivalent: set[str]
    min_cardinality: int = 1
    max_cardinality: int = 10000


    def __str__(self) -> str:
        return f"OwlProperty(\nuri={self.uri},\ntype={self.type},\nlabel={self.label},\nalias={self.alias},\ndomain={self.domain},\nrange={self.range},\nequivalent={self.equivalent})"

@dataclass
class Ontology:
    # uri: str TODO
    classes: List[OwlClass]
    properties: List[OwlProperty]

    def check_equivalent(self, uri_1: str, uri_2: str):

        for _class in self.classes:
            if uri_1 in _class.equivalent or uri_2 in _class.equivalent:
                return True

        for property in self.properties:
            if uri_1 in property.equivalent or uri_2 in property.equivalent:
                return True

        return False

    def get_domain_range(self, uri: str):
        for property in self.properties:
            if uri == property.uri:
                domain = property.domain.uri if property.domain else None
                range = property.range.uri if property.range else None
                return domain, range
        return None, None

    def __str__(self) -> str:
        # INSERT_YOUR_CODE
        import json
        def default(obj):
            # Handle dataclasses and enums
            if hasattr(obj, "__dataclass_fields__"):
                return {k: v for k, v in obj.__dict__.items()}
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, Enum):
                return obj.value
            return str(obj)
        return json.dumps(self, default=default, indent=2)

from rdflib import URIRef, Graph
from rdflib.query import ResultRow
from rdflib.namespace import OWL

def get_property_cardinality(ontology_graph: Graph, property: str):
    # if functional property

    # print(property)
    property_type_rs = ontology_graph.query(
        """
        SELECT ?type
        WHERE {
            ?property a ?type .
        }
        """,
        initBindings={"property": URIRef(property)}
    )

    property_types = set(str(row["type"]) for row in property_type_rs if isinstance(row, ResultRow))
    
    if str(OWL.FunctionalProperty) in property_types:
        return (0, 1)
    
    # if restriction on property
    #_:restriction a owl:Restriction ;
    # owl:onProperty ex:hasSSN ;
    # owl:maxCardinality "1"^^xsd:nonNegativeInteger .

    restriction_rs = ontology_graph.query(
        """
        SELECT ?maxCardinality ?minCardinality
        WHERE {
            ?restriction a owl:Restriction ;
            owl:onProperty ?property ;
            OPTIONAL { ?restriction owl:maxCardinality ?maxCardinality . }
            OPTIONAL { ?restriction owl:minCardinality ?minCardinality . }
        }
        """,
        initBindings={"property": URIRef(property)}
    )

    max_cardinality = 10000
    min_cardinality = 1
    for row in restriction_rs:
        if isinstance(row, ResultRow):
            if row["maxCardinality"] is not None:
                max_cardinality = int(row["maxCardinality"])
            if row["minCardinality"] is not None:
                min_cardinality = int(row["minCardinality"])
        
    return (min_cardinality, max_cardinality)

class OntologyUtil:

    def __init__(self):
        pass

    @staticmethod
    def extract_ontology(ontology_file):
        pass

    @staticmethod
    def load_ontology_from_file(ontology_file: Path):
        graph = Graph()
        graph.parse(ontology_file)
        return OntologyUtil.load_ontology_from_graph(graph)


    @staticmethod
    def load_ontology_from_graph(graph):

        classes : Dict[str, OwlClass] = {}
        properties : Dict[str, OwlProperty] = {}

        def fetch_info(uri: str):
            label = str(graph.value(s, RDFS.label))
            if not label: label = str(s)

            description = str(graph.value(s, RDFS.comment))
            if not description: description = ""

            alias = [ str(o) for s, p, o in graph.triples((uri, SKOS.altLabel, None)) ]
            if not alias: alias = []

            equivalent = set() 
            for _, _, o in graph.triples((uri, OWL.equivalentClass, None)):
                equivalent.add(str(o))
            for _, _, o in graph.triples((uri, OWL.equivalentProperty, None)):
                equivalent.add(str(o))

            return label, alias, equivalent, description

        def fetch_property_info(uri: URIRef):
            label, alias, equivalent, description = fetch_info(uri)

            domain = None
            domain_uri = graph.value(uri, RDFS.domain)
            if domain_uri and str(domain_uri) in classes:
                domain = classes[str(domain_uri)]

            range = None
            range_uri = graph.value(uri, RDFS.range)
            if range_uri:
                range_uri_str = str(range_uri)
                if range_uri_str in classes:
                    range = classes[range_uri_str]
                elif range_uri_str.startswith(XSD_NAMESPACE):
                    range = OwlClass(
                        uri=range_uri_str,
                        label=range_uri_str,
                        alias=[],
                        superclasses=[],
                        subclases=[],
                        equivalent=set(),
                        description="",
                        disjointWith=set()
                    )

            min_cardinality, max_cardinality = get_property_cardinality(graph, uri)

            return label, alias, equivalent, description, domain, range, min_cardinality, max_cardinality

        def fetch_class_info(uri: URIRef):

            label, alias, equivalent, description = fetch_info(uri)

            disjointWith = set()
            for _, _, o in graph.triples((uri, OWL.disjointWith, None)):
                disjointWith.add(str(o))

            for s, _, _ in graph.triples((None, OWL.disjointWith, uri)):
                disjointWith.add(str(s))

            return label, alias, equivalent, description, disjointWith

        for s, p, o in graph.triples((None, RDF.type, OWL.Class)):

            label, alias, equivalent, description, disjointWith = fetch_class_info(s)

            classes[str(s)] = OwlClass(uri=str(s), label=label, alias=alias, superclasses=[], subclases=[], equivalent=equivalent, description=description, disjointWith=disjointWith)

        for s, p, o in graph.triples((None, RDF.type, OWL.ObjectProperty)):
            label, alias, equivalent, description, domain, range, min_cardinality, max_cardinality = fetch_property_info(s)
            properties[str(s)] = OwlProperty(uri=str(s), type=OwlPropertyType.ObjectProperty, label=label, alias=alias, domain=domain, range=range, equivalent=equivalent, description=description, min_cardinality=min_cardinality, max_cardinality=max_cardinality)

        for s, p, o in graph.triples((None, RDF.type, OWL.DatatypeProperty)):
            label, alias, equivalent, description, domain, range, min_cardinality, max_cardinality = fetch_property_info(s)
            properties[str(s)] = OwlProperty(uri=str(s), type=OwlPropertyType.DatatypeProperty, label=label, alias=alias, domain=domain, range=range, equivalent=equivalent, description=description, min_cardinality=min_cardinality, max_cardinality=max_cardinality)


        return Ontology(classes=list(classes.values()), properties=list(properties.values()))

if __name__ == "__main__":
    ontology_util = OntologyUtil()
    ontology = ontology_util.load_ontology_from_file("/home/marvin/project/data/current/ontology.ttl")
    print(ontology)