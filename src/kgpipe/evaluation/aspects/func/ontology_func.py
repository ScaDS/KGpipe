from rdflib import URIRef, Graph
from rdflib.query import ResultRow
from rdflib.namespace import OWL

def get_property_cardinality(ontology_graph: Graph, property: str):
    # if functional property
    property_type_rs = ontology_graph.query(
        """
        SELECT ?type
        WHERE {
            ?property a ?type .
        }
        """,
        initBindings={"property": URIRef(property)}
    )

    property_types = set([str(row["type"]) for row in property_type_rs if isinstance(row, ResultRow)])

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