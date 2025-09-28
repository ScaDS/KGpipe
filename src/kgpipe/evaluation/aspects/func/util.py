from rdflib import Graph, RDF, OWL
from collections import defaultdict

# from owlrl import DeductiveClosure, OWLRL_Semantics

# def apply_reasoning(g: Graph) -> Graph:
#     inferred_graph = g.copy()
#     DeductiveClosure(OWLRL_Semantics).expand(inferred_graph)
#     return inferred_graph

def extract_cardinality_restrictions(g: Graph):
    restrictions = defaultdict(dict)
    for r in g.subjects(RDF.type, OWL.Restriction):
        prop = g.value(r, OWL.onProperty)

        max_card = g.value(r, OWL.maxCardinality)
        min_card = g.value(r, OWL.minCardinality)
        exact_card = g.value(r, OWL.cardinality)

        if max_card:
            restrictions[prop]["max"] = int(max_card)
        if min_card:
            restrictions[prop]["min"] = int(min_card)
        if exact_card:
            restrictions[prop]["exact"] = int(exact_card)

    return restrictions
