
import numpy as np
from kgpipe.evaluation.aspects.func.integration_eval import fuzzy_match, evaluate_reference_triple_alignment_fuzzy
from kgpipe.common import KG, DataFormat
from rdflib import Graph
from kgpipe.util.embeddings.st_emb import get_model

def test_fuzzy_literal_match():
    model = get_model()


    test_value = ["Berlin", "Leipzig"]
    reference_values = ["Berlin", "Berlin (city)", "Berlin, Germany", "Groß-Berlin"]

    print(fuzzy_match(test_value, reference_values, model, threshold=0.9))


def test_alignment_soft():


    test_turtle = """
    @prefix : <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .


    :o1 rdfs:label "Germany" .
    :o2 rdfs:label "Germany" .

    :s1 :p1 :o1 .
    :s1 rdfs:label "Berlin" .
    :s2 :p2 :o2 .
    :s2 rdfs:label "Leipzig" .
    """

    reference_turtle = """
    @prefix : <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    
    :o1 rdfs:label "Germany" .
    :o2 rdfs:label "Englad" .
    :o3 rdfs:label "Germany" .

    :s1 :p1 :o1 .
    :s1 rdfs:label "Berlin" .
    :s2 :p2 :o2 .
    :s2 rdfs:label "Leipzig" .
    :s3 :p3 :o1 .
    :s3 rdfs:label "Hamburg" .
    """

    test_graph = Graph()
    test_graph.parse(data=test_turtle, format="turtle")
    reference_graph = Graph()
    reference_graph.parse(data=reference_turtle, format="turtle")

    result = evaluate_reference_triple_alignment_fuzzy(test_graph, reference_graph)
    print(result)