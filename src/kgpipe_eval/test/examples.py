TEST_TURTLE_TRIPLES = """
@prefix : <http://example.org/bookstore/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:itemA rdf:type :Book ;
    rdfs:label "itemA" ;
    :bookTitle "The Hobbit, or There and Back Again" ;
    :bookAuthor :authorTolkien ;
    :isbn13 "9780261102217" .
"""

REFERENCE_TURTLE_TRIPLES = """
@prefix : <http://example.org/bookstore/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:itemA rdf:type :Book ;
    rdfs:label "itemA" ;
    :bookTitle "The Hobbit, or There and Back Again" ;
    :bookAuthor :authorTolkien ;
    :isbn13 "9780261102217" .
"""