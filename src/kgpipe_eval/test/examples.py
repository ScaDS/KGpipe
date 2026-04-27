SEED_TURTLE_TRIPLES = """
@prefix : <http://example.org/seed_bookstore/> .
@prefix o: <http://example.org/ontology/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:store1 rdf:type o:BookStore ;
    rdfs:label "Example Books (Downtown)"@en ;
    :countryCode "US" ;
    :hasInventory :itemA, :itemB, :itemC .

:publisherHC rdf:type o:Publisher ;
    rdfs:label "HarperCollins" ;
    :countryCode "GB" .
"""
TEST_TURTLE_TRIPLES = """
@prefix : <http://example.org/test_bookstore/> .
@prefix o: <http://example.org/ontology/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Entities designed to exercise alignment corner-cases:
# - multiple entities per type (Book/Author/Publisher/Store)
# - missing / extra attributes across graphs
# - literal variations (lang tags, datatypes, different lexical forms)
# - ambiguous labels (near-duplicates, casing differences)
# - multi-valued properties

:store1 rdf:type o:BookStore ;
    rdfs:label "Example Books (Downtown)"@en ;
    :countryCode "US" ;
    :hasInventory :itemA, :itemB, :itemC .

:publisherHC rdf:type o:Publisher ;
    rdfs:label "HarperCollins" ;
    :countryCode "GB" .

# different wrong type
:publisherPenguin rdf:type o:Author ;
    rdfs:label "Penguin Books"@en ;
    :countryCode "GB" .

:authorTolkien rdf:type o:Author ;
    rdfs:label "J. R. R. Tolkien" ;
    :born "1892-01-03"^^xsd:date ;
    :died "1973-09-02"^^xsd:date ;
    :sameAs <https://www.wikidata.org/entity/Q892> .

:authorRowling rdf:type o:Author ;
    rdfs:label "J.K. Rowling" ;
    :born "1965-07-31"^^xsd:date .

:itemA rdf:type o:Book ;
    rdfs:label "The Hobbit"@en ;
    :bookTitle "The Hobbit, or There and Back Again"@en ;
    :bookAuthor :authorTolkien ;
    :publisher :publisherHC ;
    :isbn13 "9780261102217" ;
    :pageCount "310"^^xsd:integer ;
    :tags "fantasy", "classic" ;
    :inSeries :seriesMiddleEarth .

:itemB rdf:type o:Book ;
    rdfs:label "The Hobbit (Illustrated)"@en ;
    :bookTitle "The Hobbit"@en ;
    :bookAuthor :authorTolkien ;
    :publisher :publisherHC ;
    :isbn13 "978-0-261-10221-7" ; # lexical variation
    :pageCount 320 ; # integer without explicit datatype
    :publicationYear "1997"^^xsd:gYear .

:itemC rdf:type o:Book ;
    rdfs:label "Harry Potter and the Philosopher's Stone"@en ;
    :bookTitle "Harry Potter and the Philosopher's Stone"@en ;
    :bookAuthor :authorRowling ;
    :publisher :publisherPenguin ;
    :isbn13 "9780747532699" ;
    :pageCount "223"^^xsd:integer .

# Same label, different type (common edge case for label-only alignment)
:hobbit rdf:type o:Film ;
    rdfs:label "The Hobbit"@en ;
    :releaseYear "2012"^^xsd:gYear .

# Missing rdf:type but has label (edge case for type-aware matching)
:unknownEntity rdfs:label "HarperCollins" .

:seriesMiddleEarth rdf:type o:Series ;
    rdfs:label "Middle-earth Legendarium"@en .

# false positive unexpected entity
:unexpectedEntity rdf:type o:Book ;
    rdfs:label "Unexpected Entity"@en .
"""

GENERATED_TURTLE_TRIPLES = """
@prefix : <http://example.org/ontology/> .
@prefix o: <http://example.org/ontology/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Entities designed to exercise alignment corner-cases:
# - multiple entities per type (Book/Author/Publisher/Store)
# - missing / extra attributes across graphs
# - literal variations (lang tags, datatypes, different lexical forms)
# - ambiguous labels (near-duplicates, casing differences)
# - multi-valued properties

:store1 rdf:type o:BookStore ;
    rdfs:label "Example Books (Downtown)"@en ;
    :countryCode "US" ;
    :hasInventory :itemA, :itemB, :itemC .

:publisherHC rdf:type o:Publisher ;
    rdfs:label "HarperCollins" ;
    :countryCode "GB" .

# different wrong type
:publisherPenguin rdf:type o:Author ;
    rdfs:label "Penguin Books"@en ;
    :countryCode "GB" .

:authorTolkien rdf:type o:Author ;
    rdfs:label "J. R. R. Tolkien" ;
    :born "1892-01-03"^^xsd:date ;
    :died "1973-09-02"^^xsd:date ;
    :sameAs <https://www.wikidata.org/entity/Q892> .

:authorRowling rdf:type o:Author ;
    rdfs:label "J.K. Rowling" ;
    :born "1965-07-31"^^xsd:date .

:itemA rdf:type o:Book ;
    rdfs:label "The Hobbit"@en ;
    :bookTitle "The Hobbit, or There and Back Again"@en ;
    :bookAuthor :authorTolkien ;
    :publisher :publisherHC ;
    :isbn13 "9780261102217" ;
    :pageCount "310"^^xsd:integer ;
    :tags "fantasy", "classic" ;
    :inSeries :seriesMiddleEarth .

:itemB rdf:type o:Book ;
    rdfs:label "The Hobbit (Illustrated)"@en ;
    :bookTitle "The Hobbit"@en ;
    :bookAuthor :authorTolkien ;
    :publisher :publisherHC ;
    :isbn13 "978-0-261-10221-7" ; # lexical variation
    :pageCount 320 ; # integer without explicit datatype
    :publicationYear "1997"^^xsd:gYear .

:itemC rdf:type o:Book ;
    rdfs:label "Harry Potter and the Philosopher's Stone"@en ;
    :bookTitle "Harry Potter and the Philosopher's Stone"@en ;
    :bookAuthor :authorRowling ;
    :publisher :publisherPenguin ;
    :isbn13 "9780747532699" ;
    :pageCount "223"^^xsd:integer .

# Same label, different type (common edge case for label-only alignment)
:hobbit rdf:type o:Film ;
    rdfs:label "The Hobbit"@en ;
    :releaseYear "2012"^^xsd:gYear .

# Missing rdf:type but has label (edge case for type-aware matching)
:unknownEntity rdfs:label "HarperCollins" .

:seriesMiddleEarth rdf:type o:Series ;
    rdfs:label "Middle-earth Legendarium"@en .

# false positive unexpected entity
:unexpectedEntity rdf:type o:Book ;
    rdfs:label "Unexpected Entity"@en .
"""

REFERENCE_TURTLE_TRIPLES = """
@prefix : <http://example.org/ontology/> .
@prefix o: <http://example.org/ontology/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Reference graph intentionally differs from TEST_TURTLE_TRIPLES:
# - different labels / casing / punctuation
# - extra / missing properties
# - alternate modeling (blank nodes, different predicates)
# - near-duplicate entities to test ambiguity

:storeMain rdf:type o:BookStore ;
    rdfs:label "Example Books - Downtown"@en ;
    :countryCode "USA" ; # lexical variation
    :hasInventory :refItemA, :refItemC .

:publisherHC rdf:type o:Publisher ;
    rdfs:label "Harper Collins"@en ; # spacing difference
    :countryCode "UK" .

:publisherPenguin rdf:type o:Publisher ;
    rdfs:label "Penguin"@en ;
    :countryCode "GB" .

:authorTolkien rdf:type o:Author ;
    rdfs:label "J.R.R. Tolkien" ; # punctuation difference
    :born "1892-01-03"^^xsd:date ;
    :sameAs <https://www.wikidata.org/entity/Q892> ;
    :nameParts [ :given "John" ; :middle "Ronald Reuel" ; :family "Tolkien" ] .

:authorRowling rdf:type o:Author ;
    rdfs:label "Joanne Rowling"@en ; # alias-ish label
    :born "1965-07-31"^^xsd:date .

:refItemA rdf:type o:Book ;
    rdfs:label "The Hobbit"@en ;
    :title "The Hobbit, or There and Back Again"@en ; # different predicate
    :bookAuthor :authorTolkien ;
    :publisher :publisherHC ;
    :isbn13 "9780261102217" ;
    :pageCount "310"^^xsd:integer ;
    :tags "classic" . # missing one tag compared to test

# Same-work but modeled as a separate edition entity
:refItemA_Edition1 rdf:type o:Edition ;
    rdfs:label "The Hobbit (1st edition)"@en ;
    :about :refItemA ;
    :publicationYear "1937"^^xsd:gYear .

:refItemC rdf:type o:Book ;
    rdfs:label "Harry Potter and the Philosopher’s Stone"@en ; # curly apostrophe
    :bookTitle "Harry Potter and the Philosopher's Stone"@en ;
    :bookAuthor :authorRowling ;
    :publisher :publisherPenguin ;
    :isbn13 "9780747532699" ;
    :pageCount "223"^^xsd:integer ;
    :tags "fantasy"@en .

# Near-duplicate label (to trigger ambiguity in label similarity)
:refItemC_US rdf:type o:Book ;
    rdfs:label "Harry Potter and the Sorcerer's Stone"@en ;
    :sameAs :refItemC .
"""

VERIFIED_ENTITIES = """
dataset,entity_id,entity_label,entity_type
test,http://example.org/reference_bookstore/itemA,The Hobbit,o:Book
test,http://example.org/reference_bookstore/itemB,The Hobbit (Illustrated),o:Book
test,http://example.org/reference_bookstore/itemC,Harry Potter and the Philosopher's Stone,o:Book
test,http://example.org/reference_bookstore/authorTolkien,J. R. R. Tolkien,o:Author
test,http://example.org/reference_bookstore/authorRowling,J.K. Rowling,o:Author
test,http://example.org/reference_bookstore/publisherHC,HarperCollins,o:Publisher
test,http://example.org/reference_bookstore/publisherPenguin,Penguin Books,o:Publisher
test,http://example.org/reference_bookstore/store1,Example Books (Downtown),o:BookStore
test,http://example.org/reference_bookstore/seriesMiddleEarth,Middle-earth Legendarium,o:Series
test,http://example.org/reference_bookstore/missingEntity,Gone with the Wind,o:Book
"""