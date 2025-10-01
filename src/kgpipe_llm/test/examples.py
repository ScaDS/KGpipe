"""
Examples demonstrating the usage of LLM-based data integration tasks.
"""

import json
from typing import Optional
from kgpipe_llm.common import LLMClient
from kgpipe_llm.common.llm_tasks import (
    extract_triplets_from_text,
    generate_transformation_mapping,
    check_extraction_result,
    match_ontology_to_schema,
    validate_entity_matches
)
from kgpipe_llm.common.rdf_tasks import (
    extract_rdf_triples_from_text,
    map_json_to_rdf_schema,
    generate_rdf_queries
)
from kgpipe_llm.common.config import config_manager, COMMON_NAMESPACES, ONTOLOGY_CONSTRAINTS

def example_text_extraction():
    """Example: Extract triplets from text."""
    print("=== Text Extraction Example ===")
    
    text = """
    Inception is a 2010 science fiction film directed by Christopher Nolan. 
    The film stars Leonardo DiCaprio as Cobb, a thief who steals corporate secrets 
    through dream-sharing technology. The film was produced by Emma Thomas and 
    features music by Hans Zimmer.
    """
    
    result = extract_triplets_from_text(text)
    if result:
        print("Extracted triplets:")
        for triple in result.triplets:
            print(f"  {triple.subject} -- {triple.predicate} --> {triple.object}")
    else:
        print("Failed to extract triplets")

def example_schema_mapping():
    """Example: Map JSON schema to ontology."""
    print("\n=== Schema Mapping Example ===")
    
    json_data = """
    {
        "genres": ["Action", "Adventure", "Sci-Fi"],
        "id": "tt1375666",
        "involvedPeople": [
            {
                "birthYear": 1974,
                "category": "actor",
                "id": "nm0000138",
                "ordering": 1,
                "primaryName": "Leonardo DiCaprio"
            },
            {
                "birthYear": 1970,
                "category": "director",
                "id": "nm0634240",
                "ordering": 5,
                "primaryName": "Christopher Nolan"
            }
        ],
        "originalTitle": "Inception",
        "primaryTitle": "Inception",
        "runtimeMinutes": 148,
        "startYear": 2010,
        "titleType": "movie"
    }
    """
    
    ontology = """
    Movie {
        id: string
        title: string
        genres: list[string]
        actors: list[Person]
        directors: list[Person]
        runtime: int
        releaseYear: int
    }
    
    Person {
        id: string
        name: string
        birthYear: int
    }
    """
    
    result = generate_transformation_mapping(json_data, ontology)
    if result:
        print("Generated mappings:")
        for mapping in result.mappings:
            print(f"  {mapping.key} -> {mapping.link} ({mapping.rule})")
    else:
        print("Failed to generate mappings")

def example_rdf_extraction():
    """Example: Extract RDF triples with proper URIs."""
    print("\n=== RDF Extraction Example ===")
    
    text = """
    Inception is a 2010 science fiction film directed by Christopher Nolan. 
    The film stars Leonardo DiCaprio and was produced by Emma Thomas.
    """
    
    # Use Wikidata namespace prefixes
    namespace_prefixes = {
        "wd": "http://www.wikidata.org/entity/",
        "wdt": "http://www.wikidata.org/prop/direct/",
        "schema": "http://schema.org/"
    }
    
    result = extract_rdf_triples_from_text(text, namespace_prefixes)
    if result:
        print("Extracted RDF triples:")
        for triple in result.triples:
            print(f"  {triple.subject} {triple.predicate} {triple.object}")
        print(f"Namespaces used: {result.namespace_prefixes}")
    else:
        print("Failed to extract RDF triples")

def example_entity_matching():
    """Example: Validate entity matches."""
    print("\n=== Entity Matching Example ===")
    
    source_entities = ["Leonardo DiCaprio", "Christopher Nolan", "Inception"]
    target_entities = ["Q38111", "Q44564", "Q151904"]  # Wikidata IDs
    
    result = validate_entity_matches(source_entities, target_entities)
    if result:
        print("Entity matches:")
        for match in result.matches:
            print(f"  {match.source_entity} -> {match.target_entity} "
                  f"(similarity: {match.similarity_score:.2f}, "
                  f"confidence: {match.confidence:.2f})")
    else:
        print("Failed to validate entity matches")

def example_ontology_matching():
    """Example: Match source schema to target ontology."""
    print("\n=== Ontology Matching Example ===")
    
    source_schema = """
    {
        "movie_id": "string",
        "title": "string", 
        "director": "string",
        "actors": ["string"],
        "release_date": "date",
        "genre": "string"
    }
    """
    
    target_ontology = """
    Movie {
        id: string
        title: string
        director: Person
        actors: list[Person]
        releaseDate: date
        genres: list[Genre]
    }
    
    Person {
        id: string
        name: string
    }
    
    Genre {
        id: string
        name: string
    }
    """
    
    result = match_ontology_to_schema(source_schema, target_ontology)
    if result:
        print("Ontology mappings:")
        for mapping in result.mappings:
            print(f"  {mapping.source_field} -> {mapping.target_class}.{mapping.target_property} "
                  f"({mapping.mapping_type}, confidence: {mapping.confidence:.2f})")
    else:
        print("Failed to match ontology")

def example_json_to_rdf_mapping():
    """Example: Map JSON data to RDF schema."""
    print("\n=== JSON to RDF Mapping Example ===")
    
    json_data = """
    {
        "id": "tt1375666",
        "title": "Inception",
        "director": {
            "id": "nm0634240",
            "name": "Christopher Nolan"
        },
        "actors": [
            {
                "id": "nm0000138", 
                "name": "Leonardo DiCaprio"
            }
        ],
        "releaseYear": 2010
    }
    """
    
    rdf_ontology = """
    @prefix schema: <http://schema.org/>
    @prefix wd: <http://www.wikidata.org/entity/>
    
    schema:Movie {
        schema:identifier: string
        schema:name: string
        schema:director: schema:Person
        schema:actor: list[schema:Person]
        schema:datePublished: date
    }
    
    schema:Person {
        schema:identifier: string
        schema:name: string
    }
    """
    
    result = map_json_to_rdf_schema(json_data, rdf_ontology)
    if result:
        print("JSON to RDF mappings:")
        for mapping in result.mappings:
            print(f"  {mapping.source_field} -> {mapping.target_class}.{mapping.target_property} "
                  f"({mapping.mapping_type}, confidence: {mapping.confidence:.2f})")
    else:
        print("Failed to map JSON to RDF")

def example_query_generation():
    """Example: Generate SPARQL queries from RDF triples."""
    print("\n=== Query Generation Example ===")
    
    from kgpipe_llm.common.models import RDFTriple
    
    triples = [
        RDFTriple(
            subject="http://www.wikidata.org/entity/Q151904",
            predicate="http://schema.org/name", 
            object="Inception"
        ),
        RDFTriple(
            subject="http://www.wikidata.org/entity/Q151904",
            predicate="http://schema.org/director",
            object="http://www.wikidata.org/entity/Q44564"
        )
    ]
    
    query = generate_rdf_queries(triples, "sparql")
    if query:
        print("Generated SPARQL query:")
        print(query)
    else:
        print("Failed to generate query")

def example_validation():
    """Example: Validate extraction results."""
    print("\n=== Validation Example ===")
    
    text = """
    Inception is a 2010 science fiction film directed by Christopher Nolan.
    """
    
    triple = {
        "subject": "Inception",
        "predicate": "director", 
        "object": "Christopher Nolan"
    }
    
    result = check_extraction_result(text, triple)
    if result:
        print(f"Validation result: {result.contained}")
        print(f"Text: {result.text}")
        print(f"Triple: {result.triple.subject} -- {result.triple.predicate} --> {result.triple.object}")
        assert result.triple.subject == "Inception"
        assert result.triple.predicate == "director"
        assert result.triple.object == "Christopher Nolan"
    else:
        print("Failed to validate extraction")

def run_all_examples():
    """Run all examples to demonstrate the module functionality."""
    print("Running LLM-based Data Integration Examples")
    print("=" * 50)
    
    try:
        # example_text_extraction()
        # example_schema_mapping()
        # example_rdf_extraction()
        # example_entity_matching()
        # example_ontology_matching()
        # example_json_to_rdf_mapping()
        example_query_generation()
        example_validation()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")

if __name__ == "__main__":
    run_all_examples() 