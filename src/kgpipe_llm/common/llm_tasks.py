"""
LLM-based data integration tasks for Knowledge Graph construction.

This module provides structured LLM tasks for integrating text, JSON, and RDF data
into existing RDF Knowledge Graphs using Pydantic for structured output validation.

This file serves as a compatibility layer for the refactored modular structure.
For new code, prefer importing directly from the specific modules:
- core: LLMClient, BaseTask
- models: All Pydantic models
- tasks: Main task functions
- rdf_tasks: RDF-specific tasks
- config: Configuration management
"""

# Import all the main components for backward compatibility
from .core import LLMClient, BaseTask, get_client_from_env, get_config_from_env
from .models import (
    Triplet, Triplets,
    TransformationMapping, TransformationMappings,
    SimpleTriplet, TextExtractionValidation,
    OntologyMapping, OntologyMappings,
    EntityMatch, EntityMatches,
    SchemaMapping, SchemaMappings,
    RDFTriple, RDFGraph
)
from .tasks import (
    extract_triplets_from_text,
    generate_transformation_mapping,
    check_extraction_result,
    match_ontology_to_schema,
    validate_entity_matches,
    generate_schema_mappings,
    TextExtractionTask,
    SchemaMappingTask,
    EntityMatchingTask
)
from .rdf_tasks import (
    extract_rdf_triples_from_text,
    validate_rdf_triple,
    map_json_to_rdf_schema,
    generate_rdf_queries,
    RDFExtractionTask,
    RDFValidationTask,
    RDFMappingTask
)
from .config import (
    config_manager,
    COMMON_NAMESPACES,
    ONTOLOGY_CONSTRAINTS,
    get_namespace_prefixes,
    get_ontology_constraints,
    create_task_pipeline_config
)

# Legacy compatibility - these were the original function signatures
def send_prompt(prompt: str, schema_class):
    """
    Legacy function for backward compatibility.
    
    Args:
        prompt: The text prompt to send to the LLM
        schema_class: The Pydantic model class to validate the response against
        
    Returns:
        Validated Pydantic model instance or None if validation fails
    """
    return get_client_from_env().send_prompt(prompt, schema_class)

# Legacy constants for backward compatibility
ENDPOINT_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:27B"

# Legacy example data for backward compatibility
JSON = """
{
    "genres" : ["Action", "Adventure", "Sci-Fi"],
    "id" : "tt1375666",
    "involvedPeople" : [
       {
          "birthYear" : 1974,
          "category" : "actor",
          "id" : "nm0000138",
          "ordering" : 1,
          "primaryName" : "Leonardo DiCaprio"
       },
       {
          "birthYear" : 1981,
          "category" : "actor",
          "id" : "nm0330687",
          "ordering" : 2,
          "primaryName" : "Joseph Gordon-Levitt"
       },
       {
          "birthYear" : 1987,
          "category" : "actor",
          "id" : "nm0680983",
          "ordering" : 3,
          "primaryName" : "Elliot Page"
       },
       {
          "birthYear" : 1959,
          "category" : "actor",
          "id" : "nm0913822",
          "ordering" : 4,
          "primaryName" : "Ken Watanabe"
       },
       {
          "birthYear" : 1970,
          "category" : "director",
          "id" : "nm0634240",
          "ordering" : 5,
          "primaryName" : "Christopher Nolan"
       },
       {
          "birthYear" : 1971,
          "category" : "producer",
          "id" : "nm0858799",
          "ordering" : 6,
          "primaryName" : "Emma Thomas"
       },
       {
          "birthYear" : 1957,
          "category" : "composer",
          "id" : "nm0001877",
          "ordering" : 7,
          "primaryName" : "Hans Zimmer"
       },
       {
          "birthYear" : 1961,
          "category" : "cinematographer",
          "id" : "nm0002892",
          "ordering" : 8,
          "primaryName" : "Wally Pfister"
       },
       {
          "birthYear" : 1960,
          "category" : "editor",
          "id" : "nm0809059",
          "ordering" : 9,
          "primaryName" : "Lee Smith"
       },
       {
          "category" : "production_designer",
          "id" : "nm0245596",
          "ordering" : 10,
          "primaryName" : "Guy Hendrix Dyas"
       }
    ],
    "isAdult" : 0,
    "originalTitle" : "Inception",
    "primaryTitle" : "Inception",
    "runtimeMinutes" : 148,
    "startYear" : 2010,
    "titleType" : "movie"
}
"""

ONTOLOGY = """
Movie {
    id: string
    title: string
    genres: list[string]
    actors: list[Person]
    directors: list[Person]
    writers: list[Person]
    producers: list[Person]
    cinematographers: list[Person]
    editors: list[Person]
    production_designers: list[Person]
    composers: list[Person]
}

Person {
    id: string
    name: string
    birthYear: int
}
"""

# Legacy functions for backward compatibility
def ontology_matching():
    """Legacy function placeholder."""
    pass

# Example usage for backward compatibility
if __name__ == "__main__":
    # Example: Extract triplets from text
    text = "Ollama is 22 years old and is busy saving the world."
    result = extract_triplets_from_text(text)
    if result:
        print("Extracted triplets:")
        for triple in result.triplets:
            print(f"  {triple.subject} -- {triple.predicate} --> {triple.object}")
    
    # Example: Generate transformation mapping
    result = generate_transformation_mapping(JSON, ONTOLOGY)
    if result:
        print("\nGenerated mappings:")
        for mapping in result.mappings:
            print(f"  {mapping.key} -> {mapping.link} ({mapping.rule})")
    
    # Example: Check extraction result
    abstract = """
    Casino Royale is a 1967 spy parody film originally distributed by Columbia Pictures featuring an ensemble cast. 
    It is loosely based on the 1953 novel of the same name by Ian Fleming, the first novel to feature the character James Bond.
    """
    triple = {"subject": "Casino Royale", "predicate": "release date", "object": "1967"}
    result = check_extraction_result(abstract, triple)
    if result:
        print(f"\nValidation result: {result.contained}")
        print(f"Triple: {result.triple.subject} -- {result.triple.predicate} --> {result.triple.object}")

# Export all the main components for easy access
__all__ = [
    # Core components
    "LLMClient", "BaseTask", "default_client",
    
    # Models
    "Triplet", "Triplets",
    "TransformationMapping", "TransformationMappings",
    "SimpleTriplet", "TextExtractionValidation",
    "OntologyMapping", "OntologyMappings",
    "EntityMatch", "EntityMatches",
    "SchemaMapping", "SchemaMappings",
    "RDFTriple", "RDFGraph",
    
    # Task functions
    "extract_triplets_from_text",
    "generate_transformation_mapping",
    "check_extraction_result",
    "match_ontology_to_schema",
    "validate_entity_matches",
    "generate_schema_mappings",
    
    # RDF tasks
    "extract_rdf_triples_from_text",
    "validate_rdf_triple",
    "map_json_to_rdf_schema",
    "generate_rdf_queries",
    
    # Task classes
    "TextExtractionTask",
    "SchemaMappingTask", 
    "EntityMatchingTask",
    "RDFExtractionTask",
    "RDFValidationTask",
    "RDFMappingTask",
    
    # Configuration
    "config_manager",
    "COMMON_NAMESPACES",
    "ONTOLOGY_CONSTRAINTS",
    "get_namespace_prefixes",
    "get_ontology_constraints",
    "create_task_pipeline_config",
    
    # Legacy compatibility
    "send_prompt",
    "ENDPOINT_URL",
    "MODEL_NAME",
    "JSON",
    "ONTOLOGY",
    "ontology_matching"
]