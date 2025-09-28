"""
LLM-based data integration tasks for Knowledge Graph construction.

This module provides structured LLM tasks for integrating text, JSON, and RDF data
into existing RDF Knowledge Graphs using Pydantic for structured output validation.
"""

from .tasks import *

# from .core import LLMClient, BaseTask
# from .models import (
#     Triplet, Triplets, 
#     TransformationMapping, TransformationMappings,
#     SimpleTriplet, TextExtractionValidation,
#     OntologyMapping, OntologyMappings,
#     EntityMatch, EntityMatches,
#     SchemaMapping, SchemaMappings
# )
# from .tasks import (
#     extract_triplets_from_text,
#     generate_transformation_mapping,
#     check_extraction_result,
#     match_ontology_to_schema,
#     validate_entity_matches,
#     generate_schema_mappings
# )

# __all__ = [
#     # Core components
#     "LLMClient", "BaseTask",
    
#     # Models
#     "Triplet", "Triplets",
#     "TransformationMapping", "TransformationMappings", 
#     "SimpleTriplet", "TextExtractionValidation",
#     "OntologyMapping", "OntologyMappings",
#     "EntityMatch", "EntityMatches",
#     "SchemaMapping", "SchemaMappings",
    
#     # Tasks
#     "extract_triplets_from_text",
#     "generate_transformation_mapping", 
#     "check_extraction_result",
#     "match_ontology_to_schema",
#     "validate_entity_matches",
#     "generate_schema_mappings"
# ] 