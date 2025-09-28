"""
Data integration tasks for text, JSON, and RDF processing.
"""

import json
import time
from typing import Optional, Dict, Any
from .core import LLMClient, BaseTask, default_client
from .models import (
    Triplets, TransformationMappings, TextExtractionValidation,
    OntologyMappings, EntityMatches, SchemaMappings
)

# Text extraction tasks
def extract_triplets_from_text(text: str, client: Optional[LLMClient] = None) -> Optional[Triplets]:
    """
    Extract RDF triplets from unstructured text using LLM.
    
    Args:
        text: Input text to extract triplets from
        client: LLM client instance (uses default if None)
        
    Returns:
        Triplets object containing extracted triplets or None if extraction fails
    """
    client = client or default_client
    prompt = f"""Extract RDF triplets from the following text. 
    For each fact, identify the subject, predicate, and object.
    Return the results as a JSON object with a 'triplets' array.
    
    Text: {text}
    
    Respond with a list of triplets in JSON format."""
    
    return client.send_prompt(prompt, Triplets)

def check_extraction_result(text: str, triple: Dict[str, str], 
                          client: Optional[LLMClient] = None) -> Optional[TextExtractionValidation]:
    """
    Validate if a specific triple can be extracted from given text.
    
    Args:
        text: Input text to check
        triple: Dictionary with 'subject', 'predicate', 'object' keys
        client: LLM client instance (uses default if None)
        
    Returns:
        TextExtractionValidation object or None if validation fails
    """
    client = client or default_client
    prompt = f"""Check if the following text contains the specified triple.
    
    Text: {text}
    
    Triple to check: {json.dumps(triple)}
    
    Determine if this triple can be extracted from the text and respond with:
    - text: the input text
    - triple: the triple being checked  
    - contained: true/false indicating if the triple is contained in the text"""
    
    return client.send_prompt(prompt, TextExtractionValidation)

# Schema mapping tasks
def generate_transformation_mapping(json_schema: str, ontology: str, 
                                 client: Optional[LLMClient] = None) -> Optional[TransformationMappings]:
    """
    Generate transformation mappings from JSON schema to target ontology.
    
    Args:
        json_schema: JSON schema or example data
        ontology: Target ontology definition
        client: LLM client instance (uses default if None)
        
    Returns:
        TransformationMappings object or None if mapping generation fails
    """
    client = client or default_client
    prompt = f"""Generate transformation mappings from the following JSON schema to the target ontology.
    
    JSON Schema/Data:
    {json_schema}
    
    Target Ontology:
    {ontology}
    
    For each field in the JSON, create a mapping that specifies:
    - key: the source field in JSON
    - link: the target property/class in the ontology
    - rule: the transformation rule or mapping logic
    
    Respond with a list of mappings in JSON format."""
    
    return client.send_prompt(prompt, TransformationMappings)

def match_ontology_to_schema(source_schema: str, target_ontology: str,
                           client: Optional[LLMClient] = None) -> Optional[OntologyMappings]:
    """
    Match source schema fields to target ontology classes and properties.
    
    Args:
        source_schema: Source schema definition
        target_ontology: Target ontology definition
        client: LLM client instance (uses default if None)
        
    Returns:
        OntologyMappings object or None if matching fails
    """
    client = client or default_client
    prompt = f"""Match the source schema fields to the target ontology.
    
    Source Schema:
    {source_schema}
    
    Target Ontology:
    {target_ontology}
    
    For each field in the source schema, create a mapping that specifies:
    - source_field: the field in source schema
    - target_class: the target class in ontology
    - target_property: the target property in ontology
    - mapping_type: type of mapping (direct, transform, etc.)
    - confidence: confidence score (0.0 to 1.0)
    
    Respond with a list of ontology mappings in JSON format."""
    
    return client.send_prompt(prompt, OntologyMappings)

# Entity matching tasks
def validate_entity_matches(source_entities: list, target_entities: list,
                          client: Optional[LLMClient] = None) -> Optional[EntityMatches]:
    """
    Validate entity matches between source and target entity lists.
    
    Args:
        source_entities: List of source entity identifiers
        target_entities: List of target entity identifiers
        client: LLM client instance (uses default if None)
        
    Returns:
        EntityMatches object or None if validation fails
    """
    client = client or default_client
    prompt = f"""Validate entity matches between source and target entities.
    
    Source Entities: {json.dumps(source_entities)}
    Target Entities: {json.dumps(target_entities)}
    
    For each potential match, provide:
    - source_entity: source entity identifier
    - target_entity: target entity identifier
    - similarity_score: similarity score (0.0 to 1.0)
    - match_type: type of match (exact, fuzzy, etc.)
    - confidence: confidence in the match (0.0 to 1.0)
    
    Respond with a list of entity matches in JSON format."""
    
    return client.send_prompt(prompt, EntityMatches)

# Schema integration tasks
def generate_schema_mappings(source_schema: str, target_schema: str,
                           client: Optional[LLMClient] = None) -> Optional[SchemaMappings]:
    """
    Generate comprehensive schema mappings for data integration.
    
    Args:
        source_schema: Source schema definition
        target_schema: Target schema definition
        client: LLM client instance (uses default if None)
        
    Returns:
        SchemaMappings object or None if mapping generation fails
    """
    client = client or default_client
    prompt = f"""Generate comprehensive schema mappings for data integration.
    
    Source Schema:
    {source_schema}
    
    Target Schema:
    {target_schema}
    
    Create mappings that include:
    - source_schema: source schema definition
    - target_schema: target schema definition
    - field_mappings: detailed field-level mappings
    - validation_rules: validation rules for the mapping
    
    Respond with schema mappings in JSON format."""
    
    return client.send_prompt(prompt, SchemaMappings)

# Task classes for more complex operations
class TextExtractionTask(BaseTask[Triplets]):
    """Task for extracting structured data from text."""
    
    def execute(self, text: str, extraction_type: str = "triplets") -> Optional[Triplets]:
        """Execute text extraction task."""
        if extraction_type == "triplets":
            return extract_triplets_from_text(text, self.client)
        else:
            raise ValueError(f"Unsupported extraction type: {extraction_type}")

class SchemaMappingTask(BaseTask[TransformationMappings]):
    """Task for mapping between different schemas."""
    
    def execute(self, source_schema: str, target_schema: str) -> Optional[TransformationMappings]:
        """Execute schema mapping task."""
        return generate_transformation_mapping(source_schema, target_schema, self.client)

class EntityMatchingTask(BaseTask[EntityMatches]):
    """Task for entity matching and validation."""
    
    def execute(self, source_entities: list, target_entities: list) -> Optional[EntityMatches]:
        """Execute entity matching task."""
        return validate_entity_matches(source_entities, target_entities, self.client)

# Utility functions
def create_integration_pipeline(tasks: list) -> Dict[str, Any]:
    """
    Create a data integration pipeline from a list of tasks.
    
    Args:
        tasks: List of task configurations
        
    Returns:
        Pipeline configuration dictionary
    """
    pipeline = {
        "tasks": tasks,
        "created_at": time.time(),
        "status": "configured"
    }
    return pipeline

def validate_integration_result(result: Dict[str, Any]) -> bool:
    """
    Validate the result of an integration task.
    
    Args:
        result: Integration result dictionary
        
    Returns:
        True if validation passes, False otherwise
    """
    required_fields = ["task_id", "success", "output_data"]
    return all(field in result for field in required_fields) and result.get("success", False) 