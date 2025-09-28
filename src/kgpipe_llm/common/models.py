"""
Pydantic models for structured output validation in data integration tasks.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# Text extraction models
class Triplet(BaseModel):
    """Represents a subject-predicate-object triple."""
    subject: str = Field(..., description="The subject of the triple")
    predicate: str = Field(..., description="The predicate/relation of the triple")
    object: str = Field(..., description="The object of the triple")

class Triplets(BaseModel):
    """Container for a list of triplets extracted from text."""
    triplets: List[Triplet] = Field(..., description="List of extracted triplets")

class SimpleTriplet(BaseModel):
    """Simplified triplet for validation tasks."""
    subject: str
    predicate: str
    object: str

class TextExtractionValidation(BaseModel):
    """Validation result for text extraction."""
    text: str = Field(..., description="The input text")
    triple: SimpleTriplet = Field(..., description="The triple to validate")
    contained: bool = Field(..., description="Whether the triple is contained in the text")

# Schema mapping models
class TransformationMapping(BaseModel):
    """Mapping between JSON schema and target ontology."""
    key: str = Field(..., description="Source field/key in JSON")
    link: str = Field(..., description="Target property/class in ontology")
    rule: str = Field(..., description="Transformation rule or mapping logic")

class TransformationMappings(BaseModel):
    """Container for transformation mappings."""
    mappings: List[TransformationMapping] = Field(..., description="List of transformation mappings")

# Ontology matching models
class OntologyMapping(BaseModel):
    """Mapping between source schema and target ontology."""
    source_field: str = Field(..., description="Field in source schema")
    target_class: str = Field(..., description="Target class in ontology")
    target_property: str = Field(..., description="Target property in ontology")
    mapping_type: str = Field(..., description="Type of mapping (direct, transform, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the mapping")

class OntologyMappings(BaseModel):
    """Container for ontology mappings."""
    mappings: List[OntologyMapping] = Field(..., description="List of ontology mappings")

# Entity matching models
class EntityMatch(BaseModel):
    """Entity matching result."""
    source_entity: str = Field(..., description="Source entity identifier")
    target_entity: str = Field(..., description="Target entity identifier")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    match_type: str = Field(..., description="Type of match (exact, fuzzy, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the match")

class EntityMatches(BaseModel):
    """Container for entity matches."""
    matches: List[EntityMatch] = Field(..., description="List of entity matches")

# Schema mapping models
class SchemaMapping(BaseModel):
    """Schema mapping for data integration."""
    source_schema: str = Field(..., description="Source schema definition")
    target_schema: str = Field(..., description="Target schema definition")
    field_mappings: List[TransformationMapping] = Field(..., description="Field-level mappings")
    validation_rules: List[str] = Field(default=[], description="Validation rules for the mapping")

class SchemaMappings(BaseModel):
    """Container for schema mappings."""
    mappings: List[SchemaMapping] = Field(..., description="List of schema mappings")

# RDF-specific models
class RDFTriple(BaseModel):
    """RDF triple with URIs."""
    subject: str = Field(..., description="Subject URI")
    predicate: str = Field(..., description="Predicate URI")
    object: str = Field(..., description="Object URI or literal")

class RDFGraph(BaseModel):
    """RDF graph container."""
    triples: List[RDFTriple] = Field(..., description="List of RDF triples")
    namespace_prefixes: dict = Field(default={}, description="Namespace prefixes")

# Integration task models
class IntegrationTask(BaseModel):
    """Configuration for a data integration task."""
    task_type: str = Field(..., description="Type of integration task")
    source_format: str = Field(..., description="Source data format (json, text, rdf)")
    target_format: str = Field(..., description="Target data format")
    mapping_config: dict = Field(default={}, description="Mapping configuration")
    validation_rules: List[str] = Field(default=[], description="Validation rules")

class IntegrationResult(BaseModel):
    """Result of a data integration task."""
    task_id: str = Field(..., description="Task identifier")
    success: bool = Field(..., description="Whether the integration was successful")
    output_data: Optional[dict] = Field(None, description="Output data")
    validation_errors: List[str] = Field(default=[], description="Validation errors")
    processing_time: float = Field(..., description="Processing time in seconds") 