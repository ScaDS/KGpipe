"""
Configuration management for LLM-based data integration tasks.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel

@dataclass
class LLMConfig:
    """Configuration for LLM client settings."""
    endpoint_url: str = "http://localhost:11434/api/generate"
    model_name: str = "gemma3:27B"
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.1
    max_tokens: Optional[int] = None

@dataclass
class TaskConfig:
    """Configuration for task execution settings."""
    batch_size: int = 10
    parallel_execution: bool = False
    max_workers: int = 4
    validation_enabled: bool = True
    logging_level: str = "INFO"

@dataclass
class IntegrationConfig:
    """Configuration for data integration pipeline."""
    source_format: str = "json"
    target_format: str = "rdf"
    namespace_prefixes: Dict[str, str] = field(default_factory=dict)
    ontology_constraints: Dict[str, Any] = field(default_factory=dict)
    validation_rules: list = field(default_factory=list)

class ConfigManager:
    """Manager for configuration settings across the module."""
    
    def __init__(self):
        self.llm_config = LLMConfig()
        self.task_config = TaskConfig()
        self.integration_config = IntegrationConfig()
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # LLM Configuration
        if os.getenv("LLM_ENDPOINT_URL"):
            self.llm_config.endpoint_url = os.getenv("LLM_ENDPOINT_URL")
        if os.getenv("LLM_MODEL_NAME"):
            self.llm_config.model_name = os.getenv("LLM_MODEL_NAME")
        if os.getenv("LLM_TIMEOUT"):
            self.llm_config.timeout = int(os.getenv("LLM_TIMEOUT"))
        
        # Task Configuration
        if os.getenv("TASK_BATCH_SIZE"):
            self.task_config.batch_size = int(os.getenv("TASK_BATCH_SIZE"))
        if os.getenv("TASK_PARALLEL_EXECUTION"):
            self.task_config.parallel_execution = os.getenv("TASK_PARALLEL_EXECUTION").lower() == "true"
        if os.getenv("TASK_MAX_WORKERS"):
            self.task_config.max_workers = int(os.getenv("TASK_MAX_WORKERS"))
    
    def update_llm_config(self, **kwargs):
        """Update LLM configuration."""
        for key, value in kwargs.items():
            if hasattr(self.llm_config, key):
                setattr(self.llm_config, key, value)
    
    def update_task_config(self, **kwargs):
        """Update task configuration."""
        for key, value in kwargs.items():
            if hasattr(self.task_config, key):
                setattr(self.task_config, key, value)
    
    def update_integration_config(self, **kwargs):
        """Update integration configuration."""
        for key, value in kwargs.items():
            if hasattr(self.integration_config, key):
                setattr(self.integration_config, key, value)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            "llm": self.llm_config.__dict__,
            "task": self.task_config.__dict__,
            "integration": self.integration_config.__dict__
        }

# Global configuration instance
config_manager = ConfigManager()

# Predefined namespace prefixes for common ontologies
COMMON_NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "schema": "http://schema.org/",
    "dbpedia": "http://dbpedia.org/resource/",
    "dbpedia-owl": "http://dbpedia.org/ontology/"
}

# Predefined ontology constraints for common domains
ONTOLOGY_CONSTRAINTS = {
    "movie": {
        "classes": ["Movie", "Person", "Genre", "Company"],
        "properties": {
            "Movie": ["title", "releaseDate", "genre", "director", "actor"],
            "Person": ["name", "birthDate", "deathDate"],
            "Genre": ["name", "description"],
            "Company": ["name", "foundedDate", "type"]
        },
        "data_types": {
            "title": "string",
            "releaseDate": "date",
            "genre": "string",
            "name": "string",
            "birthDate": "date"
        }
    },
    "scientific": {
        "classes": ["Paper", "Author", "Journal", "Institution"],
        "properties": {
            "Paper": ["title", "abstract", "publicationDate", "doi"],
            "Author": ["name", "affiliation", "email"],
            "Journal": ["name", "issn", "publisher"],
            "Institution": ["name", "country", "type"]
        },
        "data_types": {
            "title": "string",
            "abstract": "text",
            "publicationDate": "date",
            "doi": "string",
            "name": "string"
        }
    }
}

def get_namespace_prefixes(domain: Optional[str] = None) -> Dict[str, str]:
    """
    Get namespace prefixes for a specific domain or return common ones.
    
    Args:
        domain: Optional domain-specific namespace prefixes
        
    Returns:
        Dictionary of namespace prefixes
    """
    if domain and domain in COMMON_NAMESPACES:
        return {domain: COMMON_NAMESPACES[domain]}
    return COMMON_NAMESPACES

def get_ontology_constraints(domain: str) -> Optional[Dict[str, Any]]:
    """
    Get ontology constraints for a specific domain.
    
    Args:
        domain: Domain name (e.g., "movie", "scientific")
        
    Returns:
        Ontology constraints dictionary or None if not found
    """
    return ONTOLOGY_CONSTRAINTS.get(domain)

def create_task_pipeline_config(source_format: str, target_format: str, 
                              domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a complete pipeline configuration for a data integration task.
    
    Args:
        source_format: Source data format (json, text, rdf)
        target_format: Target data format (json, text, rdf)
        domain: Optional domain for specialized configuration
        
    Returns:
        Complete pipeline configuration dictionary
    """
    config = {
        "source_format": source_format,
        "target_format": target_format,
        "namespace_prefixes": get_namespace_prefixes(domain),
        "ontology_constraints": get_ontology_constraints(domain) if domain else {},
        "llm_config": config_manager.llm_config.__dict__,
        "task_config": config_manager.task_config.__dict__
    }
    
    return config 