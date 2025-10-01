"""
KGbench Vocabularies

This module defines controlled vocabularies for the KGbench ontology,
including data formats, task categories, execution statuses, and metric types.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class VocabularyTerm:
    """A term in a controlled vocabulary."""
    iri: str
    label: str
    description: Optional[str] = None
    broader: Optional[str] = None
    narrower: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    deprecated: bool = False


@dataclass
class ControlledVocabulary:
    """A controlled vocabulary with terms."""
    name: str
    description: str
    namespace: str
    terms: Dict[str, VocabularyTerm] = field(default_factory=dict)
    
    def add_term(self, term: VocabularyTerm) -> None:
        """Add a term to the vocabulary."""
        self.terms[term.iri] = term
    
    def get_term(self, iri: str) -> Optional[VocabularyTerm]:
        """Get a term by IRI."""
        return self.terms.get(iri)
    
    def list_terms(self) -> List[VocabularyTerm]:
        """List all terms in the vocabulary."""
        return list(self.terms.values())


class DataFormatVocabulary(ControlledVocabulary):
    """Vocabulary for data formats supported by KGbench."""
    
    def __init__(self):
        super().__init__(
            name="Data Format Vocabulary",
            description="Controlled vocabulary for data formats supported by KGbench",
            namespace="http://kgpipe.org/vocabularies/data-format#"
        )
        
        # Standard formats
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}RDF_TURTLE",
            label="RDF Turtle",
            description="RDF data in Turtle format"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}RDF_NTRIPLES",
            label="RDF N-Triples",
            description="RDF data in N-Triples format"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}RDF_JSONLD",
            label="RDF JSON-LD",
            description="RDF data in JSON-LD format"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}CSV",
            label="CSV",
            description="Comma-separated values format"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}TSV",
            label="TSV",
            description="Tab-separated values format"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}JSON",
            label="JSON",
            description="JavaScript Object Notation format"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}XML",
            label="XML",
            description="Extensible Markup Language format"
        ))
        
        # Special formats
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}PARIS_CSV",
            label="Paris CSV",
            description="Special CSV format for Paris matcher output"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}JEDAI_JSON",
            label="JedAI JSON",
            description="Special JSON format for JedAI matcher output"
        ))


class TaskCategoryVocabulary(ControlledVocabulary):
    """Vocabulary for task categories in KGbench."""
    
    def __init__(self):
        super().__init__(
            name="Task Category Vocabulary",
            description="Controlled vocabulary for task categories in KGbench",
            namespace="http://kgpipe.org/vocabularies/task-category#"
        )
        
        # Core categories
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}ENTITY_LINKING",
            label="Entity Linking",
            description="Tasks that link entities to knowledge bases"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}INFORMATION_EXTRACTION",
            label="Information Extraction",
            description="Tasks that extract structured information from text"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}SCHEMA_MATCHING",
            label="Schema Matching",
            description="Tasks that match schemas between datasets"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}ONTOLOGY_MATCHING",
            label="Ontology Matching",
            description="Tasks that match ontologies"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}ENTITY_MATCHING",
            label="Entity Matching",
            description="Tasks that match entities between datasets"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}NER",
            label="Named Entity Recognition",
            description="Tasks that recognize named entities in text"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}CONVERSION",
            label="Conversion",
            description="Tasks that convert between data formats"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}FUSION",
            label="Fusion",
            description="Tasks that fuse multiple datasets"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}MAPPING",
            label="Mapping",
            description="Tasks that create mappings between datasets"
        ))


class ExecutionStatusVocabulary(ControlledVocabulary):
    """Vocabulary for execution statuses."""
    
    def __init__(self):
        super().__init__(
            name="Execution Status Vocabulary",
            description="Controlled vocabulary for execution statuses",
            namespace="http://kgpipe.org/vocabularies/execution-status#"
        )
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}PENDING",
            label="Pending",
            description="Execution is queued and waiting to start"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}RUNNING",
            label="Running",
            description="Execution is currently running"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}COMPLETED",
            label="Completed",
            description="Execution completed successfully"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}FAILED",
            label="Failed",
            description="Execution failed with an error"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}CANCELLED",
            label="Cancelled",
            description="Execution was cancelled"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}TIMEOUT",
            label="Timeout",
            description="Execution timed out"
        ))


class MetricTypeVocabulary(ControlledVocabulary):
    """Vocabulary for metric types."""
    
    def __init__(self):
        super().__init__(
            name="Metric Type Vocabulary",
            description="Controlled vocabulary for metric types",
            namespace="http://kgpipe.org/vocabularies/metric-type#"
        )
        
        # Performance metrics
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}EXECUTION_TIME",
            label="Execution Time",
            description="Time taken to execute a task or pipeline"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}MEMORY_USAGE",
            label="Memory Usage",
            description="Memory consumed during execution"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}CPU_USAGE",
            label="CPU Usage",
            description="CPU utilization during execution"
        ))
        
        # Quality metrics
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}PRECISION",
            label="Precision",
            description="Precision of matching or extraction results"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}RECALL",
            label="Recall",
            description="Recall of matching or extraction results"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}F1_SCORE",
            label="F1 Score",
            description="F1 score combining precision and recall"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}ACCURACY",
            label="Accuracy",
            description="Overall accuracy of results"
        ))
        
        # Data metrics
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}INPUT_SIZE",
            label="Input Size",
            description="Size of input data"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}OUTPUT_SIZE",
            label="Output Size",
            description="Size of output data"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}ENTITIES_PROCESSED",
            label="Entities Processed",
            description="Number of entities processed"
        ))
        
        self.add_term(VocabularyTerm(
            iri=f"{self.namespace}MATCHES_FOUND",
            label="Matches Found",
            description="Number of matches found"
        ))


# Create vocabulary instances
DATA_FORMAT_VOCAB = DataFormatVocabulary()
TASK_CATEGORY_VOCAB = TaskCategoryVocabulary()
EXECUTION_STATUS_VOCAB = ExecutionStatusVocabulary()
METRIC_TYPE_VOCAB = MetricTypeVocabulary()

# Convenience functions
def get_data_format_iri(format_name: str) -> Optional[str]:
    """Get the IRI for a data format by name."""
    for term in DATA_FORMAT_VOCAB.list_terms():
        if term.label.upper().replace(" ", "_") == format_name.upper():
            return term.iri
    return None

def get_task_category_iri(category_name: str) -> Optional[str]:
    """Get the IRI for a task category by name."""
    for term in TASK_CATEGORY_VOCAB.list_terms():
        if term.label.upper().replace(" ", "_") == category_name.upper():
            return term.iri
    return None

def get_execution_status_iri(status_name: str) -> Optional[str]:
    """Get the IRI for an execution status by name."""
    for term in EXECUTION_STATUS_VOCAB.list_terms():
        if term.label.upper() == status_name.upper():
            return term.iri
    return None

def get_metric_type_iri(metric_name: str) -> Optional[str]:
    """Get the IRI for a metric type by name."""
    for term in METRIC_TYPE_VOCAB.list_terms():
        if term.label.upper().replace(" ", "_") == metric_name.upper():
            return term.iri
    return None 