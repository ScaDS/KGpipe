"""
Pydantic models for raw parameter extraction results.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    """Types of sources for parameter extraction."""
    CLI = "cli"
    PYTHON_LIB = "python_lib"
    HTTP_API = "http_api"
    DOCKER = "docker"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Methods used for parameter extraction."""
    REGEX = "regex"
    LLM = "llm"
    AUTO = "auto"


class RawParameter(BaseModel):
    """
    Intermediate representation of an extracted parameter.
    This is the raw extraction result before conversion to Parameter model.
    """
    name: str = Field(..., description="Normalized parameter name")
    native_keys: List[str] = Field(default_factory=list, description="Original parameter names/flags from source")
    description: Optional[str] = Field(None, description="Parameter description/documentation")
    type_hint: Optional[str] = Field(None, description="Type hint or type name from source")
    default_value: Optional[Union[str, int, float, bool]] = Field(None, description="Default value if present")
    required: bool = Field(False, description="Whether parameter is required")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Constraints like min, max, allowed_values")
    source: str = Field(..., description="Source text or file path where parameter was found")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about extraction")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "threshold",
                "native_keys": ["--threshold", "-t", "THRESHOLD"],
                "description": "Matching threshold value",
                "type_hint": "float",
                "default_value": 0.5,
                "required": False,
                "constraints": {"minimum": 0.0, "maximum": 1.0},
                "source": "tool.py --help",
                "provenance": {"line_number": 42, "extraction_method": "regex"}
            }
        }
    )


class ExtractionResult(BaseModel):
    """
    Container for extracted parameters with metadata.
    """
    tool_name: str = Field(..., description="Name of the tool/library being analyzed")
    source_type: SourceType = Field(..., description="Type of source (CLI, Python, API, Docker)")
    extraction_method: ExtractionMethod = Field(..., description="Method used for extraction")
    parameters: List[RawParameter] = Field(default_factory=list, description="List of extracted parameters")
    timestamp: datetime = Field(default_factory=datetime.now, description="When extraction was performed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the extraction")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during extraction")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tool_name": "paris_matcher",
                "source_type": "cli",
                "extraction_method": "regex",
                "parameters": [],
                "timestamp": "2024-01-01T00:00:00",
                "metadata": {"source_file": "paris --help"},
                "errors": []
            }
        }
    )

