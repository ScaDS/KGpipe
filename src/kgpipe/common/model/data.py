from __future__ import annotations

import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union, Type
import json
from uuid import uuid4
import logging
import shutil
from rdflib import Graph
from pydantic import BaseModel, field_validator
from pydantic_core import core_schema

# Format descriptions for built-in formats
FORMAT_DESCRIPTIONS = {
    "ttl": "Turtle RDF format",
    "nquads": "N-Quads RDF format",
    "json": "JSON format",
    "csv": "CSV format",
    "parquet": "Parquet format",
    "xml": "XML format",
    "rdf": "RDF format",
    "jsonld": "JSON-LD format",
    "txt": "Text format",
    "paris_csv": "Paris CSV format",
    "openrefine_json": "OpenRefine JSON format",
    "limes_xml": "LIMES XML format",
    "spotlight_json": "DBpedia Spotlight JSON format",
    "falcon_json": "FALCON JSON format",
    "ie_json": "Information Extraction JSON format",
    "valentine_json": "Valentine JSON format",
    "corenlp_json": "CoreNLP JSON format",
    "openie_json": "OpenIE JSON format",
    "agreementmaker_rdf": "AgreementMaker RDF format",
    "em_json": "Entity Matching JSON format",
}


class DataFormat(Enum):
    """Built-in data formats with enum benefits."""
    # Standard formats
    RDF_TTL = "ttl"
    RDF_NQUADS = "nq"
    RDF_NTRIPLES = "nt"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    RDF_XML = "xml"
    RDF = "rdf"
    RDF_JSONLD = "jsonld"
    TEXT = "txt"
    XML = "xml"
    ANY = "any"
    
    # Tool-specific formats
    PARIS_CSV = "paris.csv"
    OPENREFINE_JSON = "openrefine.json"
    LIMES_XML = "limes.xml"
    SPOTLIGHT_JSON = "spotlight.json"
    FALCON_JSON = "falcon.json"
    VALENTINE_JSON = "valentine.json"
    CORENLP_JSON = "corenlp.json"
    OPENIE_JSON = "openie.json"
    AGREEMENTMAKER_RDF = "agreementmaker.rdf"

    # Exchange formats
    ER_JSON = "er.json" # Entity Resolution JSON format
    TE_JSON = "te.json" # Text Extraction JSON format

    # LLM Tasks
    JSON_ONTO_MAPPING_JSON = "json_onto_mapping.json"

    @classmethod
    def from_extension(cls, extension: str) -> DataFormat:
        """Get a format by file extension. If fails print available formats and raise ValueError."""
        try:
            return cls(extension)
        except ValueError:
            print(f"Available formats: {[f.value for f in cls]}")
            raise ValueError(f"Invalid format: {extension}")


    @property
    def extension(self) -> str:
        """Get the file extension for this format."""
        return self.value
    
    @property
    def description(self) -> str:
        """Get the description for this format."""
        return FORMAT_DESCRIPTIONS.get(self.value, self.value)
    
    @property
    def is_tool_specific(self) -> bool:
        """Check if this is a tool-specific format."""
        tool_specific_formats = {
            "paris_csv", "openrefine_json", "limes_xml", "spotlight_json",
            "falcon_json", "ie_json", "valentine_json", "corenlp_json",
            "openie_json", "agreementmaker_rdf", "em_json"
        }
        return self.value in tool_specific_formats

    def __str__(self) -> str:
        return f".{self.value}"
    
    def __repr__(self) -> str:
        return f".{self.value}"


class DynamicFormat:
    """Dynamic format for submodules to register custom formats."""
    
    def __init__(self, name: str, extension: str, description: str, is_tool_specific: bool = False):
        self.name = name
        self.extension = extension
        self.description = description
        self.is_tool_specific = is_tool_specific

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler) -> Any:
        """Provide Pydantic schema for this type."""
        return core_schema.union_schema([
            core_schema.is_instance_schema(cls),
            core_schema.str_schema()
        ])
    
    @property
    def value(self) -> str:
        """Get the format value (same as name for compatibility)."""
        return self.name
    
    def __eq__(self, other) -> bool:
        """Compare formats by name."""
        if isinstance(other, DynamicFormat):
            return self.name == other.name
        elif isinstance(other, DataFormat):
            return self.name == other.value
        elif isinstance(other, str):
            return self.name == other
        return False
    
    def __hash__(self) -> int:
        """Hash based on name."""
        return hash(self.name)
    
    def __str__(self) -> str:
        return f"DynamicFormat({self.name})"
    
    def __repr__(self) -> str:
        return f"DynamicFormat(name='{self.name}', extension='{self.extension}', description='{self.description}', is_tool_specific={self.is_tool_specific})"


class FormatRegistry:
    """Registry for managing and discovering data formats."""
    
    _dynamic_formats: Dict[str, DynamicFormat] = {}
    
    @classmethod
    def register_format(cls, name: str, extension: str, description: str, is_tool_specific: bool = False) -> DynamicFormat:
        """Register a new dynamic data format."""
        if name in cls._dynamic_formats:
            return cls._dynamic_formats[name]
        
        format_obj = DynamicFormat(name, extension, description, is_tool_specific)
        cls._dynamic_formats[name] = format_obj
        return format_obj
    
    @classmethod
    def get_format(cls, name: str) -> Optional[Union[DataFormat, DynamicFormat]]:
        """Get a format by name, checking built-in formats first."""
        # Try built-in formats first
        try:
            return DataFormat(name)
        except ValueError:
            # Then check dynamic formats
            return cls._dynamic_formats.get(name)
    
    @classmethod
    def list_formats(cls, tool_specific_only: bool = False) -> List[Union[DataFormat, DynamicFormat]]:
        """List all registered formats."""
        formats = list(DataFormat) + list(cls._dynamic_formats.values())
        if tool_specific_only:
            formats = [f for f in formats if getattr(f, 'is_tool_specific', False)]
        return formats
    
    @classmethod
    def list_standard_formats(cls) -> List[Union[DataFormat, DynamicFormat]]:
        """List all standard (non-tool-specific) formats."""
        formats = list(DataFormat) + list(cls._dynamic_formats.values())
        return [f for f in formats if not getattr(f, 'is_tool_specific', False)]
    
    @classmethod
    def list_tool_specific_formats(cls) -> List[Union[DataFormat, DynamicFormat]]:
        """List all tool-specific formats."""
        formats = list(DataFormat) + list(cls._dynamic_formats.values())
        return [f for f in formats if getattr(f, 'is_tool_specific', False)]
    
    @classmethod
    def list_rdf_formats(cls) -> List[Union[DataFormat, DynamicFormat]]:
        """List all RDF formats."""
        rdf_formats = [DataFormat.RDF_TTL, DataFormat.RDF_NQUADS, DataFormat.RDF, DataFormat.RDF_JSONLD]
        dynamic_rdf = [f for f in cls._dynamic_formats.values() if 'rdf' in f.name.lower() or 'ttl' in f.name.lower()]
        return rdf_formats + dynamic_rdf
    
    @classmethod
    def list_text_formats(cls) -> List[Union[DataFormat, DynamicFormat]]:
        """List all text formats."""
        text_formats = [DataFormat.JSON, DataFormat.CSV, DataFormat.XML, DataFormat.TEXT]
        dynamic_text = [f for f in cls._dynamic_formats.values() if f.name.lower() in ['json', 'csv', 'xml', 'txt', 'yaml']]
        return text_formats + dynamic_text
    
    @classmethod
    def clear_dynamic_formats(cls) -> None:
        """Clear all dynamically registered formats (useful for testing)."""
        cls._dynamic_formats.clear()


# Type alias for any format
Format = Union[DataFormat, DynamicFormat]

class Data(BaseModel):
    """Represents a data file with a specific format."""
    path: Path
    format: Format
    
    def __init__(self, *args, **data):
        # Handle positional arguments for backward compatibility
        if args:
            if len(args) == 2:
                data['path'] = args[0]
                data['format'] = args[1]
            elif len(args) == 1:
                data['path'] = args[0]
        
        if 'path' in data and isinstance(data['path'], str):
            data['path'] = Path(data['path'])
        super().__init__(**data)
    
    @field_validator('format', mode='before')
    @classmethod
    def validate_format(cls, v):
        """Convert string format to proper Format object."""
        if isinstance(v, str):
            # Try to convert string to DataFormat enum
            try:
                return DataFormat(v)
            except ValueError:
                # If it's not a DataFormat, it might be a DynamicFormat
                from .models import FormatRegistry
                dynamic_format = FormatRegistry.get_format(v)
                if dynamic_format:
                    return dynamic_format
                raise ValueError(f"Unknown format: {v}")
        return v
    
    def exists(self) -> bool:
        """Check if the data file exists."""
        return self.path.exists()

    def to_dict(self) -> Dict[str, str]:
        return {
            "path": str(self.path),
            "format": self.format.value
        }
    
    def __str__(self) -> str:
        return f"Data({self.path}, {self.format.value})"
    
    def __eq__(self, other):
        """Custom equality to handle format comparison."""
        if not isinstance(other, Data):
            return False
        return (self.path == other.path and 
                (hasattr(self.format, 'value') and hasattr(other.format, 'value') and 
                 self.format.value == other.format.value))





@dataclass
class DataSet:
    """Represents a dataset that can be used as input to a pipeline or stage."""
    id: str
    name: str
    path: Path
    format: Format
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
    
    def exists(self) -> bool:
        """Check if the dataset file exists."""
        return self.path.exists()
    
    def __str__(self) -> str:
        return f"DataSet({self.name}, {self.path}, {self.format.value})"
