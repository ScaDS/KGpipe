from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, field_validator
from .default_catalog import BasicDataFormats, CustomDataFormats

# Backward-compatible alias used across the codebase.
DataFormat = BasicDataFormats


# Type alias for any format
Format = Union[DataFormat, CustomDataFormats]


def _format_value(fmt: Format) -> str:
    return str(fmt.value)

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
        if isinstance(v, (DataFormat, CustomDataFormats)):
            return v
        if isinstance(v, Enum) and isinstance(v.value, str):
            # Allow user-defined enum values for strong typing/autocomplete.
            return v
        if isinstance(v, str):
            # Try to convert string to DataFormat enum
            try:
                return DataFormat(v)
            except ValueError:
                raise ValueError(f"Unknown format: {v}")
        return v
    
    def exists(self) -> bool:
        """Check if the data file exists."""
        return self.path.exists()

    def to_dict(self) -> Dict[str, str]:
        return {
            "path": str(self.path),
            "format": _format_value(self.format)
        }
    
    def __str__(self) -> str:
        return f"Data({self.path}, {_format_value(self.format)})"
    
    def __eq__(self, other):
        """Custom equality to handle format comparison."""
        if not isinstance(other, Data):
            return False
        return self.path == other.path and _format_value(self.format) == _format_value(other.format)

KgData = Data

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
        return f"DataSet({self.name}, {self.path}, {_format_value(self.format)})"
