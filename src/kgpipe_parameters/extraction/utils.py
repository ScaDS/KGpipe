"""
Utility functions for parameter extraction and conversion.
"""

import re
from typing import Optional, Union, List, Any, Dict
from .models import RawParameter
from kgpipe.common.model.configuration import Parameter, ParameterType


def infer_parameter_type(type_hint: Optional[str], default_value: Any = None) -> ParameterType:
    """
    Infer ParameterType from type hint string or default value.
    
    Args:
        type_hint: Type hint string (e.g., "int", "float", "str", "bool")
        default_value: Default value to infer type from if type_hint is None
        
    Returns:
        ParameterType enum value
    """
    if type_hint:
        type_hint_lower = type_hint.lower().strip()
        
        # Check for boolean
        if any(x in type_hint_lower for x in ["bool", "boolean"]):
            return ParameterType.boolean
        
        # Check for integer
        if any(x in type_hint_lower for x in ["int", "integer"]):
            return ParameterType.integer
        
        # Check for float/number
        if any(x in type_hint_lower for x in ["float", "number", "double", "decimal"]):
            return ParameterType.number
        
        # Check for array/list
        if any(x in type_hint_lower for x in ["list", "array", "[]", "List"]):
            return ParameterType.array
        
        # Check for object/dict
        if any(x in type_hint_lower for x in ["dict", "object", "Dict", "{}"]):
            return ParameterType.object
        
        # Check for enum
        if "enum" in type_hint_lower or "choice" in type_hint_lower:
            return ParameterType.enum
    
    # Infer from default value
    if default_value is not None:
        if isinstance(default_value, bool):
            return ParameterType.boolean
        elif isinstance(default_value, int):
            return ParameterType.integer
        elif isinstance(default_value, float):
            return ParameterType.number
        elif isinstance(default_value, list):
            return ParameterType.array
        elif isinstance(default_value, dict):
            return ParameterType.object
    
    # Default to string
    return ParameterType.string


def parse_default_value(value_str: Optional[str]) -> Optional[Union[str, int, float, bool]]:
    """
    Parse a default value string into appropriate Python type.
    
    Args:
        value_str: String representation of default value
        
    Returns:
        Parsed value (str, int, float, or bool) or None
    """
    if value_str is None:
        return None
    
    value_str = value_str.strip().strip('"').strip("'")
    
    # Try boolean
    if value_str.lower() in ["true", "false", "yes", "no", "1", "0"]:
        return value_str.lower() in ["true", "yes", "1"]
    
    # Try integer
    try:
        if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
            return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


def normalize_parameter_name(name: str) -> str:
    """
    Normalize parameter name to a standard format.
    
    Args:
        name: Original parameter name (may include --, -, etc.)
        
    Returns:
        Normalized name (lowercase, underscores instead of hyphens)
    """
    # Remove leading dashes and spaces
    name = name.lstrip("-").lstrip()
    
    # Replace hyphens with underscores
    name = name.replace("-", "_")
    
    # Convert to lowercase
    name = name.lower()
    
    # Remove special characters except underscores
    name = re.sub(r"[^a-z0-9_]", "", name)
    
    return name


def extract_constraints(description: Optional[str], type_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract constraints (min, max, allowed_values) from description or type hint.
    
    Args:
        description: Parameter description text
        type_hint: Type hint string
        
    Returns:
        Dictionary with constraint information
    """
    constraints = {}
    
    if not description:
        return constraints
    
    # Extract min/max values - try combined first, then separate
    min_max_pattern = re.compile(r"(?:min|minimum)[=:]\s*([0-9.]+).*(?:max|maximum)[=:]\s*([0-9.]+)", re.IGNORECASE)
    min_max_match = min_max_pattern.search(description)
    if min_max_match:
        constraints["minimum"] = float(min_max_match.group(1))
        constraints["maximum"] = float(min_max_match.group(2))
    
    # Try separate min and max (even if combined pattern didn't match)
    # More flexible pattern to handle "Minimum value: 10" or "min: 10" formats
    min_pattern = re.compile(r"(?:min|minimum)(?:\s+value)?[=:]\s*([0-9.]+)", re.IGNORECASE)
    max_pattern = re.compile(r"(?:max|maximum)(?:\s+value)?[=:]\s*([0-9.]+)", re.IGNORECASE)
    min_match = min_pattern.search(description)
    max_match = max_pattern.search(description)
    if min_match and "minimum" not in constraints:
        constraints["minimum"] = float(min_match.group(1))
    if max_match and "maximum" not in constraints:
        constraints["maximum"] = float(max_match.group(1))
    
    # Extract allowed values / choices
    choices_pattern = re.compile(r"(?:choices|enum|options|allowed)[=:]\s*\[([^\]]+)\]", re.IGNORECASE)
    choices_match = choices_pattern.search(description)
    if choices_match:
        choices_str = choices_match.group(1)
        # Split by comma and clean up
        choices = [c.strip().strip('"').strip("'") for c in choices_str.split(",")]
        constraints["allowed_values"] = choices
    
    return constraints


def to_parameter_model(raw_param: RawParameter) -> Parameter:
    """
    Convert a RawParameter to a Parameter model.
    
    Args:
        raw_param: RawParameter instance
        
    Returns:
        Parameter model instance
    """
    # Infer parameter type
    param_type = infer_parameter_type(raw_param.type_hint, raw_param.default_value)
    
    # Parse default value
    default_val = raw_param.default_value
    if isinstance(default_val, str):
        default_val = parse_default_value(default_val)
    
    # Ensure default value matches the inferred type
    if default_val is None:
        # Set appropriate default based on type
        if param_type == ParameterType.boolean:
            default_val = False
        elif param_type == ParameterType.integer:
            default_val = 0
        elif param_type == ParameterType.number:
            default_val = 0.0
        elif param_type == ParameterType.string:
            default_val = ""
        elif param_type == ParameterType.array:
            default_val = []
        elif param_type == ParameterType.object:
            default_val = {}
    
    # Extract constraints
    constraints = extract_constraints(raw_param.description, raw_param.type_hint)
    constraints.update(raw_param.constraints)
    
    # Get allowed values
    allowed_values = constraints.get("allowed_values", [])
    if allowed_values:
        # Convert to appropriate types
        typed_allowed = []
        for val in allowed_values:
            parsed = parse_default_value(str(val))
            typed_allowed.append(parsed if parsed is not None else str(val))
        allowed_values = typed_allowed
    
    # Ensure native_keys includes the name
    native_keys = list(raw_param.native_keys)
    if raw_param.name not in native_keys:
        native_keys.insert(0, raw_param.name)
    
    return Parameter(
        name=raw_param.name,
        native_keys=native_keys,
        datatype=param_type,
        default_value=default_val,
        required=raw_param.required,
        allowed_values=allowed_values,
        minimum=constraints.get("minimum"),
        maximum=constraints.get("maximum"),
        unit=constraints.get("unit"),
    )

