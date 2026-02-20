"""
Regex patterns for parameter extraction from various sources.
"""

import re
from typing import Dict, List, Tuple, Optional


# CLI argument patterns
CLI_PATTERNS = {
    # Long form: --param, --param=VALUE, --param VALUE
    "long_flag": re.compile(r"--([a-zA-Z][a-zA-Z0-9_-]*)(?:[=\s]+([^\s]+))?"),
    # Short form: -p, -p VALUE, -pVALUE
    "short_flag": re.compile(r"-([a-zA-Z])(?:\s+([^\s]+))?"),
    # Combined: -p, --param
    "combined_flag": re.compile(r"(-[a-zA-Z]|--[a-zA-Z][a-zA-Z0-9_-]+)"),
    # Description lines (common in help output)
    "description": re.compile(r"^\s+([^\s]+(?:\s+[^\s]+)*)\s+(.+)$"),
    # Required/optional indicators
    "required": re.compile(r"(required|mandatory|must)", re.IGNORECASE),
    "optional": re.compile(r"(optional|\[optional\]|\[default)", re.IGNORECASE),
    # Default values: [default: value], (default: value), default=value
    # Match: (default: 0.5) -> capture "0.5", [default: value] -> capture "value", default=value -> capture "value"
    # The pattern matches "default:" or "default=" and captures the value until closing bracket/paren or end
    "default_value": re.compile(r"default[=:]\s*([^\])]+?)(?:\]|\)|$)", re.IGNORECASE),
    # Type hints: <int>, [str], (float)
    "type_hint": re.compile(r"[<\[\(]([a-zA-Z]+)[>\]\)]"),
}

# Python code patterns
PYTHON_PATTERNS = {
    # Function parameter: param: type = default
    "function_param": re.compile(r"(\w+)(?:\s*:\s*([^=]+))?(?:\s*=\s*([^,)]+))?"),
    # Type hints: param: int, param: Optional[str] = None
    "type_annotation": re.compile(r":\s*([^=,)]+)"),
    # Default values in function signatures
    "default_in_sig": re.compile(r"=\s*([^,)]+)"),
    # Docstring parameter descriptions: :param name: description
    "docstring_param": re.compile(r":param\s+(\w+):\s*(.+?)(?=\n|:param|$)", re.MULTILINE),
    # Docstring type: :type name: type
    "docstring_type": re.compile(r":type\s+(\w+):\s*([^\n]+)"),
    # Class attributes with type hints
    "class_attr": re.compile(r"(\w+)\s*:\s*([^=\n]+)(?:\s*=\s*([^\n]+))?"),
    # Environment variable assignments: VAR = value
    "env_var": re.compile(r"([A-Z_][A-Z0-9_]*)\s*=\s*(.+)"),
}

# HTTP API patterns
API_PATTERNS = {
    # Query parameters: ?param=value
    "query_param": re.compile(r"[?&]([^=&]+)(?:=([^&]+))?"),
    # Path parameters: /{param}/
    "path_param": re.compile(r"/\{([^}]+)\}/"),
    # Header parameters: X-Header-Name: value
    "header": re.compile(r"([A-Z][a-zA-Z0-9-]+):\s*(.+)"),
    # JSON schema properties
    "json_property": re.compile(r'"([^"]+)":\s*\{[^}]*"type":\s*"([^"]+)"'),
    # OpenAPI parameter definitions
    "openapi_param": re.compile(r'"([^"]+)":\s*\{[^}]*"in":\s*"([^"]+)"'),
}

# Docker patterns
DOCKER_PATTERNS = {
    # ENV variable: ENV VAR=value or ENV VAR value
    "env_declaration": re.compile(r"ENV\s+([A-Z_][A-Z0-9_]*)(?:\s*=\s*|\s+)(.+)", re.IGNORECASE),
    # ARG declaration: ARG VAR[=default]
    "arg_declaration": re.compile(r"ARG\s+([A-Z_][A-Z0-9_]*)(?:\s*=\s*([^\s]+))?", re.IGNORECASE),
    # Environment variable in docker-compose: VAR: value
    "compose_env": re.compile(r"([A-Z_][A-Z0-9_]*)\s*:\s*(.+)"),
    # Volume mounts: -v /host:/container
    "volume_mount": re.compile(r"-v\s+([^:\s]+):([^:\s]+)"),
    # Port mappings: -p HOST:CONTAINER
    "port_mapping": re.compile(r"-p\s+(\d+):(\d+)"),
}

# README / documentation patterns
README_PATTERNS = {
    # Flags or options mentioned in code blocks or inline code: --param, -p
    "inline_flag": re.compile(r"`(-{1,2}[a-zA-Z][a-zA-Z0-9_-]*)`"),
    # Command-line invocations in code blocks: tool --param value
    "code_block_flag": re.compile(r"(?:^|\s)(-{1,2}[a-zA-Z][a-zA-Z0-9_-]*)(?:\s+(\S+))?", re.MULTILINE),
    # Environment variable references: $VAR, ${VAR}, ENV VAR, set VAR=
    "env_reference": re.compile(r"(?:\$\{?|(?:set|export)\s+)([A-Z_][A-Z0-9_]*)(?:\}|=([^\s]+))?"),
    # Config key-value in YAML/properties style: key: value  or  key = value
    "config_kv": re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_.]+)\s*[=:]\s*(.+)$", re.MULTILINE),
    # JVM-style flags: -Xmx47000m, -XX:+UseG1GC
    "jvm_flag": re.compile(r"(-X[a-z]+\d*[a-zA-Z]*|-XX:[+\-]?\w+(?:=\S+)?)"),
    # Markdown table rows with parameter-like content: | param | type | description |
    "table_param": re.compile(r"\|\s*`?([a-zA-Z_][a-zA-Z0-9_-]*)`?\s*\|([^|]*)\|([^|]*)\|"),
    # Setting/configuration references: "set X to Y", "configure X as Y"
    "setting_reference": re.compile(
        r"(?:set|configure|specify|use)\s+[`\"']?([a-zA-Z_][a-zA-Z0-9_-]*)[`\"']?\s+(?:to|as|=)\s+[`\"']?([^\s,`\"']+)",
        re.IGNORECASE,
    ),
    # Parameter descriptions in lists: - `param`: description  or  * param — description
    "list_param": re.compile(r"^\s*[-*]\s+`([a-zA-Z_][a-zA-Z0-9_-]*)`[:\s]+(.+)$", re.MULTILINE),
    # Placeholder patterns like <param>, [param], {param} in usage lines
    "placeholder": re.compile(r"<([a-zA-Z_][a-zA-Z0-9_]*)>"),
}

# Common patterns for all sources
COMMON_PATTERNS = {
    # Numeric constraints: min=0, max=100
    "min_max": re.compile(r"(?:min|minimum)[=:]\s*([0-9.]+).*(?:max|maximum)[=:]\s*([0-9.]+)", re.IGNORECASE),
    # Allowed values: choices=[a, b, c] or enum: [a, b, c]
    "allowed_values": re.compile(r"(?:choices|enum|options)[=:]\s*\[([^\]]+)\]", re.IGNORECASE),
    # Boolean flags: true/false, yes/no, 1/0
    "boolean": re.compile(r"(true|false|yes|no|1|0)", re.IGNORECASE),
    # Numeric types: int, float, number
    "numeric": re.compile(r"(int|integer|float|number|double)", re.IGNORECASE),
    # String types: str, string, text
    "string": re.compile(r"(str|string|text)", re.IGNORECASE),
}


def get_patterns(source_type: str) -> Dict[str, re.Pattern]:
    """
    Get regex patterns for a specific source type.
    
    Args:
        source_type: One of 'cli', 'python', 'api', 'docker'
        
    Returns:
        Dictionary of compiled regex patterns
    """
    patterns_map = {
        "cli": CLI_PATTERNS,
        "python": PYTHON_PATTERNS,
        "api": API_PATTERNS,
        "docker": DOCKER_PATTERNS,
        "readme": README_PATTERNS,
    }
    return patterns_map.get(source_type.lower(), {})


def match_pattern(text: str, pattern: re.Pattern, group_names: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Match a pattern against text and return structured results.
    
    Args:
        text: Text to search
        pattern: Compiled regex pattern
        group_names: Optional names for capture groups
        
    Returns:
        List of dictionaries with match information
    """
    matches = []
    for match in pattern.finditer(text):
        groups = match.groups()
        if group_names and len(group_names) == len(groups):
            matches.append(dict(zip(group_names, groups)))
        else:
            matches.append({"match": match.group(0), "groups": groups})
    return matches

