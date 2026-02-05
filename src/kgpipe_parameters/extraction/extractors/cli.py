"""
CLI parameter extraction from help output.
"""

import re
from typing import Optional, Union

from ..models import RawParameter, ExtractionResult, SourceType, ExtractionMethod
from ..base import RegexExtractor, LLMExtractor
from ..patterns import CLI_PATTERNS
from ..utils import normalize_parameter_name, parse_default_value


class CLIExtractor(RegexExtractor):
    """Extract parameters from CLI help output."""
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        super().__init__(SourceType.CLI, CLI_PATTERNS)
        self.use_llm = use_llm
        if use_llm:
            self.llm_extractor = LLMCLIExtractor(llm_client)
    
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters from CLI help text."""
        if self.use_llm:
            return self.llm_extractor.extract(source, tool_name)
        
        parameters = []
        errors = []
        
        try:
            lines = source.split('\n')
            current_param = None
            
            for line in lines:
                # Skip usage lines (they contain brackets and are not actual parameter descriptions)
                if line.strip().startswith("usage:") or (line.strip().startswith("[") and "]" in line and "optional" not in line.lower() and "arguments" not in line.lower()):
                    continue
                
                # Match long flags: --param or --param=VALUE
                long_match = CLI_PATTERNS["long_flag"].search(line)
                if long_match:
                    param_name = long_match.group(1)
                    # Don't use group(2) from usage line - it's the placeholder, not default
                    default_val = None
                    
                    normalized = normalize_parameter_name(param_name)
                    native_keys = [f"--{param_name}"]
                    
                    # Check for short form on same line (but not -h from usage line)
                    short_match = CLI_PATTERNS["short_flag"].search(line)
                    if short_match and short_match.group(1) != 'h':  # Skip -h help flag
                        native_keys.append(f"-{short_match.group(1)}")
                    
                    # Extract description - skip placeholder if present
                    # Pattern: --param PLACEHOLDER  Description text
                    # We want to skip the PLACEHOLDER (uppercase word) if it exists
                    desc_match = re.search(rf"--{param_name}\s+(?:[A-Z_]+\s+)?(.+)", line)
                    if not desc_match:
                        # Fallback: just get everything after the flag
                        desc_match = re.search(r"--[^\s]+\s+(.+)", line)
                    description = desc_match.group(1).strip() if desc_match else None
                    
                    # Check if required
                    required = CLI_PATTERNS["required"].search(line) is not None
                    
                    # Extract default value from description line (not usage line)
                    default_match = CLI_PATTERNS["default_value"].search(line)
                    if default_match:
                        default_val = default_match.group(1).strip()
                    
                    # Extract type hint
                    type_match = CLI_PATTERNS["type_hint"].search(line)
                    type_hint = type_match.group(1) if type_match else None
                    
                    current_param = RawParameter(
                        name=normalized,
                        native_keys=native_keys,
                        description=description,
                        type_hint=type_hint,
                        default_value=parse_default_value(default_val) if default_val else None,
                        required=required,
                        source=line,
                        provenance={"line": lines.index(line) + 1}
                    )
                    parameters.append(current_param)
                
                # Match short flags: -p
                elif CLI_PATTERNS["short_flag"].search(line) and not long_match:
                    short_match = CLI_PATTERNS["short_flag"].search(line)
                    param_name = short_match.group(1)
                    normalized = normalize_parameter_name(param_name)
                    
                    current_param = RawParameter(
                        name=normalized,
                        native_keys=[f"-{param_name}"],
                        description=None,
                        source=line,
                        provenance={"line": lines.index(line) + 1}
                    )
                    parameters.append(current_param)
                
                # If we have a current param, try to extract description from continuation lines
                elif current_param and line.strip() and not line.strip().startswith('-'):
                    if not current_param.description:
                        current_param.description = line.strip()
                    else:
                        current_param.description += " " + line.strip()
        
        except Exception as e:
            errors.append(f"Error extracting CLI parameters: {str(e)}")
        
        return ExtractionResult(
            tool_name=tool_name or "unknown_cli_tool",
            source_type=SourceType.CLI,
            extraction_method=ExtractionMethod.REGEX,
            parameters=parameters,
            errors=errors
        )


class LLMCLIExtractor(LLMExtractor):
    """LLM-based CLI parameter extraction."""
    
    def __init__(self, llm_client=None):
        super().__init__(SourceType.CLI, llm_client)
    
    def _create_prompt(self, source: str, tool_name: Optional[str] = None) -> str:
        return f"""Extract all configuration parameters from the following CLI help output.
For each parameter, identify:
- Parameter name (normalized, without -- or -)
- Native keys/flags (--flag, -f, etc.)
- Description
- Type (if mentioned)
- Default value (if mentioned)
- Whether it's required or optional

CLI Help Output:
{source}

Return a JSON object with a 'parameters' array. Each parameter should have: name, native_keys, description, type_hint, default_value, required."""
    
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters using LLM."""
        from pydantic import BaseModel
        from typing import List as TypingList
        
        class ParameterSchema(BaseModel):
            name: str
            native_keys: TypingList[str]
            description: Optional[str] = None
            type_hint: Optional[str] = None
            default_value: Optional[Union[str, int, float, bool]] = None
            required: bool = False
        
        class ExtractionSchema(BaseModel):
            parameters: TypingList[ParameterSchema]
        
        try:
            prompt = self._create_prompt(source, tool_name)
            response = self.llm_client.send_prompt(prompt, ExtractionSchema)
            
            parameters = []
            if "parameters" in response:
                for param_data in response["parameters"]:
                    raw_param = RawParameter(
                        name=normalize_parameter_name(param_data["name"]),
                        native_keys=param_data.get("native_keys", []),
                        description=param_data.get("description"),
                        type_hint=param_data.get("type_hint"),
                        default_value=param_data.get("default_value"),
                        required=param_data.get("required", False),
                        source=source[:100],  # First 100 chars
                        provenance={"method": "llm"}
                    )
                    parameters.append(raw_param)
            
            return ExtractionResult(
                tool_name=tool_name or "unknown_cli_tool",
                source_type=SourceType.CLI,
                extraction_method=ExtractionMethod.LLM,
                parameters=parameters
            )
        except Exception as e:
            return ExtractionResult(
                tool_name=tool_name or "unknown_cli_tool",
                source_type=SourceType.CLI,
                extraction_method=ExtractionMethod.LLM,
                parameters=[],
                errors=[f"LLM extraction failed: {str(e)}"]
            )


