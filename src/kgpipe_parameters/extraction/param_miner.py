"""
Parameter mining/extraction from various sources (CLI, Python, HTTP APIs, Docker).
"""

import re
import ast
import json
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from .models import (
    RawParameter, ExtractionResult, SourceType, ExtractionMethod
)
from .base import RegexExtractor, LLMExtractor
from .patterns import get_patterns, CLI_PATTERNS, PYTHON_PATTERNS, DOCKER_PATTERNS
from .utils import normalize_parameter_name, parse_default_value, infer_parameter_type


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


class PythonLibExtractor(RegexExtractor):
    """Extract parameters from Python code (functions, classes, docstrings)."""
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        super().__init__(SourceType.PYTHON_LIB, PYTHON_PATTERNS)
        self.use_llm = use_llm
        if use_llm:
            self.llm_extractor = LLMPythonExtractor(llm_client)
    
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters from Python source code."""
        if self.use_llm:
            return self.llm_extractor.extract(source, tool_name)
        
        parameters = []
        errors = []
        
        try:
            # Try to parse as Python AST
            try:
                tree = ast.parse(source)
                parameters.extend(self._extract_from_ast(tree, source))
            except SyntaxError:
                # If not valid Python, try regex-based extraction
                parameters.extend(self._extract_from_regex(source))
        
        except Exception as e:
            errors.append(f"Error extracting Python parameters: {str(e)}")
        
        return ExtractionResult(
            tool_name=tool_name or "unknown_python_lib",
            source_type=SourceType.PYTHON_LIB,
            extraction_method=ExtractionMethod.REGEX,
            parameters=parameters,
            errors=errors
        )
    
    def _extract_from_ast(self, tree: ast.AST, source: str) -> List[RawParameter]:
        """Extract parameters from Python AST."""
        parameters = []
        
        class ParameterVisitor(ast.NodeVisitor):
            def __init__(self):
                self.params = []
                self.source_lines = source.split('\n')
            
            def visit_FunctionDef(self, node):
                # Extract function parameters
                for arg in node.args.args:
                    if arg.arg == 'self':
                        continue
                    
                    # Get type hint
                    type_hint = None
                    if arg.annotation:
                        type_hint = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                    
                    # Get default value
                    default_val = None
                    default_idx = len(node.args.args) - len(node.args.defaults)
                    if arg in node.args.args[default_idx:]:
                        default_node = node.args.defaults[node.args.args[default_idx:].index(arg)]
                        if hasattr(ast, 'unparse'):
                            default_val = ast.unparse(default_node)
                        else:
                            default_val = ast.literal_eval(default_node) if isinstance(default_node, (ast.Constant, ast.Str, ast.Num)) else None
                    
                    # Extract docstring info
                    description = None
                    if ast.get_docstring(node):
                        docstring = ast.get_docstring(node)
                        # Look for :param arg: description
                        param_pattern = re.compile(rf":param\s+{arg.arg}:\s*(.+?)(?=\n|:param|$)", re.MULTILINE)
                        match = param_pattern.search(docstring)
                        if match:
                            description = match.group(1).strip()
                    
                    param = RawParameter(
                        name=normalize_parameter_name(arg.arg),
                        native_keys=[arg.arg],
                        description=description,
                        type_hint=type_hint,
                        default_value=parse_default_value(default_val) if default_val else None,
                        required=default_val is None,
                        source=f"{node.name}()",
                        provenance={"function": node.name, "line": node.lineno}
                    )
                    self.params.append(param)
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Extract class attributes (for dataclasses, Pydantic models, etc.)
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        # Annotated assignment: name: type = default
                        if isinstance(item.target, ast.Name):
                            attr_name = item.target.id
                            
                            # Get type hint
                            type_hint = None
                            if item.annotation:
                                type_hint = ast.unparse(item.annotation) if hasattr(ast, 'unparse') else str(item.annotation)
                            
                            # Get default value
                            default_val = None
                            if item.value:
                                if hasattr(ast, 'unparse'):
                                    default_val = ast.unparse(item.value)
                                else:
                                    try:
                                        default_val = ast.literal_eval(item.value)
                                    except (ValueError, TypeError):
                                        default_val = None
                            
                            param = RawParameter(
                                name=normalize_parameter_name(attr_name),
                                native_keys=[attr_name],
                                description=None,
                                type_hint=type_hint,
                                default_value=parse_default_value(str(default_val)) if default_val is not None else None,
                                required=default_val is None,
                                source=f"{node.name}.{attr_name}",
                                provenance={"class": node.name, "line": item.lineno if hasattr(item, 'lineno') else node.lineno}
                            )
                            self.params.append(param)
                    elif isinstance(item, ast.Assign):
                        # Regular assignment: name = value (might be in dataclass)
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                attr_name = target.id
                                # Try to get value
                                default_val = None
                                if item.value:
                                    try:
                                        default_val = ast.literal_eval(item.value)
                                    except (ValueError, TypeError):
                                        default_val = None
                                
                                param = RawParameter(
                                    name=normalize_parameter_name(attr_name),
                                    native_keys=[attr_name],
                                    description=None,
                                    type_hint=None,
                                    default_value=parse_default_value(str(default_val)) if default_val is not None else None,
                                    required=False,
                                    source=f"{node.name}.{attr_name}",
                                    provenance={"class": node.name, "line": item.lineno}
                                )
                                self.params.append(param)
                
                self.generic_visit(node)
        
        visitor = ParameterVisitor()
        visitor.visit(tree)
        return visitor.params
    
    def _extract_from_regex(self, source: str) -> List[RawParameter]:
        """Fallback regex-based extraction."""
        parameters = []
        
        # Extract function parameters
        func_pattern = re.compile(r"def\s+\w+\s*\(([^)]+)\)", re.MULTILINE)
        for match in func_pattern.finditer(source):
            params_str = match.group(1)
            for param_match in PYTHON_PATTERNS["function_param"].finditer(params_str):
                param_name = param_match.group(1)
                type_hint = param_match.group(2).strip() if param_match.group(2) else None
                default_val = param_match.group(3).strip() if param_match.group(3) else None
                
                parameters.append(RawParameter(
                    name=normalize_parameter_name(param_name),
                    native_keys=[param_name],
                    type_hint=type_hint,
                    default_value=parse_default_value(default_val) if default_val else None,
                    required=default_val is None,
                    source=match.group(0),
                    provenance={"method": "regex"}
                ))
        
        return parameters


class LLMPythonExtractor(LLMExtractor):
    """LLM-based Python parameter extraction."""
    
    def __init__(self, llm_client=None):
        super().__init__(SourceType.PYTHON_LIB, llm_client)
    
    def _create_prompt(self, source: str, tool_name: Optional[str] = None) -> str:
        return f"""Extract all configuration parameters from the following Python code.
Look for:
- Function parameters with type hints and defaults
- Class attributes with type annotations
- Configuration classes (dataclasses, Pydantic models)
- Environment variables

Python Code:
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
                        source=source[:200],  # First 200 chars
                        provenance={"method": "llm"}
                    )
                    parameters.append(raw_param)
            
            return ExtractionResult(
                tool_name=tool_name or "unknown_python_lib",
                source_type=SourceType.PYTHON_LIB,
                extraction_method=ExtractionMethod.LLM,
                parameters=parameters
            )
        except Exception as e:
            return ExtractionResult(
                tool_name=tool_name or "unknown_python_lib",
                source_type=SourceType.PYTHON_LIB,
                extraction_method=ExtractionMethod.LLM,
                parameters=[],
                errors=[f"LLM extraction failed: {str(e)}"]
            )


class HTTPAPIExtractor(RegexExtractor):
    """Extract parameters from HTTP API documentation (OpenAPI, Swagger, etc.)."""
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        super().__init__(SourceType.HTTP_API, {})
        self.use_llm = use_llm
        if use_llm:
            self.llm_extractor = LLMHTTPExtractor(llm_client)
    
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters from API documentation."""
        if self.use_llm:
            return self.llm_extractor.extract(source, tool_name)
        
        parameters = []
        errors = []
        
        try:
            # Try to parse as OpenAPI/Swagger spec
            spec = None
            try:
                # Try JSON first
                if source.strip().startswith('{'):
                    spec = json.loads(source)
                else:
                    # Try YAML
                    spec = yaml.safe_load(source)
                
                # Check if it looks like OpenAPI/Swagger spec
                if spec and isinstance(spec, dict) and ("openapi" in spec or "swagger" in spec or "paths" in spec):
                    parameters.extend(self._extract_from_openapi(spec))
                else:
                    # Not a valid spec, try regex-based extraction
                    parameters.extend(self._extract_from_docs(source))
            except (json.JSONDecodeError, yaml.YAMLError):
                # If parsing fails, try regex-based extraction
                parameters.extend(self._extract_from_docs(source))
        
        except Exception as e:
            errors.append(f"Error extracting API parameters: {str(e)}")
        
        return ExtractionResult(
            tool_name=tool_name or "unknown_api",
            source_type=SourceType.HTTP_API,
            extraction_method=ExtractionMethod.REGEX,
            parameters=parameters,
            errors=errors
        )
    
    def _extract_from_openapi(self, spec: dict) -> List[RawParameter]:
        """Extract parameters from OpenAPI specification."""
        parameters = []
        
        # Extract from paths
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, operation in methods.items():
                # Path parameters
                for param in operation.get("parameters", []):
                    param_name = param.get("name", "")
                    param_schema = param.get("schema", {})
                    
                    raw_param = RawParameter(
                        name=normalize_parameter_name(param_name),
                        native_keys=[param_name],
                        description=param.get("description"),
                        type_hint=param_schema.get("type"),
                        default_value=param_schema.get("default"),
                        required=param.get("required", False),
                        source=f"{method.upper()} {path}",
                        provenance={"location": "path", "method": method}
                    )
                    parameters.append(raw_param)
                
                # Request body parameters
                request_body = operation.get("requestBody", {})
                content = request_body.get("content", {})
                for content_type, schema_obj in content.items():
                    schema = schema_obj.get("schema", {})
                    if "properties" in schema:
                        for prop_name, prop_schema in schema["properties"].items():
                            raw_param = RawParameter(
                                name=normalize_parameter_name(prop_name),
                                native_keys=[prop_name],
                                description=prop_schema.get("description"),
                                type_hint=prop_schema.get("type"),
                                default_value=prop_schema.get("default"),
                                required=prop_name in schema.get("required", []),
                                source=f"{method.upper()} {path} (body)",
                                provenance={"location": "body", "method": method}
                            )
                            parameters.append(raw_param)
        
        return parameters
    
    def _extract_from_docs(self, source: str) -> List[RawParameter]:
        """Extract parameters from unstructured API documentation."""
        parameters = []
        # Basic regex extraction for common patterns
        # This is a simplified version - LLM would be better for complex docs
        return parameters


class LLMHTTPExtractor(LLMExtractor):
    """LLM-based HTTP API parameter extraction."""
    
    def __init__(self, llm_client=None):
        super().__init__(SourceType.HTTP_API, llm_client)
    
    def _create_prompt(self, source: str, tool_name: Optional[str] = None) -> str:
        return f"""Extract all API parameters from the following API documentation or specification.
Look for:
- Query parameters
- Path parameters
- Request body parameters
- Header parameters

API Documentation:
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
                        source=source[:200],
                        provenance={"method": "llm"}
                    )
                    parameters.append(raw_param)
            
            return ExtractionResult(
                tool_name=tool_name or "unknown_api",
                source_type=SourceType.HTTP_API,
                extraction_method=ExtractionMethod.LLM,
                parameters=parameters
            )
        except Exception as e:
            return ExtractionResult(
                tool_name=tool_name or "unknown_api",
                source_type=SourceType.HTTP_API,
                extraction_method=ExtractionMethod.LLM,
                parameters=[],
                errors=[f"LLM extraction failed: {str(e)}"]
            )


class DockerExtractor(RegexExtractor):
    """Extract parameters from Docker configurations (Dockerfile, docker-compose.yml)."""
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        super().__init__(SourceType.DOCKER, DOCKER_PATTERNS)
        self.use_llm = use_llm
        if use_llm:
            self.llm_extractor = LLMDockerExtractor(llm_client)
    
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters from Docker configuration."""
        if self.use_llm:
            return self.llm_extractor.extract(source, tool_name)
        
        parameters = []
        errors = []
        
        try:
            # Check if it's a Dockerfile or docker-compose.yml
            if "FROM" in source or "RUN" in source:
                # Dockerfile
                parameters.extend(self._extract_from_dockerfile(source))
            elif "version:" in source or "services:" in source:
                # docker-compose.yml
                try:
                    compose = yaml.safe_load(source)
                    parameters.extend(self._extract_from_compose(compose))
                except yaml.YAMLError:
                    parameters.extend(self._extract_from_dockerfile(source))
            else:
                parameters.extend(self._extract_from_dockerfile(source))
        
        except Exception as e:
            errors.append(f"Error extracting Docker parameters: {str(e)}")
        
        return ExtractionResult(
            tool_name=tool_name or "unknown_docker",
            source_type=SourceType.DOCKER,
            extraction_method=ExtractionMethod.REGEX,
            parameters=parameters,
            errors=errors
        )
    
    def _extract_from_dockerfile(self, source: str) -> List[RawParameter]:
        """Extract ENV and ARG declarations from Dockerfile."""
        parameters = []
        lines = source.split('\n')
        
        for line in lines:
            # ENV declarations
            env_match = DOCKER_PATTERNS["env_declaration"].search(line)
            if env_match:
                var_name = env_match.group(1)
                var_value = env_match.group(2) if env_match.group(2) else None
                
                parameters.append(RawParameter(
                    name=normalize_parameter_name(var_name),
                    native_keys=[var_name],
                    description=f"Environment variable: {var_name}",
                    default_value=parse_default_value(var_value) if var_value else None,
                    required=False,
                    source=line,
                    provenance={"type": "ENV", "line": lines.index(line) + 1}
                ))
            
            # ARG declarations
            arg_match = DOCKER_PATTERNS["arg_declaration"].search(line)
            if arg_match:
                var_name = arg_match.group(1)
                var_value = arg_match.group(2) if arg_match.group(2) else None
                
                parameters.append(RawParameter(
                    name=normalize_parameter_name(var_name),
                    native_keys=[var_name],
                    description=f"Build argument: {var_name}",
                    default_value=parse_default_value(var_value) if var_value else None,
                    required=False,
                    source=line,
                    provenance={"type": "ARG", "line": lines.index(line) + 1}
                ))
        
        return parameters
    
    def _extract_from_compose(self, compose: dict) -> List[RawParameter]:
        """Extract environment variables from docker-compose.yml."""
        parameters = []
        
        services = compose.get("services", {})
        for service_name, service_config in services.items():
            env = service_config.get("environment", {})
            if isinstance(env, dict):
                for var_name, var_value in env.items():
                    parameters.append(RawParameter(
                        name=normalize_parameter_name(var_name),
                        native_keys=[var_name],
                        description=f"Environment variable for service {service_name}",
                        default_value=parse_default_value(str(var_value)) if var_value else None,
                        required=False,
                        source=f"services.{service_name}.environment",
                        provenance={"service": service_name, "type": "environment"}
                    ))
        
        return parameters


class LLMDockerExtractor(LLMExtractor):
    """LLM-based Docker parameter extraction."""
    
    def __init__(self, llm_client=None):
        super().__init__(SourceType.DOCKER, llm_client)
    
    def _create_prompt(self, source: str, tool_name: Optional[str] = None) -> str:
        return f"""Extract all configuration parameters from the following Docker configuration.
Look for:
- ENV variables
- ARG build arguments
- Environment variables in docker-compose.yml
- Volume mounts and port mappings that could be parameterized

Docker Configuration:
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
                        source=source[:200],
                        provenance={"method": "llm"}
                    )
                    parameters.append(raw_param)
            
            return ExtractionResult(
                tool_name=tool_name or "unknown_docker",
                source_type=SourceType.DOCKER,
                extraction_method=ExtractionMethod.LLM,
                parameters=parameters
            )
        except Exception as e:
            return ExtractionResult(
                tool_name=tool_name or "unknown_docker",
                source_type=SourceType.DOCKER,
                extraction_method=ExtractionMethod.LLM,
                parameters=[],
                errors=[f"LLM extraction failed: {str(e)}"]
            )


class ParameterMiner:
    """
    Main class for parameter extraction from various sources.
    Provides unified interface for extracting configuration parameters.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize ParameterMiner.
        
        Args:
            llm_client: Optional LLMClient instance for LLM-based extraction
        """
        self.llm_client = llm_client
        self.extractors = {
            SourceType.CLI: CLIExtractor(use_llm=False),
            SourceType.PYTHON_LIB: PythonLibExtractor(use_llm=False),
            SourceType.HTTP_API: HTTPAPIExtractor(use_llm=False),
            SourceType.DOCKER: DockerExtractor(use_llm=False),
        }
    
    def extract_parameters(
        self,
        source: Union[str, Path],
        source_type: Optional[SourceType] = None,
        method: ExtractionMethod = ExtractionMethod.AUTO,
        tool_name: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract parameters from a source.
        
        Args:
            source: Source content (text, file path, etc.)
            source_type: Type of source (auto-detected if None)
            method: Extraction method ('regex', 'llm', or 'auto')
            tool_name: Optional name of the tool being analyzed
            
        Returns:
            ExtractionResult containing extracted parameters
        """
        # Read file if Path provided
        if isinstance(source, Path):
            source_path = source
            source = source_path.read_text()
            if not tool_name:
                tool_name = source_path.stem
        elif isinstance(source, str):
            # Only treat as file path if it's a short string without newlines
            # and actually exists as a file
            if len(source) < 260 and '\n' not in source and Path(source).exists():
                source_path = Path(source)
                source = source_path.read_text()
                if not tool_name:
                    tool_name = source_path.stem
        
        # Auto-detect source type if not provided
        if source_type is None:
            source_type = self._detect_source_type(source)
        
        # Select extraction method
        if method == ExtractionMethod.AUTO:
            # Try regex first, fallback to LLM if available
            try:
                if source_type == SourceType.UNKNOWN:
                    # For unknown source types, try CLI extractor as fallback
                    extractor = self.extractors[SourceType.CLI]
                else:
                    extractor = self.extractors[source_type]
                result = extractor.extract(source, tool_name)
                # If regex extraction found few/no parameters and LLM is available, try LLM
                if len(result.parameters) == 0 and self.llm_client:
                    method = ExtractionMethod.LLM
                else:
                    return result
            except (KeyError, Exception):
                if self.llm_client:
                    method = ExtractionMethod.LLM
                else:
                    # Return empty result for unknown types
                    return ExtractionResult(
                        tool_name=tool_name or "unknown",
                        source_type=source_type,
                        extraction_method=ExtractionMethod.REGEX,
                        parameters=[],
                        errors=[f"No extractor available for source type: {source_type}"]
                    )
        
        # Use LLM if requested or as fallback
        if method == ExtractionMethod.LLM:
            if not self.llm_client:
                raise ValueError("LLM extraction requires an LLMClient instance")
            
            # Create LLM extractor for the source type
            llm_extractors = {
                SourceType.CLI: LLMCLIExtractor(self.llm_client),
                SourceType.PYTHON_LIB: LLMPythonExtractor(self.llm_client),
                SourceType.HTTP_API: LLMHTTPExtractor(self.llm_client),
                SourceType.DOCKER: LLMDockerExtractor(self.llm_client),
            }
            extractor = llm_extractors.get(source_type)
            if extractor:
                return extractor.extract(source, tool_name)
        
        # Use regex extractor
        extractor = self.extractors[source_type]
        return extractor.extract(source, tool_name)
    
    def _detect_source_type(self, source: str) -> SourceType:
        """Auto-detect source type from content."""
        source_lower = source.lower()
        
        # Check for CLI help patterns
        if any(x in source_lower for x in ["usage:", "options:", "--help", "arguments:"]):
            return SourceType.CLI
        
        # Check for Python code
        if any(x in source for x in ["def ", "class ", "import ", "@"]):
            try:
                ast.parse(source)
                return SourceType.PYTHON_LIB
            except SyntaxError:
                pass
        
        # Check for OpenAPI/Swagger
        if any(x in source for x in ['"openapi"', '"swagger"', "paths:", "components:"]):
            return SourceType.HTTP_API
        
        # Check for Docker
        if any(x in source for x in ["FROM ", "ENV ", "ARG ", "docker-compose", "services:"]):
            return SourceType.DOCKER
        
        return SourceType.UNKNOWN
    
    def to_parameter_model(self, raw_param: RawParameter):
        """
        Convert RawParameter to Parameter model.
        
        Args:
            raw_param: RawParameter instance
            
        Returns:
            Parameter model instance
        """
        from .utils import to_parameter_model
        return to_parameter_model(raw_param)
    
    def to_json(self, result: ExtractionResult) -> str:
        """
        Convert ExtractionResult to JSON string.
        
        Args:
            result: ExtractionResult instance
            
        Returns:
            JSON string representation
        """
        return result.model_dump_json(indent=2)

