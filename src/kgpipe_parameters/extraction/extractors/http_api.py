"""
HTTP API parameter extraction from OpenAPI/Swagger specs and documentation.
"""

import json
import yaml
from typing import List, Optional, Union

from ..models import RawParameter, ExtractionResult, SourceType, ExtractionMethod
from ..base import RegexExtractor, LLMExtractor
from ..utils import normalize_parameter_name


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


