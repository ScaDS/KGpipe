"""
Docker parameter extraction from Dockerfile and docker-compose.yml.
"""

import yaml
from typing import List, Optional, Union

from ..models import RawParameter, ExtractionResult, SourceType, ExtractionMethod
from ..base import RegexExtractor, LLMExtractor
from ..patterns import DOCKER_PATTERNS
from ..utils import normalize_parameter_name, parse_default_value


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


