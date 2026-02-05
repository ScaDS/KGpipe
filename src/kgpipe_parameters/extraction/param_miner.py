"""
Parameter mining/extraction from various sources (CLI, Python, HTTP APIs, Docker).

This module provides the main ParameterMiner class for unified parameter extraction.
Individual extractors are implemented in the extractors/ submodule.
"""

import ast
from pathlib import Path
from typing import Optional, Union

from .models import RawParameter, ExtractionResult, SourceType, ExtractionMethod
from .extractors import (
    CLIExtractor,
    PythonLibExtractor,
    HTTPAPIExtractor,
    DockerExtractor,
    LLMCLIExtractor,
    LLMPythonExtractor,
    LLMHTTPExtractor,
    LLMDockerExtractor,
)

# Re-export extractors for backwards compatibility
__all__ = [
    "ParameterMiner",
    "CLIExtractor",
    "PythonLibExtractor",
    "HTTPAPIExtractor",
    "DockerExtractor",
    "LLMCLIExtractor",
    "LLMPythonExtractor",
    "LLMHTTPExtractor",
    "LLMDockerExtractor",
]


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
