"""
Base classes for parameter extractors.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .models import RawParameter, ExtractionResult, SourceType, ExtractionMethod


class BaseExtractor(ABC):
    """Abstract base class for all parameter extractors."""
    
    def __init__(self, source_type: SourceType):
        self.source_type = source_type
    
    @abstractmethod
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """
        Extract parameters from the given source.
        
        Args:
            source: Source content (text, file path, etc.)
            tool_name: Optional name of the tool being analyzed
            
        Returns:
            ExtractionResult containing extracted parameters
        """
        pass


class RegexExtractor(BaseExtractor):
    """Base class for regex-based parameter extraction."""
    
    def __init__(self, source_type: SourceType, patterns: Optional[dict] = None):
        super().__init__(source_type)
        self.patterns = patterns or {}
    
    @abstractmethod
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters using regex patterns."""
        pass
    
    def _apply_patterns(self, text: str) -> List[RawParameter]:
        """
        Apply regex patterns to extract parameters.
        Subclasses should override this with their specific pattern matching logic.
        """
        return []


class LLMExtractor(BaseExtractor):
    """Base class for LLM-based parameter extraction."""
    
    def __init__(self, source_type: SourceType, llm_client=None):
        super().__init__(source_type)
        self.llm_client = llm_client
        if llm_client is None:
            try:
                from kgpipe_llm.common.core import LLMClient, get_client_from_env
                self.llm_client = get_client_from_env()
            except ImportError:
                raise ImportError(
                    "LLM extraction requires kgpipe_llm. "
                    "Install it or provide an LLMClient instance."
                )
    
    @abstractmethod
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters using LLM."""
        pass
    
    def _create_prompt(self, source: str, tool_name: Optional[str] = None) -> str:
        """
        Create a prompt for LLM extraction.
        Subclasses should override this with their specific prompt template.
        """
        return f"Extract configuration parameters from:\n\n{source}"
    
    def _parse_llm_response(self, response: dict) -> List[RawParameter]:
        """
        Parse LLM response into RawParameter objects.
        Subclasses should override this with their specific parsing logic.
        """
        return []

