"""
Parameter extraction module for mining configuration parameters from various sources.
"""

from .param_miner import ParameterMiner
from .extractors import (
    CLIExtractor,
    PythonLibExtractor,
    HTTPAPIExtractor,
    DockerExtractor,
    ReadmeDocExtractor,
    LLMCLIExtractor,
    LLMPythonExtractor,
    LLMHTTPExtractor,
    LLMDockerExtractor,
    LLMReadmeDocExtractor,
)
from .models import (
    RawParameter,
    ExtractionResult,
    SourceType,
    ExtractionMethod,
)
from .base import (
    BaseExtractor,
    RegexExtractor,
    LLMExtractor,
)
from .utils import (
    to_parameter_model,
    normalize_parameter_name,
    parse_default_value,
    infer_parameter_type,
    extract_constraints,
)
from .chunk_filter import (
    score_chunk,
    has_parameter_signals,
    KEYWORD_SETS,
)

__all__ = [
    # Main class
    "ParameterMiner",
    # Extractors
    "CLIExtractor",
    "PythonLibExtractor",
    "HTTPAPIExtractor",
    "DockerExtractor",
    "ReadmeDocExtractor",
    "LLMCLIExtractor",
    "LLMPythonExtractor",
    "LLMHTTPExtractor",
    "LLMDockerExtractor",
    "LLMReadmeDocExtractor",
    # Base classes
    "BaseExtractor",
    "RegexExtractor",
    "LLMExtractor",
    # Models
    "RawParameter",
    "ExtractionResult",
    "SourceType",
    "ExtractionMethod",
    # Utilities
    "to_parameter_model",
    "normalize_parameter_name",
    "parse_default_value",
    "infer_parameter_type",
    "extract_constraints",
    # Chunk filtering
    "score_chunk",
    "has_parameter_signals",
    "KEYWORD_SETS",
]
