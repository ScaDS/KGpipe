"""
Extractor implementations for different source types.
"""

from .cli import CLIExtractor, LLMCLIExtractor
from .python_lib import PythonLibExtractor, LLMPythonExtractor
from .http_api import HTTPAPIExtractor, LLMHTTPExtractor
from .docker import DockerExtractor, LLMDockerExtractor

__all__ = [
    # CLI
    "CLIExtractor",
    "LLMCLIExtractor",
    # Python
    "PythonLibExtractor",
    "LLMPythonExtractor",
    # HTTP API
    "HTTPAPIExtractor",
    "LLMHTTPExtractor",
    # Docker
    "DockerExtractor",
    "LLMDockerExtractor",
]


