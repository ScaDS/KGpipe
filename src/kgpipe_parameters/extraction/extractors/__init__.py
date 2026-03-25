"""
Extractor implementations for different source types.
"""

from .cli import CLIExtractor, LLMCLIExtractor
from .python_lib import PythonLibExtractor, LLMPythonExtractor
from .http_api import HTTPAPIExtractor, LLMHTTPExtractor
from .docker import DockerExtractor, LLMDockerExtractor
from .readme_doc import ReadmeDocExtractor, LLMReadmeDocExtractor

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
    # README / documentation
    "ReadmeDocExtractor",
    "LLMReadmeDocExtractor",
]


