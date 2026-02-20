"""
Pytest fixtures for parameter extraction tests.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any


def get_test_data_path(relative_path: str) -> Path:
    """Get path to test data file."""
    test_dir = Path(__file__).parent
    path = test_dir / "test_data" / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Test data path {path} does not exist")
    return path


@pytest.fixture
def test_data_dir():
    """Fixture for test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def cli_help_argparse():
    """Fixture for argparse CLI help text."""
    path = get_test_data_path("cli/argparse_help.txt")
    return path.read_text()


@pytest.fixture
def cli_help_click():
    """Fixture for click CLI help text."""
    path = get_test_data_path("cli/click_help.txt")
    return path.read_text()


@pytest.fixture
def cli_help_simple():
    """Fixture for simple CLI help text."""
    path = get_test_data_path("cli/simple_help.txt")
    return path.read_text()


@pytest.fixture
def python_function_code():
    """Fixture for Python function code."""
    path = get_test_data_path("python/function_with_params.py")
    return path.read_text()


@pytest.fixture
def python_dataclass_code():
    """Fixture for Python dataclass code."""
    path = get_test_data_path("python/dataclass_config.py")
    return path.read_text()


@pytest.fixture
def python_pydantic_code():
    """Fixture for Python Pydantic model code."""
    path = get_test_data_path("python/pydantic_model.py")
    return path.read_text()


@pytest.fixture
def openapi_spec():
    """Fixture for OpenAPI specification."""
    path = get_test_data_path("api/openapi_spec.yaml")
    return path.read_text()


@pytest.fixture
def swagger_spec():
    """Fixture for Swagger specification."""
    path = get_test_data_path("api/swagger_spec.json")
    return path.read_text()


@pytest.fixture
def dockerfile_content():
    """Fixture for Dockerfile content."""
    path = get_test_data_path("docker/Dockerfile")
    return path.read_text()


@pytest.fixture
def docker_compose_content():
    """Fixture for docker-compose.yml content."""
    path = get_test_data_path("docker/docker-compose.yml")
    return path.read_text()


@pytest.fixture
def readme_tool_doc():
    """Fixture for a tool README with configuration parameters."""
    path = get_test_data_path("readme/tool_readme.md")
    return path.read_text()


@pytest.fixture
def readme_minimal():
    """Fixture for a minimal README."""
    path = get_test_data_path("readme/minimal_readme.md")
    return path.read_text()


@pytest.fixture
def mock_llm_client():
    """Fixture for mocked LLM client."""
    mock_client = Mock()
    
    # Mock response structure
    mock_response = {
        "parameters": [
            {
                "name": "threshold",
                "native_keys": ["--threshold", "-t"],
                "description": "Matching threshold",
                "type_hint": "float",
                "default_value": 0.5,
                "required": False
            }
        ]
    }
    
    mock_client.send_prompt = Mock(return_value=mock_response)
    return mock_client

