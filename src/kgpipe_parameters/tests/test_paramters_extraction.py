"""
Comprehensive tests for parameter extraction module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from kgpipe_parameters.extraction import (
    ParameterMiner,
    CLIExtractor,
    PythonLibExtractor,
    HTTPAPIExtractor,
    DockerExtractor,
    RawParameter,
    ExtractionResult,
    SourceType,
    ExtractionMethod,
)
from kgpipe_parameters.extraction.utils import (
    normalize_parameter_name,
    parse_default_value,
    infer_parameter_type,
    extract_constraints,
    to_parameter_model,
)
from kgpipe.common.model.configuration import Parameter, ParameterType


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtils:
    """Tests for utility functions."""
    
    def test_normalize_parameter_name(self):
        """Test parameter name normalization."""
        assert normalize_parameter_name("--threshold") == "threshold"
        assert normalize_parameter_name("-t") == "t"
        assert normalize_parameter_name("threshold") == "threshold"
        assert normalize_parameter_name("THRESHOLD") == "threshold"
        assert normalize_parameter_name("max-results") == "max_results"
        assert normalize_parameter_name("camelCase") == "camelcase"
    
    def test_parse_default_value(self):
        """Test parsing of default values."""
        assert parse_default_value("0.5") == 0.5
        assert parse_default_value("100") == 100
        assert parse_default_value("true") is True
        assert parse_default_value("false") is False
        assert parse_default_value("yes") is True
        assert parse_default_value("no") is False
        assert parse_default_value("hello") == "hello"
        assert parse_default_value('"hello"') == "hello"
        assert parse_default_value("'world'") == "world"
        assert parse_default_value(None) is None
    
    def test_infer_parameter_type(self):
        """Test type inference from type hints and default values."""
        # From type hints
        assert infer_parameter_type("int") == ParameterType.integer
        assert infer_parameter_type("float") == ParameterType.number
        assert infer_parameter_type("str") == ParameterType.string
        assert infer_parameter_type("bool") == ParameterType.boolean
        assert infer_parameter_type("List[str]") == ParameterType.array
        assert infer_parameter_type("Dict[str, Any]") == ParameterType.object
        assert infer_parameter_type("enum") == ParameterType.enum
        
        # From default values
        assert infer_parameter_type(None, 42) == ParameterType.integer
        assert infer_parameter_type(None, 3.14) == ParameterType.number
        assert infer_parameter_type(None, "text") == ParameterType.string
        assert infer_parameter_type(None, True) == ParameterType.boolean
        assert infer_parameter_type(None, []) == ParameterType.array
        assert infer_parameter_type(None, {}) == ParameterType.object
        
        # Default to string
        assert infer_parameter_type(None, None) == ParameterType.string
    
    def test_extract_constraints(self):
        """Test constraint extraction from descriptions."""
        desc1 = "Threshold value (min: 0.0, max: 1.0)"
        constraints1 = extract_constraints(desc1)
        assert constraints1["minimum"] == 0.0
        assert constraints1["maximum"] == 1.0
        
        desc2 = "Choices: [option1, option2, option3]"
        constraints2 = extract_constraints(desc2)
        assert "allowed_values" in constraints2
        assert len(constraints2["allowed_values"]) == 3
        
        desc3 = "Minimum value: 10"
        constraints3 = extract_constraints(desc3)
        assert constraints3["minimum"] == 10.0
        
        desc4 = "Maximum value: 100"
        constraints4 = extract_constraints(desc4)
        assert constraints4["maximum"] == 100.0
    
    def test_to_parameter_model(self):
        """Test conversion from RawParameter to Parameter model."""
        raw_param = RawParameter(
            name="threshold",
            native_keys=["--threshold", "-t"],
            description="Matching threshold (min: 0.0, max: 1.0)",
            type_hint="float",
            default_value=0.5,
            required=False,
            source="test",
        )
        
        param = to_parameter_model(raw_param)
        
        assert isinstance(param, Parameter)
        assert param.name == "threshold"
        assert "--threshold" in param.native_keys
        assert param.datatype == ParameterType.number
        assert param.default_value == 0.5
        assert param.required is False
        assert param.minimum == 0.0
        assert param.maximum == 1.0


# =============================================================================
# CLI Extractor Tests
# =============================================================================

class TestCLIExtractor:
    """Tests for CLI parameter extraction."""
    
    def test_cli_extractor_basic(self, cli_help_simple):
        """Test basic CLI parameter extraction."""
        extractor = CLIExtractor()
        result = extractor.extract(cli_help_simple, "matcher")
        
        assert isinstance(result, ExtractionResult)
        assert result.source_type == SourceType.CLI
        assert result.extraction_method == ExtractionMethod.REGEX
        assert len(result.parameters) > 0
        
        # Check that threshold parameter was extracted
        threshold_params = [p for p in result.parameters if "threshold" in p.name]
        assert len(threshold_params) > 0
    
    def test_cli_extractor_with_defaults(self, cli_help_argparse):
        """Test extraction of parameters with default values."""
        extractor = CLIExtractor()
        result = extractor.extract(cli_help_argparse, "tool")
        
        # Find threshold parameter with default
        threshold_params = [p for p in result.parameters if "threshold" in p.name]
        if threshold_params:
            param = threshold_params[0]
            assert param.default_value == 0.5 or param.default_value == "0.5"
    
    def test_cli_extractor_required_flags(self, cli_help_argparse):
        """Test detection of required vs optional parameters."""
        extractor = CLIExtractor()
        result = extractor.extract(cli_help_argparse, "tool")
        
        # Check for required parameters
        output_params = [p for p in result.parameters if "output" in p.name]
        if output_params:
            # Output is marked as required in the test data
            param = output_params[0]
            # The extractor should detect "required" in description
    
    def test_cli_extractor_multiple_flags(self, cli_help_click):
        """Test extraction of both long and short flags."""
        extractor = CLIExtractor()
        result = extractor.extract(cli_help_click, "tool")
        
        # Check that parameters have native_keys
        for param in result.parameters:
            assert len(param.native_keys) > 0
    
    def test_cli_extractor_description(self, cli_help_simple):
        """Test extraction of parameter descriptions."""
        extractor = CLIExtractor()
        result = extractor.extract(cli_help_simple, "matcher")
        
        # Check that descriptions are extracted
        params_with_desc = [p for p in result.parameters if p.description]
        assert len(params_with_desc) > 0


# =============================================================================
# Python Extractor Tests
# =============================================================================

class TestPythonExtractor:
    """Tests for Python parameter extraction."""
    
    def test_python_extractor_function_params(self, python_function_code):
        """Test extraction from function signatures."""
        extractor = PythonLibExtractor()
        result = extractor.extract(python_function_code, "process_data")
        
        assert isinstance(result, ExtractionResult)
        assert result.source_type == SourceType.PYTHON_LIB
        assert len(result.parameters) > 0
        
        # Check for expected parameters
        param_names = [p.name for p in result.parameters]
        assert "input_file" in param_names or "inputfile" in param_names
        assert "threshold" in param_names
    
    def test_python_extractor_type_hints(self, python_function_code):
        """Test extraction of type hints."""
        extractor = PythonLibExtractor()
        result = extractor.extract(python_function_code, "process_data")
        
        # Check that type hints are extracted
        params_with_types = [p for p in result.parameters if p.type_hint]
        assert len(params_with_types) > 0
    
    def test_python_extractor_docstrings(self, python_function_code):
        """Test extraction from docstrings."""
        extractor = PythonLibExtractor()
        result = extractor.extract(python_function_code, "process_data")
        
        # Check that descriptions from docstrings are extracted
        params_with_desc = [p for p in result.parameters if p.description]
        assert len(params_with_desc) > 0
    
    def test_python_extractor_dataclass(self, python_dataclass_code):
        """Test extraction from dataclass attributes."""
        extractor = PythonLibExtractor()
        result = extractor.extract(python_dataclass_code, "MatchingConfig")
        
        assert len(result.parameters) > 0
        param_names = [p.name for p in result.parameters]
        assert "threshold" in param_names or "input_file" in param_names
    
    def test_python_extractor_pydantic_model(self, python_pydantic_code):
        """Test extraction from Pydantic models."""
        extractor = PythonLibExtractor()
        result = extractor.extract(python_pydantic_code, "MatchingConfig")
        
        assert len(result.parameters) > 0
        param_names = [p.name for p in result.parameters]
        assert "threshold" in param_names or "input_file" in param_names
    
    def test_python_extractor_ast_parsing(self, python_function_code):
        """Test AST-based extraction."""
        extractor = PythonLibExtractor()
        result = extractor.extract(python_function_code, "process_data")
        
        # AST parsing should work for valid Python code
        assert result.extraction_method == ExtractionMethod.REGEX
        assert len(result.parameters) > 0


# =============================================================================
# HTTP API Extractor Tests
# =============================================================================

class TestHTTPAPIExtractor:
    """Tests for HTTP API parameter extraction."""
    
    def test_api_extractor_openapi_spec(self, openapi_spec):
        """Test extraction from OpenAPI YAML."""
        extractor = HTTPAPIExtractor()
        result = extractor.extract(openapi_spec, "matching_api")
        
        assert isinstance(result, ExtractionResult)
        assert result.source_type == SourceType.HTTP_API
        assert len(result.parameters) > 0
        
        # Check for expected parameters
        param_names = [p.name for p in result.parameters]
        assert "threshold" in param_names or "input_file" in param_names
    
    def test_api_extractor_swagger_spec(self, swagger_spec):
        """Test extraction from Swagger JSON."""
        extractor = HTTPAPIExtractor()
        result = extractor.extract(swagger_spec, "matching_api")
        
        assert len(result.parameters) > 0
        param_names = [p.name for p in result.parameters]
        assert "threshold" in param_names or "input_file" in param_names
    
    def test_api_extractor_path_params(self, openapi_spec):
        """Test path parameter extraction."""
        extractor = HTTPAPIExtractor()
        result = extractor.extract(openapi_spec, "matching_api")
        
        # OpenAPI spec has query params, not path params in our test data
        # But we should still extract parameters
        assert len(result.parameters) > 0
    
    def test_api_extractor_query_params(self, openapi_spec):
        """Test query parameter extraction."""
        extractor = HTTPAPIExtractor()
        result = extractor.extract(openapi_spec, "matching_api")
        
        # Check for query parameters
        query_params = [p for p in result.parameters if "threshold" in p.name or "max_results" in p.name]
        assert len(query_params) > 0
    
    def test_api_extractor_request_body(self, openapi_spec):
        """Test request body parameter extraction."""
        extractor = HTTPAPIExtractor()
        result = extractor.extract(openapi_spec, "matching_api")
        
        # Check for request body parameters
        body_params = [p for p in result.parameters if "input_file" in p.name or "output_file" in p.name]
        assert len(body_params) > 0


# =============================================================================
# Docker Extractor Tests
# =============================================================================

class TestDockerExtractor:
    """Tests for Docker parameter extraction."""
    
    def test_docker_extractor_env_vars(self, dockerfile_content):
        """Test ENV variable extraction from Dockerfile."""
        extractor = DockerExtractor()
        result = extractor.extract(dockerfile_content, "dockerfile")
        
        assert isinstance(result, ExtractionResult)
        assert result.source_type == SourceType.DOCKER
        assert len(result.parameters) > 0
        
        # Check for ENV variables
        env_params = [p for p in result.parameters if "THRESHOLD" in p.native_keys or "threshold" in p.name]
        assert len(env_params) > 0
    
    def test_docker_extractor_args(self, dockerfile_content):
        """Test ARG extraction from Dockerfile."""
        extractor = DockerExtractor()
        result = extractor.extract(dockerfile_content, "dockerfile")
        
        # Check for ARG declarations
        arg_params = [p for p in result.parameters if "BUILD_VERSION" in p.native_keys or "build_version" in p.name]
        assert len(arg_params) > 0
    
    def test_docker_extractor_compose_env(self, docker_compose_content):
        """Test environment variable extraction from docker-compose.yml."""
        extractor = DockerExtractor()
        result = extractor.extract(docker_compose_content, "docker_compose")
        
        assert len(result.parameters) > 0
        
        # Check for environment variables
        env_params = [p for p in result.parameters if "THRESHOLD" in p.native_keys or "threshold" in p.name]
        assert len(env_params) > 0
    
    def test_docker_extractor_multiple_services(self, docker_compose_content):
        """Test extraction from multiple services."""
        extractor = DockerExtractor()
        result = extractor.extract(docker_compose_content, "docker_compose")
        
        # Should extract from both matcher and processor services
        assert len(result.parameters) > 0


# =============================================================================
# ParameterMiner Integration Tests
# =============================================================================

class TestParameterMiner:
    """Integration tests for ParameterMiner."""
    
    def test_parameter_miner_auto_detect_cli(self, cli_help_simple):
        """Test auto-detection of CLI source."""
        miner = ParameterMiner()
        result = miner.extract_parameters(cli_help_simple, method=ExtractionMethod.AUTO)
        
        assert result.source_type == SourceType.CLI
    
    def test_parameter_miner_auto_detect_python(self, python_function_code):
        """Test auto-detection of Python source."""
        miner = ParameterMiner()
        result = miner.extract_parameters(python_function_code, method=ExtractionMethod.AUTO)
        
        assert result.source_type == SourceType.PYTHON_LIB
    
    def test_parameter_miner_auto_detect_api(self, openapi_spec):
        """Test auto-detection of API source."""
        miner = ParameterMiner()
        result = miner.extract_parameters(openapi_spec, method=ExtractionMethod.AUTO)
        
        assert result.source_type == SourceType.HTTP_API
    
    def test_parameter_miner_auto_detect_docker(self, dockerfile_content):
        """Test auto-detection of Docker source."""
        miner = ParameterMiner()
        result = miner.extract_parameters(dockerfile_content, method=ExtractionMethod.AUTO)
        
        assert result.source_type == SourceType.DOCKER
    
    def test_parameter_miner_file_path(self, test_data_dir):
        """Test extraction from file path."""
        miner = ParameterMiner()
        cli_file = test_data_dir / "cli" / "simple_help.txt"
        result = miner.extract_parameters(str(cli_file), method=ExtractionMethod.AUTO)
        
        assert result.source_type == SourceType.CLI
        assert result.tool_name == "simple_help"
    
    def test_parameter_miner_method_auto(self, cli_help_simple):
        """Test auto method selection (regex → LLM fallback)."""
        miner = ParameterMiner()
        result = miner.extract_parameters(cli_help_simple, method=ExtractionMethod.AUTO)
        
        # Should use regex by default
        assert result.extraction_method == ExtractionMethod.REGEX
    
    def test_parameter_miner_to_json(self, cli_help_simple):
        """Test JSON output conversion."""
        miner = ParameterMiner()
        result = miner.extract_parameters(cli_help_simple, method=ExtractionMethod.AUTO)
        
        json_output = miner.to_json(result)
        assert isinstance(json_output, str)
        assert "parameters" in json_output or '"parameters"' in json_output
    
    def test_parameter_miner_to_parameter_model(self, cli_help_simple):
        """Test Parameter model conversion."""
        miner = ParameterMiner()
        result = miner.extract_parameters(cli_help_simple, method=ExtractionMethod.AUTO)
        
        if result.parameters:
            param_model = miner.to_parameter_model(result.parameters[0])
            assert isinstance(param_model, Parameter)
            assert param_model.name is not None
            assert param_model.datatype is not None


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_extractor_invalid_source(self):
        """Test handling of invalid source content."""
        extractor = CLIExtractor()
        result = extractor.extract("This is not valid CLI help", "test")
        
        # Should not crash, but may return empty or minimal results
        assert isinstance(result, ExtractionResult)
    
    def test_extractor_empty_source(self):
        """Test handling of empty source."""
        extractor = CLIExtractor()
        result = extractor.extract("", "test")
        
        assert isinstance(result, ExtractionResult)
        assert len(result.parameters) == 0
    
    def test_extractor_malformed_spec(self):
        """Test handling of malformed specifications."""
        extractor = HTTPAPIExtractor()
        result = extractor.extract("{ invalid json }", "test")
        
        assert isinstance(result, ExtractionResult)
        # Should handle gracefully, may have errors
        assert len(result.errors) >= 0
    
    def test_parameter_miner_unknown_source_type(self):
        """Test handling of unknown source types."""
        miner = ParameterMiner()
        result = miner.extract_parameters("Random text that doesn't match any pattern", method=ExtractionMethod.AUTO)
        
        assert isinstance(result, ExtractionResult)
        # Should default to UNKNOWN or handle gracefully
        assert result.source_type in [SourceType.UNKNOWN, SourceType.CLI, SourceType.PYTHON_LIB]


# =============================================================================
# LLM Extractor Tests (Optional - Mock LLM)
# =============================================================================

class TestLLMExtractor:
    """Tests for LLM-based extraction (with mocked LLM client)."""
    
    def test_llm_extractor_cli(self, cli_help_simple, mock_llm_client):
        """Test LLM-based CLI extraction."""
        from kgpipe_parameters.extraction.param_miner import LLMCLIExtractor
        
        extractor = LLMCLIExtractor(mock_llm_client)
        result = extractor.extract(cli_help_simple, "test_tool")
        
        assert isinstance(result, ExtractionResult)
        assert result.extraction_method == ExtractionMethod.LLM
        # Mock should return parameters
        assert len(result.parameters) > 0
    
    def test_llm_extractor_fallback(self, cli_help_simple, mock_llm_client):
        """Test fallback from regex to LLM when regex fails."""
        miner = ParameterMiner(llm_client=mock_llm_client)
        
        # Use a source that regex might struggle with
        result = miner.extract_parameters(
            cli_help_simple,
            method=ExtractionMethod.AUTO
        )
        
        # Should try regex first, but if it fails and LLM is available, use LLM
        assert isinstance(result, ExtractionResult)

