"""
Tests for keyword-based chunk scoring / filtering.
"""

import pytest

from kgpipe_parameters.extraction.chunk_filter import (
    score_chunk,
    has_parameter_signals,
    KEYWORD_SETS,
    _detect_language,
)


# ── Language detection ──────────────────────────────────────────────────

class TestLanguageDetection:
    """Tests for _detect_language helper."""

    def test_python_extension(self):
        assert _detect_language("src/foo/bar.py") == "python"

    def test_java_extension(self):
        assert _detect_language("src/Main.java") == "java"

    def test_properties_extension(self):
        assert _detect_language("conf/server.properties") == "properties"

    def test_xml_extension(self):
        assert _detect_language("config.xml") == "xml"

    def test_dockerfile(self):
        assert _detect_language("Dockerfile") == "docker"
        assert _detect_language("docker-compose.yml") == "docker"

    def test_readme(self):
        assert _detect_language("README.md") == "readme"
        assert _detect_language("INSTALL.txt") == "readme"

    def test_unknown_defaults_to_generic(self):
        assert _detect_language("random.xyz") == "generic"
        assert _detect_language(None) == "generic"


# ── Scoring ─────────────────────────────────────────────────────────────

class TestScoreChunk:
    """Tests for score_chunk."""

    def test_empty_text(self):
        score, matched = score_chunk("")
        assert score == 0
        assert matched == []

    def test_python_argparse(self):
        code = '''
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.5, help="Matching threshold")
'''
        score, matched = score_chunk(code, file_path="cli.py")
        assert score >= 3  # argparse, add_argument, default=, type=, help=
        assert "argparse" in matched
        assert "add_argument" in matched

    def test_python_dataclass(self):
        code = '''
from dataclasses import dataclass, field

@dataclass
class Config:
    threshold: float = 0.5
    batch_size: int = Field(default=32)
'''
        score, matched = score_chunk(code, file_path="config.py")
        assert score >= 2
        assert "@dataclass" in matched
        assert "Field(" in matched

    def test_python_no_signals(self):
        code = '''
def compute_arabic_segmenter(text):
    tokens = text.split()
    return [t for t in tokens if len(t) > 2]
'''
        score, matched = score_chunk(code, file_path="segmenter.py")
        assert score < 2  # No real parameter signals

    def test_java_option_annotation(self):
        code = '''
public class RunPARIS {
    @Option(name = "-n", usage = "number of iterations")
    int numIterations = 10;

    @Option(name = "-t", usage = "threshold")
    double threshold = 0.5;
}
'''
        score, matched = score_chunk(code, file_path="RunPARIS.java")
        assert score >= 1  # @Option is the signal; Java threshold is 1
        assert "@Option" in matched
        # The file still passes the filter (Java auto-lowers threshold to 1)
        assert has_parameter_signals(code, file_path="RunPARIS.java") is True

    def test_java_properties_access(self):
        code = '''
Properties props = new Properties();
props.load(new FileInputStream("config.properties"));
String value = props.getProperty("matchThreshold");
int maxIter = Integer.parseInt(props.getProperty("maxIterations"));
'''
        score, matched = score_chunk(code, file_path="Config.java")
        assert score >= 2
        assert "getProperty(" in matched
        assert "Properties" in matched

    def test_java_no_signals(self):
        code = '''
public class ArabicTokenizer {
    public List<String> tokenize(String text) {
        return Arrays.asList(text.split(" "));
    }
}
'''
        score, matched = score_chunk(code, file_path="ArabicTokenizer.java")
        assert score < 2

    def test_properties_file(self):
        content = '''
# Server configuration
server.port=8080
matching.threshold=0.5
max.iterations=100
'''
        score, matched = score_chunk(content, file_path="server.properties")
        assert score >= 1  # .properties files have low bar

    def test_xml_config(self):
        content = '''
<configuration>
    <property name="threshold" value="0.5"/>
    <param name="maxIterations" value="100"/>
</configuration>
'''
        score, matched = score_chunk(content, file_path="config.xml")
        assert score >= 2
        assert "<property" in matched
        assert "<param" in matched

    def test_docker_env(self):
        content = '''
FROM python:3.11
ENV THRESHOLD=0.5
ARG BUILD_VERSION=1.0
EXPOSE 8080
'''
        score, matched = score_chunk(content, file_path="Dockerfile")
        assert score >= 3
        assert "ENV " in matched
        assert "ARG " in matched
        assert "EXPOSE " in matched

    def test_readme_with_params(self):
        content = '''
# My Tool

## Usage

```bash
mytool --threshold 0.5 --output result.txt
```

## Configuration

- `threshold`: Matching threshold (default: 0.5)
- `max_iter`: Maximum iterations (default: 100)
'''
        score, matched = score_chunk(content, file_path="README.md")
        assert score >= 3

    def test_readme_no_params(self):
        content = '''
# My Project

This is a library for natural language processing.

## License

MIT License
'''
        score, matched = score_chunk(content, file_path="README.md")
        # Very few or no config signals
        assert score <= 2

    def test_explicit_language_override(self):
        code = "parser.add_argument('--foo')"
        score, matched = score_chunk(code, language="python")
        assert "add_argument" in matched


# ── has_parameter_signals ───────────────────────────────────────────────

class TestHasParameterSignals:
    """Tests for the boolean filter function."""

    def test_python_with_signals(self):
        code = 'parser = argparse.ArgumentParser()\nparser.add_argument("--x", default=5)'
        assert has_parameter_signals(code, file_path="cli.py") is True

    def test_python_without_signals(self):
        code = "x = 1 + 2\nprint(x)"
        assert has_parameter_signals(code, file_path="math.py") is False

    def test_threshold_override(self):
        code = "argparse"
        # With default threshold=2 this would fail (only 1 keyword)
        assert has_parameter_signals(code, file_path="x.py", threshold=2) is False
        # With threshold=1 it passes
        assert has_parameter_signals(code, file_path="x.py", threshold=1) is True

    def test_properties_low_bar(self):
        content = "key=value"
        # .properties files auto-lower threshold to 1
        assert has_parameter_signals(content, file_path="app.properties") is True

    def test_xml_low_bar(self):
        content = '<property name="x" value="1"/>'
        assert has_parameter_signals(content, file_path="config.xml") is True

    def test_java_config_class_passes(self):
        code = '''
public class AppConfig {
    @Option(name = "-t")
    double threshold = DEFAULT_THRESHOLD;
}
'''
        assert has_parameter_signals(code, file_path="AppConfig.java") is True

    def test_java_non_config_class_fails(self):
        code = '''
public class Utils {
    public static String trim(String s) {
        return s.trim();
    }
}
'''
        assert has_parameter_signals(code, file_path="Utils.java") is False


# ── Keyword set sanity ──────────────────────────────────────────────────

class TestKeywordSets:
    """Sanity checks on the keyword dictionaries."""

    def test_all_sets_non_empty(self):
        for name, kws in KEYWORD_SETS.items():
            assert len(kws) > 0, f"Keyword set '{name}' is empty"

    def test_no_empty_keywords(self):
        for name, kws in KEYWORD_SETS.items():
            for kw in kws:
                assert kw.strip() != "", f"Empty keyword in set '{name}'"

