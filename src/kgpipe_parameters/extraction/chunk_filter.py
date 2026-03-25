"""
Keyword-based chunk scoring for pre-filtering files before extraction.

Counts parameter-signal keywords in a text chunk and returns a relevance
score.  Files/chunks that score below a configurable threshold are skipped
entirely, preventing noise (e.g. Arabic segmenter scripts in CoreNLP) from
polluting both the regex and LLM extraction paths.

No embeddings, no extra dependencies — pure keyword counting.
"""

import re
from typing import Dict, List, Optional, Tuple

__all__ = ["score_chunk", "has_parameter_signals", "KEYWORD_SETS"]


# ── Keyword sets per language / file-type ────────────────────────────────

_PYTHON_KEYWORDS: List[str] = [
    # argparse / click / typer
    "argparse",
    "add_argument",
    "ArgumentParser",
    "click.option",
    "click.argument",
    "click.command",
    "typer.Option",
    "typer.Argument",
    # dataclass / pydantic
    "@dataclass",
    "Field(",
    "BaseModel",
    "BaseSettings",
    # general config signals
    "default=",
    "default_factory",
    "required=",
    "choices=",
    "type=",
    "nargs=",
    "help=",
    "metavar=",
    # plain constructor parameters (frameworks like valentine, etc.)
    "def __init__(self,",
    "self.__",
    "self._",
    # env vars
    "os.environ",
    "os.getenv",
    "environ.get",
    # configparser / yaml / json config
    "configparser",
    "ConfigParser",
    "config.get",
    "config[",
    "yaml.load",
    "yaml.safe_load",
    "json.load",
    # hydra / omegaconf
    "@hydra.main",
    "OmegaConf",
    "DictConfig",
]

_JAVA_KEYWORDS: List[str] = [
    # JCommander / picocli / commons-cli
    "@Option",
    "@Parameter",
    "@CommandLine",
    "@Command",
    ".addOption(",
    "Options(",
    "new Option(",
    "OptionBuilder",
    "CommandLine",
    # Java properties / config
    "getProperty(",
    "setProperty(",
    "properties.get(",
    "Properties",
    ".properties",
    "loadProperties",
    "getConfig(",
    "getString(",
    "getInt(",
    "getDouble(",
    "getBoolean(",
    # Spring
    "@Value(",
    "@ConfigurationProperties",
    "@RequestParam",
    "@PathVariable",
    # general
    "default:",
    "DEFAULT_",
    "CONFIG_",
    "PARAM_",
]

_PROPERTIES_KEYWORDS: List[str] = [
    # .properties files are inherently config
    "=",
    ":",
]

_XML_KEYWORDS: List[str] = [
    "<property",
    "<param",
    "<setting",
    "<config",
    "<option",
    "<entry",
    "name=",
    "value=",
    "default=",
]

_DOCKER_KEYWORDS: List[str] = [
    "ENV ",
    "ARG ",
    "EXPOSE ",
    "-e ",
    "--env",
    "environment:",
]

_README_KEYWORDS: List[str] = [
    # Parameter documentation signals
    "--",
    "-D",
    "default:",
    "Default:",
    "default=",
    "required",
    "optional",
    "parameter",
    "Parameter",
    "configuration",
    "Configuration",
    "config",
    "Config",
    "flag",
    "option",
    "threshold",
    "Usage:",
    "usage:",
    "Options:",
    "options:",
    "Arguments:",
    "arguments:",
]

_GENERIC_KEYWORDS: List[str] = [
    # Cross-language signals
    "default",
    "param",
    "config",
    "option",
    "argument",
    "threshold",
    "env",
    "setting",
    "flag",
]


KEYWORD_SETS: Dict[str, List[str]] = {
    "python": _PYTHON_KEYWORDS,
    "java": _JAVA_KEYWORDS,
    "properties": _PROPERTIES_KEYWORDS,
    "xml": _XML_KEYWORDS,
    "docker": _DOCKER_KEYWORDS,
    "readme": _README_KEYWORDS,
    "generic": _GENERIC_KEYWORDS,
}


# ── File extension → keyword set mapping ────────────────────────────────

_EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".scala": "java",       # close enough
    ".kt": "java",          # Kotlin shares Java config conventions
    ".properties": "properties",
    ".xml": "xml",
    ".yml": "docker",       # docker-compose, k8s manifests
    ".yaml": "docker",
    ".md": "readme",
    ".txt": "readme",
    ".rst": "readme",
}


def _detect_language(file_path: Optional[str]) -> str:
    """Guess the language/type from a file extension."""
    if not file_path:
        return "generic"
    # Handle Dockerfile* specially
    lower = file_path.lower()
    if "dockerfile" in lower or "docker-compose" in lower:
        return "docker"
    # Extension-based lookup
    for ext, lang in _EXT_TO_LANG.items():
        if lower.endswith(ext):
            return lang
    return "generic"


def score_chunk(
    text: str,
    file_path: Optional[str] = None,
    language: Optional[str] = None,
) -> Tuple[int, List[str]]:
    """
    Score a text chunk by counting parameter-signal keyword hits.

    Args:
        text: The text content to score.
        file_path: Optional file path (used to auto-detect language).
        language: Explicit language override (python, java, …).
                  If None, detected from *file_path*.

    Returns:
        (score, matched_keywords) — score is the number of distinct keyword
        matches found; matched_keywords lists which ones fired.
    """
    if not text:
        return 0, []

    lang = language or _detect_language(file_path)
    keywords = KEYWORD_SETS.get(lang, KEYWORD_SETS["generic"])

    matched: List[str] = []
    for kw in keywords:
        if kw in text:
            matched.append(kw)

    return len(matched), matched


def has_parameter_signals(
    text: str,
    file_path: Optional[str] = None,
    language: Optional[str] = None,
    threshold: int = 2,
) -> bool:
    """
    Return True if *text* contains at least *threshold* distinct
    parameter-signal keywords.

    For .properties and .xml files the threshold is automatically lowered
    to 1 because their content is inherently config-like.

    Args:
        text: The text content to check.
        file_path: Optional file path for language detection.
        language: Explicit language override.
        threshold: Minimum keyword hits required (default 2).

    Returns:
        True if the chunk passes the keyword filter.
    """
    lang = language or _detect_language(file_path)

    # .properties / .xml files are inherently config — lower bar.
    # Java files with *any* annotation-style signal are worth inspecting.
    if lang in ("properties", "xml", "java"):
        threshold = min(threshold, 1)

    score, _ = score_chunk(text, file_path=file_path, language=lang)
    return score >= threshold

