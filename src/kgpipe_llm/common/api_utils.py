"""Compatibility facade for API-specific LLM helpers.

Provider implementations live under ``kgpipe_llm.common.apis``.
"""

from __future__ import annotations

import os

from tiktoken import encoding_for_model

from .apis.ollama_comp import ollama_call
from .apis.openai_comp import (
    openai_call_with_json_out,
    openai_call_with_tool,
    pydantic_to_openai_tool,
    schemadict_to_openai_tool,
)


def get_token_count(text: str) -> int:
    """Return token count using the configured default GPT tokenizer."""
    model_name = os.getenv("DEFAULT_LLM_MODEL_NAME", "gpt-5-mini")
    if not model_name.startswith("gpt"):
        model_name = "gpt-5-mini"
    encoding = encoding_for_model(model_name)
    return len(encoding.encode(text))


__all__ = [
    "ollama_call",
    "openai_call_with_json_out",
    "openai_call_with_tool",
    "pydantic_to_openai_tool",
    "schemadict_to_openai_tool",
    "get_token_count",
]