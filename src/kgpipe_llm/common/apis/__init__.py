"""API-specific completion backends."""

from .ollama_comp import ollama_call
from .openai_comp import (
    openai_call_with_json_out,
    openai_call_with_tool,
    pydantic_to_openai_tool,
    schemadict_to_openai_tool,
)
from .openwebui_comp import openwebui_call_with_json_out, openwebui_call_with_tool

__all__ = [
    "ollama_call",
    "openai_call_with_json_out",
    "openai_call_with_tool",
    "openwebui_call_with_json_out",
    "openwebui_call_with_tool",
    "pydantic_to_openai_tool",
    "schemadict_to_openai_tool",
]
