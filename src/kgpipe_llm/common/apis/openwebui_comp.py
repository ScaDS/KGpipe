"""OpenWebUI completion helpers.

OpenWebUI often exposes OpenAI-compatible endpoints, so these wrappers delegate
to the OpenAI-compatible implementation while keeping a dedicated module.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type

from pydantic import BaseModel

from .openai_comp import (
    openai_call_with_json_out,
    openai_call_with_tool,
    pydantic_to_openai_tool,
    schemadict_to_openai_tool,
)


def openwebui_call_with_json_out(**kwargs: Any) -> Dict[str, Any]:
    """OpenWebUI wrapper for JSON-mode completions."""
    return openai_call_with_json_out(**kwargs)


def openwebui_call_with_tool(
    *,
    endpoint_url: str,
    api_key: str,
    model_name: str,
    user_content: str,
    pyd_model: Type[BaseModel] | Dict[str, Any],
    system_prompt: str = "You are a careful JSON-LD KG engineering assistant.",
    seed: str = "",
) -> Tuple[Dict[str, Any], BaseModel | Dict[str, Any]]:
    """OpenWebUI wrapper for tool-calling structured output."""
    return openai_call_with_tool(
        endpoint_url=endpoint_url,
        api_key=api_key,
        model_name=model_name,
        user_content=user_content,
        pyd_model=pyd_model,
        system_prompt=system_prompt,
        seed=seed,
    )
