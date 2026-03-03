"""OpenAI-compatible completion helpers with structured-output fallback."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple, Type

import requests
from pydantic import BaseModel

TIMEOUT = 900


def schemadict_to_openai_tool(
    schema_dict: Dict[str, Any],
    *,
    name: str,
    description: Optional[str] = None,
    additional_properties: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Convert a JSON schema dictionary into an OpenAI tool schema.
    """
    params = dict(schema_dict)
    params.pop("title", None)
    if additional_properties is not None and params.get("type") == "object":
        params["additionalProperties"] = additional_properties
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (description or schema_dict.get("description") or f"{name} parameters").strip(),
            "parameters": params,
        },
    }


def pydantic_to_openai_tool(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    additional_properties: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Convert a Pydantic model into an OpenAI tool schema.
    """
    schema = model.model_json_schema()
    resolved_name = name or model.__name__
    resolved_description = (
        description
        or (model.__doc__ or "").strip()
        or schema.get("description")
        or f"{resolved_name} parameters"
    )
    return schemadict_to_openai_tool(
        schema,
        name=resolved_name,
        description=resolved_description,
        additional_properties=additional_properties,
    )


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def openai_call_with_json_out(
    *,
    endpoint_url: str,
    api_key: str,
    model_name: str,
    user_content: str,
    system_prompt: str = "You are a careful JSON-LD KG engineering assistant.",
    seed: str = "",
    response_format: str = "json_object",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 1,
    }

    if response_format:
        payload["response_format"] = {"type": "json_object"}
    if seed:
        payload["seed"] = seed

    response = requests.post(
        endpoint_url,
        headers=_build_headers(api_key),
        json=payload,
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        print(response.content.decode("utf-8"))
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception as exc:
        print(f"Error parsing JSON: {exc}")
        return content


def _validate_or_passthrough(
    args: Dict[str, Any], pyd_model: Type[BaseModel] | Dict[str, Any]
) -> BaseModel | Dict[str, Any]:
    if isinstance(pyd_model, dict):
        return args
    return pyd_model.model_validate(args)


def _fallback_structured_output(
    *,
    endpoint_url: str,
    api_key: str,
    model_name: str,
    user_content: str,
    system_prompt: str,
    seed: str,
    pyd_model: Type[BaseModel] | Dict[str, Any],
) -> Tuple[Dict[str, Any], BaseModel | Dict[str, Any]]:
    fallback_args = openai_call_with_json_out(
        endpoint_url=endpoint_url,
        api_key=api_key,
        model_name=model_name,
        user_content=user_content,
        system_prompt=system_prompt,
        seed=seed,
        response_format="json_object",
    )
    if not isinstance(fallback_args, dict):
        raise ValueError("Fallback response_format=json_object did not return JSON object.")
    return fallback_args, _validate_or_passthrough(fallback_args, pyd_model)


def openai_call_with_tool(
    *,
    endpoint_url: str,
    api_key: str,
    model_name: str,
    user_content: str,
    pyd_model: Type[BaseModel] | Dict[str, Any],
    system_prompt: str = "You are a careful JSON-LD KG engineering assistant.",
    seed: str = "",
) -> Tuple[Dict[str, Any], BaseModel | Dict[str, Any]]:
    if isinstance(pyd_model, dict):
        tool = schemadict_to_openai_tool(
            pyd_model,
            name="CustomJsonSchema",
            additional_properties=True,
        )
    else:
        tool = pydantic_to_openai_tool(pyd_model, additional_properties=False)

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": tool["function"]["name"]}},
        "temperature": 1,
    }
    if seed:
        payload["seed"] = seed

    response = requests.post(
        endpoint_url,
        headers=_build_headers(api_key),
        json=payload,
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        print(response.content.decode("utf-8"))
    response.raise_for_status()
    data = response.json()

    try:
        tool_calls = data["choices"][0]["message"].get("tool_calls", [])
        if not tool_calls:
            raise ValueError("No tool calls in response.")
        args = json.loads(tool_calls[0]["function"]["arguments"])
        return args, _validate_or_passthrough(args, pyd_model)
    except Exception as exc:
        print(f"Tool-call structured parsing failed, using JSON fallback: {exc}")
        return _fallback_structured_output(
            endpoint_url=endpoint_url,
            api_key=api_key,
            model_name=model_name,
            user_content=user_content,
            system_prompt=system_prompt,
            seed=seed,
            pyd_model=pyd_model,
        )
