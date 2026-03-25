"""Ollama-compatible completion helper."""

from __future__ import annotations

import json
from typing import Any, Dict, Type

import requests
from pydantic import BaseModel

TIMEOUT = 300


def ollama_call(
    *,
    endpoint_url: str,
    api_key: str,
    schema_class: Type[BaseModel] | Dict[str, Any] | str,
    model_name: str,
    user_content: str,
    system_prompt: str = "You are a careful JSON-LD KG engineering assistant.",
    seed: str = "",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": user_content,
        "stream": False,
    }

    if isinstance(schema_class, str):
        pass
    elif isinstance(schema_class, dict):
        payload["format"] = schema_class
    else:
        payload["format"] = schema_class.model_json_schema()

    if system_prompt:
        payload["system"] = system_prompt
    if seed:
        payload["seed"] = seed

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key

    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
        if response.status_code != 200:
            print(f"Request failed: {response.status_code} - {response.text}")
            return {}

        raw_output = response.json()["response"]
        if isinstance(schema_class, str):
            return raw_output
        parsed_output = json.loads(raw_output)
        if not isinstance(schema_class, dict):
            schema_class.model_validate(parsed_output)
        return parsed_output
    except Exception as exc:
        print(f"Error processing LLM response: {exc}")
        return {}
