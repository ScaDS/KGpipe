# specific LLM API utils (ollama, openai, etc.)
import json
import requests
from typing import Tuple, List
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel
from enum import Enum
from tiktoken import encoding_for_model
import os

TIMEOUT = 900 # 10 minutes

# def schemadict_to_openai_tool(
#     schema_dict: Dict[str, Any],
#     model_name: str,
#     *,
#     name: Optional[str] = None,
#     description: Optional[str] = None,
#     additional_properties: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Convert a schema dictionary to an OpenAI 'tools' entry (function calling).
#     """

#     schema = schema_dict

#     # We want the object schema under "parameters"
#     # Keep $defs so nested models/refs work.
#     params: Dict[str, Any] = {
#         "type": "object",
#         "properties": schema.get("properties", {}),
#         "required": schema.get("required", []),
#         "additionalProperties": additional_properties,
#     }
#     if "$defs" in schema:
#         params["$defs"] = schema["$defs"]

#     tool = {
#         "type": "function",
#         "function": {
#             "name": name or model_name,
#             "description": description or (model_name.__doc__ or "").strip() or f"{model_name} schema",
#             "parameters": params,
#         },
#     }
#     return tool

# def pydantic_to_openai_tool(
#     model: Type[BaseModel],
#     *,
#     name: Optional[str] = None,
#     description: Optional[str] = None,
#     additional_properties: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Convert a Pydantic v2 model to an OpenAI 'tools' entry (function calling).
#     """
#     # Pydantic v2 emits draft-2020-12 JSON Schema. OpenAI accepts schemas
#     # that look like draft-07/2019-09 object schemas, including $defs/$ref.
#     #schema = model.model_json_schema(ref_template="#/$defs/{model}")
#     schema = model.model_json_schema()
#     return schemadict_to_openai_tool(schema, model.__name__, name=name, description=description, additional_properties=additional_properties)

from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

def schemadict_to_openai_tool(
    schema_dict: Dict[str, Any],
    *,
    name: str,
    description: Optional[str] = None,
    additional_properties: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Convert a schema dictionary to an OpenAI 'tools' entry (function calling).
    Pass the schema through unchanged (array/object/etc.), only tweaking root-level keys.
    """
    # Copy so we don't mutate the caller's schema
    params = dict(schema_dict)

    # Titles are optional noise for tool schemas; drop them at root.
    params.pop("title", None)

    # Only inject additionalProperties if the root is an object.
    if additional_properties is not None and params.get("type") == "object":
        params["additionalProperties"] = additional_properties

    tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": (description or schema_dict.get("description") or f"{name} parameters").strip(),
            "parameters": params,
        },
    }
    return tool


def pydantic_to_openai_tool(
    model: Type[BaseModel],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    additional_properties: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Convert a Pydantic v2 model to an OpenAI 'tools' entry (function calling).
    Works for object models, RootModel[list[...]], unions, literals, etc.
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


# def schemadict_to_openai_tool(
#     schema_dict: Dict[str, Any],
#     *,
#     name: str,
#     description: Optional[str] = None,
#     additional_properties: Optional[bool] = None,
# ) -> Dict[str, Any]:
#     """
#     Convert a schema dictionary to an OpenAI 'tools' entry (function calling).
#     """
#     params: Dict[str, Any] = {
#         "type": "object",
#         "properties": schema_dict.get("properties", {}),
#         "required": schema_dict.get("required", []),
#     }
#     # Only set this if the caller asked to, otherwise leave Pydantic's default intact.
#     if additional_properties is not None:
#         params["additionalProperties"] = additional_properties

#     # Keep nested refs/defs
#     if "$defs" in schema_dict:
#         params["$defs"] = schema_dict["$defs"]

#     tool = {
#         "type": "function",
#         "function": {
#             "name": name,
#             "description": (description or schema_dict.get("description") or f"{name} parameters").strip(),
#             "parameters": params,
#         },
#     }
#     return tool


# def pydantic_to_openai_tool(
#     model: Type[BaseModel],
#     *,
#     name: Optional[str] = None,
#     description: Optional[str] = None,
#     additional_properties: Optional[bool] = None,
# ) -> Dict[str, Any]:
#     """
#     Convert a Pydantic v2 model to an OpenAI 'tools' entry (function calling).
#     """
#     # Pydantic v2 emits draft-2020-12 JSON Schema (with $defs). That's fine for OpenAI tools.
#     schema = model.model_json_schema()

#     resolved_name = name or model.__name__
#     # Prefer explicit description → model docstring → schema description → fallback
#     resolved_description = (
#         description
#         or (model.__doc__ or "").strip()
#         or schema.get("description")
#         or f"{resolved_name} parameters"
#     )

#     return schemadict_to_openai_tool(
#         schema,
#         name=resolved_name,
#         description=resolved_description,
#         additional_properties=additional_properties,
#     )

def openai_call_with_json_out(
    *,
    endpoint_url: str,
    api_key: str,
    model_name: str,
    user_content: str,
    system_prompt: str = "You are a careful JSON-LD KG engineering assistant.",
    seed: str = "",
    response_format: str = "json_object"
) -> Dict[str, Any]:

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],

        "temperature": 1
    }

    if response_format and response_format != "":
        print(f"INFO: openai_call_with_json_out response_format json_object")
        payload["response_format"] = {
            "type": "json_object"
        }

    if seed and seed != "":
        payload["seed"] = seed

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(resp.content.decode("utf-8"))
    resp.raise_for_status()
    resp_data = resp.json()

    content = resp_data["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return content

def openai_call_with_tool(
    *,
    endpoint_url: str,
    api_key: str,
    model_name: str,
    user_content: str,
    pyd_model: Type[BaseModel] | Dict,
    system_prompt: str = "You are a careful JSON-LD KG engineering assistant.",
    seed: str = ""
) -> Tuple[dict, BaseModel]:
    if isinstance(pyd_model, dict):
        print(f"INFO: openai_call_with_tool CUSTOM JSON SCHEMA")
        tool = schemadict_to_openai_tool(pyd_model, name="CustomJsonSchema", additional_properties=True)
    else:
        print(f"INFO: openai_call_with_tool Pydantic model {pyd_model.__name__}")
        tool = pydantic_to_openai_tool(pyd_model, additional_properties=False)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "tools": [tool],
        # Force the model to call our function so we get structured JSON back
        "tool_choice": {"type": "function", "function": {"name": tool["function"]["name"]}},
        "temperature": 1
    }

    if seed and seed != "":
        payload["seed"] = seed

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=TIMEOUT)
    if resp.status_code != 200:
        print(resp.content.decode("utf-8"))
    resp.raise_for_status()
    data = resp.json()

    # Extract tool call arguments
    choice = data["choices"][0]
    tool_calls = choice["message"].get("tool_calls", [])
    if not tool_calls:
        raise ValueError("Model did not return a tool call; check tool_choice or prompt.")

    args_json_str = tool_calls[0]["function"]["arguments"]
    args = json.loads(args_json_str)

    # Validate using Pydantic
    if isinstance(pyd_model, dict):
        validated = args
    else:
        validated = pyd_model.model_validate(args)
    return args, validated


def ollama_call(
    *,
    endpoint_url: str,
    api_key: str,
    schema_class: Type[BaseModel] | Dict | str,
    model_name: str,
    user_content: str,
    system_prompt: str = "You are a careful JSON-LD KG engineering assistant.",
    seed: str = ""
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "prompt": user_content,
        "stream": False,
    }

    # If schema_class is a string, we want raw output
    if isinstance(schema_class, str):
        # For raw output, don't set format
        pass
    elif isinstance(schema_class, Dict):
        payload["format"] = schema_class
    else:
        # For structured output, set the JSON schema format
        payload["format"] = schema_class.model_json_schema()

    if system_prompt and system_prompt != "":
        payload["system"] = system_prompt

    if seed and seed != "":
        payload["seed"] = seed
    
    headers = {
        "Content-Type": "application/json",
    }

    if api_key and api_key != "":
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key

    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            raw_output = response.json()["response"]
            if isinstance(schema_class, str):
                return raw_output
            elif isinstance(schema_class, Dict):
                return json.loads(raw_output)
            else:
                parsed_output = json.loads(raw_output)
                schema_class.model_validate(parsed_output)
                return parsed_output
        else:
            print(f"Request failed: {response.status_code} - {response.text}")
            return {}
            
    except Exception as e:
        print(f"Error processing LLM response: {e}")
        return {}


def get_token_count(text: str) -> int:
    """
    Get the token count of a text string.
    """
    model_name = os.getenv("DEFAULT_LLM_MODEL_NAME", "gpt-5-mini")
    if not model_name.startswith("gpt"):
        model_name = "gpt-5-mini"
    encoding = encoding_for_model(model_name)
    return len(encoding.encode(text))