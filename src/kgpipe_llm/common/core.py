"""
Core LLM client and base task functionality for data integration tasks.
"""

import requests
import json
from typing import Optional, TypeVar, Generic, Dict
from pydantic import BaseModel
from typing import AnyStr
import os
from .api_utils import openai_call_with_tool, openai_call_with_json_out, ollama_call, get_token_count

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

from dotenv import load_dotenv
load_dotenv()

OPENAI_V1_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

GPT_MODELS_EXTRA = ["o4-mini", "o1-mini", "o1-preview"]

class LLMClient:
    """Client for interacting with Ollama LLM API with structured output validation."""
    
    def __init__(self, endpoint_url: str = "http://localhost:11434/api/generate", 
                 model_name: str = "gemma3:27B",
                 token: str = "",
                 seed: str = ""):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.token = token
        self.seed = seed
        if model_name.startswith("gpt") or model_name in GPT_MODELS_EXTRA:
            self.api_type = "openai"
        else:
            self.api_type = "ollama"


    # def send_message(self, messages: list[dict], schema_class: type[T] | str, system_prompt: str = "") -> Optional[T] | str:
    #     """
    #     Send a message to the LLM and validate the response against a Pydantic schema.
    #     """
    #     payload = {
    #         "model": self.model_name,
    #         "messages": messages,
    #         "stream": False,
    #     }
        
    #     if system_prompt and system_prompt != "":
    #         payload["system"] = system_prompt
            
    #     if isinstance(schema_class, str):
    #         # For raw output, don't set format
    #         pass
    #     else:
    #         # For structured output, set the JSON schema format
    #         payload["format"] = schema_class.model_json_schema()
            
    #     headers = {
    #         "Content-Type": "application/json",
    #     }

    #     if self.token and self.token != "":
    #         headers["Authorization"] = f"Bearer {self.token}"
    #         headers["X-API-Key"] = self.token
            
    #     try:
    #         response = requests.post(
    #             self.endpoint_url,
    #             headers=headers,
    #             json=payload,
    #             timeout=30
    #         )
    #         print(response.json())
            
    #         if response.status_code == 200:
    #             raw_output = response.json()["response"]
    #             if isinstance(schema_class, str):
    #                 return raw_output
    #             else:
    #                 parsed_output = json.loads(raw_output)
    #                 result = schema_class.model_validate(parsed_output)
    #                 return result
    #         else:
    #             print(f"Request failed: {response.status_code} - {response.text}")
    #             return None
                
    #     except Exception as e:
    #         print(f"Error processing LLM response: {e}")
    #         return None


    def send_prompt(self, prompt: str, schema_class: type[T] | str | Dict, system_prompt: str = "") ->  Dict:
        """
        Send a prompt to the LLM and validate the response against a Pydantic schema.
        
        Args:
            prompt: The text prompt to send to the LLM
            schema_class: The Pydantic model class to validate the response against, or str for raw output
            
        Returns:
            Validated Pydantic model instance, raw string, or None if validation fails
        """

        print("INPUT_TOKEN_COUNT", get_token_count(prompt))

        if self.api_type == "openai":
            if isinstance(schema_class, str):
                print(f"INFO: openai_call_with_json_out {type(schema_class)}")
                return openai_call_with_json_out(
                    endpoint_url=OPENAI_V1_COMPLETIONS_URL,
                    api_key=self.token,
                    model_name=self.model_name,
                    user_content=prompt,
                    system_prompt=system_prompt,
                    seed=self.seed,
                    response_format=schema_class
                )

            else:
                # special return type for openai
                print(f"INFO: openai_call_with_tool {type(schema_class)}")
                dict_val, model_val = openai_call_with_tool(
                    endpoint_url=OPENAI_V1_COMPLETIONS_URL,
                    api_key=self.token,
                    model_name=self.model_name,
                    user_content=prompt,
                    pyd_model=schema_class,
                    system_prompt=system_prompt,
                    seed=self.seed
                )
                return dict_val
        else:
            print(f"INFO: ollama_call with {type(schema_class)}")
            return ollama_call(
                endpoint_url=self.endpoint_url,
                api_key=self.token,
                model_name=self.model_name,
                user_content=prompt,
                schema_class=schema_class,
                system_prompt=system_prompt,
                seed=self.seed
            )


class BaseTask(Generic[T]):
    """Base class for LLM-based data integration tasks."""
    
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
    
    def execute(self, *args, **kwargs) -> Optional[T]:
        """Execute the task and return structured output."""
        raise NotImplementedError("Subclasses must implement execute method")


class LlmAPIConfig(BaseModel):
    """Configuration for an LLM API."""
    endpoint_url: str
    model_name: str
    ollama_token: Optional[str]
    openai_token: Optional[str]
    seed: str
    context_window: int

    # def __init__(self, endpoint_url: str, model_name: str, ollama_token: str, openai_token: str):
    #     self.endpoint_url = endpoint_url
    #     self.model_name = model_name
    #     self.ollama_token = ollama_token
    #     self.openai_token = openai_token

def get_config_from_env() -> LlmAPIConfig:
    """Get the configuration for an LLM API from the environment."""

    OPT_LLM_ENDPOINT_URL = os.getenv("LLM_ENDPOINT_URL")
    OPT_LLM_MODEL_NAME = os.getenv("DEFAULT_LLM_MODEL_NAME")
    OPT_OLLAMA_TOKEN = os.getenv("OLLAMA_TOKEN")
    OPT_OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
    LLM_SEED = os.getenv("LLM_SEED", "")
    OPT_CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", 16384))

    print(f"INFO: get_config_from_env {OPT_LLM_ENDPOINT_URL} {OPT_LLM_MODEL_NAME} {OPT_OLLAMA_TOKEN} {OPT_OPENAI_TOKEN} {LLM_SEED} {OPT_CONTEXT_WINDOW}")

    # TODO requires one token to be set
    if OPT_LLM_ENDPOINT_URL is None or (OPT_LLM_MODEL_NAME is None and OPT_OLLAMA_TOKEN is None and OPT_OPENAI_TOKEN is None):
        # raise ValueError("LLM_ENDPOINT_URL, LLM_MODEL_NAME, OLLAMA_TOKEN, and OPENAI_TOKEN must be set. Also, CONTEXT_WINDOW must be set.")
        raise ValueError("LLM_ENDPOINT_URL, LLM_MODEL_NAME, OLLAMA_TOKEN, and OPENAI_TOKEN must be set. Also, CONTEXT_WINDOW must be set.")

    return LlmAPIConfig(
        endpoint_url=OPT_LLM_ENDPOINT_URL,
        model_name=OPT_LLM_MODEL_NAME,
        ollama_token=OPT_OLLAMA_TOKEN,
        openai_token=OPT_OPENAI_TOKEN,
        seed=LLM_SEED,
        context_window=OPT_CONTEXT_WINDOW,
    )


def get_client_from_env() -> LLMClient:
    """Get the client for an LLM API from the environment."""
    config = get_config_from_env()
    print(f"INFO: get_client_from_env {config.model_name}")
    if config.model_name.startswith("gpt") or config.model_name in GPT_MODELS_EXTRA:
        api_type = "openai"
    else:
        api_type = "ollama"
    print(f"INFO: get_client_from_env with {api_type}")
    return LLMClient(
        endpoint_url=config.endpoint_url,
        model_name=config.model_name,
        token=config.ollama_token if api_type == "ollama" else config.openai_token,
        seed=config.seed
    )