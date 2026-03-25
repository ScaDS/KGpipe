"""
Core LLM client and base task functionality for data integration tasks.
"""

import os
from typing import Any, Dict, Generic, Optional, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from .api_utils import get_token_count
from .apis.ollama_comp import ollama_call
from .apis.openai_comp import openai_call_with_json_out, openai_call_with_tool
from .apis.openwebui_comp import openwebui_call_with_json_out, openwebui_call_with_tool

load_dotenv()

OPENAI_V1_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

GPT_MODELS_EXTRA = ["o4-mini", "o1-mini", "o1-preview"]
OPENAI_LIKE_TYPES = {"openai", "openwebui"}

T = TypeVar("T", bound=BaseModel)


def _infer_api_type(model_name: str, endpoint_url: str) -> str:
    endpoint = (endpoint_url or "").lower()
    if "localhost:11434" in endpoint or endpoint.endswith("/api/generate"):
        return "ollama"
    if "openwebui" in endpoint:
        return "openwebui"
    if model_name.startswith("gpt") or model_name in GPT_MODELS_EXTRA:
        return "openai"
    if endpoint.endswith("/v1/chat/completions"):
        return "openai"
    if endpoint.endswith("/api/chat/completions"):
        return "openwebui"
    return "ollama"


def _resolve_openai_like_endpoint(endpoint_url: str) -> str:
    if endpoint_url and "chat/completions" in endpoint_url:
        return endpoint_url
    return OPENAI_V1_COMPLETIONS_URL


class LLMClient:
    """Client for interacting with LLM APIs with structured output validation."""

    def __init__(
        self,
        endpoint_url: str = "http://localhost:11434/api/generate",
        model_name: str = "gemma3:27B",
        token: str = "",
        seed: str = "",
        api_type: Optional[str] = None,
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.token = token
        self.seed = seed
        self.api_type = api_type or _infer_api_type(model_name, endpoint_url)

    def send_prompt(
        self,
        prompt: str,
        schema_class: type[T] | str | Dict[str, Any],
        system_prompt: str = "",
    ) -> Any:
        """
        Send a prompt to the LLM and validate the response against a Pydantic schema.
        """
        print("INPUT_TOKEN_COUNT", get_token_count(prompt))

        if self.api_type in OPENAI_LIKE_TYPES:
            endpoint = _resolve_openai_like_endpoint(self.endpoint_url)
            json_call = (
                openwebui_call_with_json_out if self.api_type == "openwebui" else openai_call_with_json_out
            )
            tool_call = openwebui_call_with_tool if self.api_type == "openwebui" else openai_call_with_tool

            if isinstance(schema_class, str):
                print(f"INFO: {self.api_type}_call_with_json_out {type(schema_class)}")
                return json_call(
                    endpoint_url=endpoint,
                    api_key=self.token,
                    model_name=self.model_name,
                    user_content=prompt,
                    system_prompt=system_prompt,
                    seed=self.seed,
                    response_format=schema_class,
                )

            print(f"INFO: {self.api_type}_call_with_tool {type(schema_class)}")
            dict_val, _model_val = tool_call(
                endpoint_url=endpoint,
                api_key=self.token,
                model_name=self.model_name,
                user_content=prompt,
                pyd_model=schema_class,
                system_prompt=system_prompt,
                seed=self.seed,
            )
            return dict_val

        print(f"INFO: ollama_call with {type(schema_class)}")
        return ollama_call(
            endpoint_url=self.endpoint_url,
            api_key=self.token,
            model_name=self.model_name,
            user_content=prompt,
            schema_class=schema_class,
            system_prompt=system_prompt,
            seed=self.seed,
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

def get_config_from_env() -> LlmAPIConfig:
    """Get the configuration for an LLM API from the environment."""
    opt_llm_endpoint_url = os.getenv("LLM_ENDPOINT_URL")
    opt_llm_model_name = os.getenv("DEFAULT_LLM_MODEL_NAME", "gemma3:27B")
    opt_ollama_token = os.getenv("OLLAMA_TOKEN")
    opt_openai_token = os.getenv("OPENAI_TOKEN")
    llm_seed = os.getenv("LLM_SEED", "")
    opt_context_window = int(os.getenv("CONTEXT_WINDOW", 16384))

    print(
        "INFO: get_config_from_env",
        opt_llm_endpoint_url,
        opt_llm_model_name,
        opt_ollama_token,
        opt_openai_token,
        llm_seed,
        opt_context_window,
    )

    if opt_llm_endpoint_url is None:
        raise ValueError("LLM_ENDPOINT_URL must be set.")

    return LlmAPIConfig(
        endpoint_url=opt_llm_endpoint_url,
        model_name=opt_llm_model_name,
        ollama_token=opt_ollama_token,
        openai_token=opt_openai_token,
        seed=llm_seed,
        context_window=opt_context_window,
    )


def get_client_from_env() -> LLMClient:
    """Get the client for an LLM API from the environment."""
    config = get_config_from_env()
    print(f"INFO: get_client_from_env {config.model_name}")
    api_type = _infer_api_type(config.model_name, config.endpoint_url)
    token = config.ollama_token if api_type == "ollama" else config.openai_token
    print(f"INFO: get_client_from_env with {api_type}")
    return LLMClient(
        endpoint_url=config.endpoint_url,
        model_name=config.model_name,
        token=token or "",
        seed=config.seed,
        api_type=api_type,
    )


# Backward compatibility for modules importing a shared default client.
default_client = LLMClient()
