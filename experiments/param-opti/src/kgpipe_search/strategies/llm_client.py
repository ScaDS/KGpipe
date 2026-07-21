from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


class ChatCompletionClient(Protocol):
    def complete(self, *, system: str, user: str) -> str:
        ...


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass
class OpenAICompatibleClient:
    """
    Minimal OpenAI-compatible chat client.

    Configuration via environment variables:
    - endpoint: KGPipe_SEARCH_LLM_ENDPOINT, OPENAI_BASE_URL, OPENAI_API_BASE
    - token: KGPipe_SEARCH_LLM_TOKEN, OPENAI_API_KEY
    - model: KGPipe_SEARCH_LLM_MODEL (default: gpt-4o-mini)
    """

    endpoint: str
    token: str
    model: str = "gpt-4o-mini"
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        endpoint = _env_first(
            "KGPipe_SEARCH_LLM_ENDPOINT",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
        )
        token = _env_first("KGPipe_SEARCH_LLM_TOKEN", "OPENAI_API_KEY")
        if not endpoint:
            raise ValueError(
                "LLM endpoint not configured. Set KGPipe_SEARCH_LLM_ENDPOINT or OPENAI_BASE_URL."
            )
        if not token:
            raise ValueError(
                "LLM token not configured. Set KGPipe_SEARCH_LLM_TOKEN or OPENAI_API_KEY."
            )

        model = os.environ.get("KGPipe_SEARCH_LLM_MODEL", "gpt-4o-mini")
        return cls(endpoint=endpoint.rstrip("/"), token=token, model=model)

    def complete(self, *, system: str, user: str) -> str:
        url = f"{self.endpoint}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        choices: List[Dict[str, Any]] = raw.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response missing choices: {raw!r}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"LLM response missing message content: {raw!r}")
        return content
