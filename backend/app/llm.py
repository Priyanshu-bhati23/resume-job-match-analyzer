"""Minimal chat client for OpenAI-compatible and Anthropic APIs."""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict]) -> str:
        """`messages` is a list of {"role": "user"|"assistant", "content": str}."""
        ...


class OpenAICompatLLM:
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", transport=None):
        self.name = model
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=90,
            transport=transport,
        )

    def complete(self, system: str, messages: list[dict]) -> str:
        body = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": {"type": "json_object"},
        }
        response = self._http.post("/chat/completions", json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""


class AnthropicLLM:
    def __init__(self, model: str, api_key: str, transport=None):
        self.name = model
        self._http = httpx.Client(
            base_url="https://api.anthropic.com",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=90,
            transport=transport,
        )

    def complete(self, system: str, messages: list[dict]) -> str:
        body = {"model": self.name, "max_tokens": 2000, "system": system, "messages": messages}
        response = self._http.post("/v1/messages", json=body)
        response.raise_for_status()
        return "".join(b.get("text", "") for b in response.json()["content"] if b.get("type") == "text")


def from_env() -> LLM | None:
    """A client when LLM_MODEL and a key are set; otherwise None (rule-based suggestions)."""
    model = os.getenv("LLM_MODEL")
    if not model:
        return None
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicLLM(model, os.environ["ANTHROPIC_API_KEY"])
    if os.getenv("OPENAI_API_KEY"):
        return OpenAICompatLLM(
            model, os.environ["OPENAI_API_KEY"], os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
    return None
