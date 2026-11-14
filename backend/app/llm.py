"""Minimal chat client for OpenAI-compatible and Anthropic APIs."""

from __future__ import annotations

import os
import time
from typing import Protocol

import httpx


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict]) -> str:
        """`messages` is a list of {"role": "user"|"assistant", "content": str}."""
        ...


MAX_RATE_LIMIT_WAIT = 20.0  # seconds; longer waits fall back to rule-based suggestions instead


def post_with_rate_limit_retry(http: httpx.Client, url: str, body: dict, sleep=time.sleep) -> httpx.Response:
    """POST once; on a 429 whose Retry-After is short, wait it out and try exactly once more.
    Free tiers (e.g. Groq's tokens-per-minute cap) hit this often under light load."""
    response = http.post(url, json=body)
    if response.status_code == 429:
        try:
            wait = float(response.headers.get("retry-after", ""))
        except ValueError:
            wait = None
        if wait is not None and wait <= MAX_RATE_LIMIT_WAIT:
            sleep(wait)
            response = http.post(url, json=body)
    response.raise_for_status()
    return response


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
        response = post_with_rate_limit_retry(self._http, "/chat/completions", body)
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
        response = post_with_rate_limit_retry(self._http, "/v1/messages", body)
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
