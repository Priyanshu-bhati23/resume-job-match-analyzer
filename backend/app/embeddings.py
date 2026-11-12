"""Embedding backends. `calibration` maps raw cosine similarity onto 0..1."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import httpx
import numpy as np

_WORD = re.compile(r"[a-z0-9+#./-]*[a-z0-9+#]")
_STOP = frozenset(
    "a an and are as at be by for from in is it of on or that the to with we you our your will have has "
    "using used use work working experience years strong good ability skills including such etc".split()
)


class Embedder(Protocol):
    name: str
    calibration: tuple[float, float]  # cosine at/below low -> 0.0, at/above high -> 1.0

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    return (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)).astype("float32")


def _stem(w: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix) and not w.endswith("ss"):
            return w[: -len(suffix)]
    return w


class HashingEmbedder:
    """Offline lexical embedder (signed feature hashing of stemmed words and bigrams).

    Matches shared vocabulary, not meaning: "built REST services" and "developed
    HTTP APIs" score low. Fine for tests and demos; use a real model in production.
    """

    name = "hashing-4096"
    calibration = (0.05, 0.45)

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            words = [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]
            for feature in words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]:
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                out[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0 if digest[4] & 1 else -1.0
        return _normalise(np.sign(out) * np.log1p(np.abs(out)))


class OpenAIEmbedder:
    """Any OpenAI-compatible /embeddings endpoint."""

    calibration = (0.20, 0.60)

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", transport=None):
        self.name = model
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
            transport=transport,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 96):
            response = self._http.post("/embeddings", json={"model": self.name, "input": texts[start : start + 96]})
            response.raise_for_status()
            vectors.extend(item["embedding"] for item in sorted(response.json()["data"], key=lambda d: d["index"]))
        return _normalise(np.asarray(vectors, dtype="float32"))


def from_env() -> Embedder:
    model, key = os.getenv("EMBEDDING_MODEL"), os.getenv("OPENAI_API_KEY")
    if model and key:
        embedder = OpenAIEmbedder(model, key, os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        low, high = (
            os.getenv("EMBEDDING_CALIBRATION", "").split(",") if os.getenv("EMBEDDING_CALIBRATION") else (None, None)
        )
        if low and high:
            embedder.calibration = (float(low), float(high))
        return embedder
    return HashingEmbedder()
