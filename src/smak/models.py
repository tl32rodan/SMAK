"""Adapters for internal model services."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Sequence

try:  # pragma: no cover - dependency may be unavailable in minimal environments
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:  # pragma: no cover - dependency may be unavailable in minimal environments
    from llama_index.core.embeddings import BaseEmbedding
except ModuleNotFoundError:  # pragma: no cover
    class BaseEmbedding:  # type: ignore[override]
        def __init__(self, model_name: str, embed_batch_size: int) -> None:
            self.model_name = model_name
            self.embed_batch_size = embed_batch_size

try:  # pragma: no cover - dependency may be unavailable in minimal environments
    from llama_index.llms.openai_like import OpenAILike
except ModuleNotFoundError:  # pragma: no cover
    class OpenAILike:  # type: ignore[override]
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

_DEFAULT_NOMIC_API_BASE = "http://f15dtpai1:11436"
_DEFAULT_NOMIC_MODEL = "nomic_embed_text:latest"
_DEFAULT_QWEN_API_BASE = "http://f15dtpai1:11516/v1"
_DEFAULT_QWEN_LLM_MODEL = "qwen3_235B_A22B"


class InternalNomicEmbedding(BaseEmbedding):
    """Embedding adapter for the internal Nomic server."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        session: Any | None = None,
        embed_batch_size: int = 64,
    ) -> None:
        resolved_base = (
            api_base or os.environ.get("SMAK_NOMIC_API_BASE", _DEFAULT_NOMIC_API_BASE)
        ).rstrip("/")
        resolved_model = model or os.environ.get("SMAK_NOMIC_MODEL", _DEFAULT_NOMIC_MODEL)
        super().__init__(model_name=resolved_model, embed_batch_size=embed_batch_size)
        self.api_base = resolved_base
        self.model = resolved_model
        self.timeout = timeout
        self.headers = headers or {}
        self.session = session or (requests.Session() if requests else None)

    def _embedding_endpoint(self) -> str:
        return f"{self.api_base}/api/embed"

    def _post_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        if self.session is None:
            raise ModuleNotFoundError("requests is required for InternalNomicEmbedding")
        response = self.session.post(
            self._embedding_endpoint(),
            json={"model": self.model, "input": list(texts)},
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if "data" in payload:
            ordered = sorted(payload["data"], key=lambda d: d.get("index", 0))
            return [item["embedding"] for item in ordered]
        if "embeddings" in payload:
            return list(payload["embeddings"])
        raise ValueError("Unexpected response format from Nomic embedding service.")

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._post_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._post_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._post_embeddings(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._aget_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._aget_text_embeddings([text]))[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._post_embeddings, texts)

    def get_embedding_dimension(self, probe_text: str = "hello") -> int:
        return len(self._post_embeddings([probe_text])[0])


def build_internal_llm(
    *,
    provider: str = "qwen",
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> OpenAILike:
    if provider.lower() != "qwen":
        raise ValueError(f"Unknown internal provider '{provider}'.")
    return OpenAILike(
        model=model or os.environ.get("SMAK_QWEN_MODEL", _DEFAULT_QWEN_LLM_MODEL),
        api_base=(
            api_base or os.environ.get("SMAK_QWEN_API_BASE", _DEFAULT_QWEN_API_BASE)
        ).rstrip("/"),
        api_key=api_key or os.environ.get("SMAK_LLM_API_KEY", "internal"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = ["InternalNomicEmbedding", "build_internal_llm"]
