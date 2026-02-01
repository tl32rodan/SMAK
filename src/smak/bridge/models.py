"""Adapters for internal model services."""

from __future__ import annotations

import os
from typing import Any, Sequence

import httpx
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.openai_like import OpenAILike

_DEFAULT_NOMIC_API_BASE = "http://localhost:8787"
_DEFAULT_NOMIC_MODEL = "nomic-embed-text-v1.5"
_DEFAULT_QWEN_API_BASE = "http://localhost:8000/v1"
_DEFAULT_GPT_OSS_API_BASE = "http://localhost:8001/v1"
_DEFAULT_LLM_MODEL = "qwen2.5-7b-instruct"


def _resolve_api_base(base: str) -> str:
    return base.rstrip("/")


class InternalNomicEmbedding(BaseEmbedding):
    """Embedding adapter for the internal Nomic server."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
        embed_batch_size: int = 64,
    ) -> None:
        api_base = api_base or os.environ.get("SMAK_NOMIC_API_BASE", _DEFAULT_NOMIC_API_BASE)
        model = model or os.environ.get("SMAK_NOMIC_MODEL", _DEFAULT_NOMIC_MODEL)
        super().__init__(model_name=model, embed_batch_size=embed_batch_size)
        self.api_base = _resolve_api_base(api_base)
        self.model = model
        self.timeout = timeout
        self.headers = headers or {}
        self.client = client or httpx.Client(timeout=self.timeout)

    def _embedding_endpoint(self) -> str:
        if self.api_base.endswith("/v1"):
            return f"{self.api_base}/embeddings"
        return f"{self.api_base}/v1/embeddings"

    def _post_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": list(texts)}
        response = self.client.post(
            self._embedding_endpoint(), json=payload, headers=self.headers
        )
        response.raise_for_status()
        data = response.json()
        if "data" in data:
            return _normalize_openai_embeddings(data["data"], len(texts))
        if "embeddings" in data:
            return list(data["embeddings"])
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
        embeddings = await self._aget_text_embeddings([text])
        return embeddings[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": list(texts)}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self._embedding_endpoint(), json=payload, headers=self.headers
            )
        response.raise_for_status()
        data = response.json()
        if "data" in data:
            return _normalize_openai_embeddings(data["data"], len(texts))
        if "embeddings" in data:
            return list(data["embeddings"])
        raise ValueError("Unexpected response format from Nomic embedding service.")


def _normalize_openai_embeddings(data: Sequence[dict[str, Any]], size: int) -> list[list[float]]:
    embeddings: list[list[float] | None] = [None] * size
    for item in data:
        index = item.get("index")
        if index is None:
            index = len([entry for entry in embeddings if entry is not None])
        embeddings[index] = item["embedding"]
    if any(embedding is None for embedding in embeddings):
        raise ValueError("Embedding response did not include all requested entries.")
    return [embedding for embedding in embeddings if embedding is not None]


def build_internal_llm(
    *,
    provider: str = "qwen",
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> OpenAILike:
    """Create an OpenAILike client pointing at internal Qwen or GPT-OSS endpoints."""

    normalized_provider = provider.lower()
    if normalized_provider == "qwen":
        base = api_base or os.environ.get("SMAK_QWEN_API_BASE", _DEFAULT_QWEN_API_BASE)
        model_name = model or os.environ.get("SMAK_QWEN_MODEL", _DEFAULT_LLM_MODEL)
    elif normalized_provider in {"gpt-oss", "gpt_oss"}:
        base = api_base or os.environ.get(
            "SMAK_GPT_OSS_API_BASE", _DEFAULT_GPT_OSS_API_BASE
        )
        model_name = model or os.environ.get("SMAK_GPT_OSS_MODEL", "gpt-oss")
    else:
        raise ValueError(f"Unknown internal provider '{provider}'.")
    return OpenAILike(
        model=model_name,
        api_base=_resolve_api_base(base),
        api_key=api_key or os.environ.get("SMAK_LLM_API_KEY", "internal"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = ["InternalNomicEmbedding", "build_internal_llm"]
