"""Adapters for internal model services."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Sequence

import requests
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.openai_like import OpenAILike

_DEFAULT_NOMIC_API_BASE = "http://f15dtpai1:11436"
_DEFAULT_NOMIC_MODEL = "nomic_embed_text:latest"
_DEFAULT_QWEN_API_BASE = "http://f15dtpai1:11516/v1"
_DEFAULT_QWEN_LLM_MODEL = "qwen3_235B_A22B"
_DEFAULT_GPT_API_BASE = "http://f15dtpai1:11517/v1"
_DEFAULT_GPT_OSS_LLM_MODEL = "gpt-oss-120b"


def _resolve_api_base(base: str) -> str:
    return base.rstrip("/")


class InternalNomicEmbedding(BaseEmbedding):
    """Embedding adapter for the internal Nomic server."""

    api_base: str | None = None
    model: str | None = None
    timeout: float = 30.0
    headers: dict[str, str] | None = None
    session: requests.Session | None = None

    def __init__(
        self,
        *,
        api_base: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        session: requests.Session | None = None,
        embed_batch_size: int = 64,
    ) -> None:
        api_base = api_base or os.environ.get("SMAK_NOMIC_API_BASE", _DEFAULT_NOMIC_API_BASE)
        model = model or os.environ.get("SMAK_NOMIC_MODEL", _DEFAULT_NOMIC_MODEL)
        super().__init__(model_name=model, embed_batch_size=embed_batch_size)
        self.api_base = _resolve_api_base(api_base)
        self.model = model
        self.timeout = timeout
        self.headers = headers or {}
        self.session = session or requests.Session()

    def _embedding_endpoint(self) -> str:
        return f"{self.api_base}/api/embed"

    def _post_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": list(texts)}
        response = self.session.post(
            self._embedding_endpoint(), json=payload, headers=self.headers, timeout=self.timeout
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
        response = await _async_post_embeddings(
            self._embedding_endpoint(),
            payload,
            self.headers,
            self.timeout,
        )
        data = response
        if "data" in data:
            return _normalize_openai_embeddings(data["data"], len(texts))
        if "embeddings" in data:
            return list(data["embeddings"])
        raise ValueError("Unexpected response format from Nomic embedding service.")

    def get_embedding_dimension(self, probe_text: str = "hello") -> int:
        embedding = self._post_embeddings([probe_text])[0]
        if not embedding:
            raise ValueError("Embedding probe returned an empty vector.")
        return len(embedding)


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


async def _async_post_embeddings(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    response = await asyncio.to_thread(
        _sync_post_embeddings,
        url,
        payload,
        headers,
        timeout,
    )
    return response


def _sync_post_embeddings(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


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
        model_name = model or os.environ.get("SMAK_QWEN_MODEL", _DEFAULT_QWEN_LLM_MODEL)
    elif normalized_provider in {"gpt-oss", "gpt_oss"}:
        base = api_base or os.environ.get(
            "SMAK_GPT_OSS_API_BASE", _DEFAULT_GPT_API_BASE
        )
        model_name = model or os.environ.get("SMAK_GPT_OSS_MODEL", _DEFAULT_GPT_OSS_LLM_MODEL)
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
