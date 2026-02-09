"""Embedding helpers and internal embedding adapters for SMAK."""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from typing import Any, Mapping, Protocol, Sequence

from smak.config import SmakConfig

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


_DEFAULT_NOMIC_API_BASE = "http://f15dtpai1:11436"
_DEFAULT_NOMIC_MODEL = "nomic_embed_text:latest"


class InternalNomicEmbedding(BaseEmbedding):
    """Embedding adapter for the internal Nomic server."""

    api_base: str
    model: str
    timeout: float
    headers: dict[str, str]
    session: Any | None
    model_name: str
    embed_batch_size: int

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
        self.headers = dict(headers or {})
        self.session = session or (requests.Session() if requests else None)
        self.model_name = resolved_model
        self.embed_batch_size = embed_batch_size

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


class EmbeddingProbe(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def get_text_embeddings(self, texts: list[str]) -> list[list[float]]: ...

    def get_embedding_dimension(self) -> int: ...


def initialize_embedding_dimensions(config: SmakConfig, embedder: EmbeddingProbe) -> SmakConfig:
    dimension = detect_embedding_dimension(embedder)
    if config.embedding_dimensions == dimension:
        return config
    return replace(config, embedding_dimensions=dimension)


def detect_embedding_dimension(embedder: EmbeddingProbe, probe_text: str = "hello") -> int:
    if hasattr(embedder, "get_embedding_dimension"):
        dimension = embedder.get_embedding_dimension()
        return _validate_dimension(dimension)
    vectors = _probe_embeddings(embedder, probe_text)
    return _validate_dimension(len(vectors[0]))


def _probe_embeddings(embedder: EmbeddingProbe, probe_text: str) -> list[list[float]]:
    if hasattr(embedder, "embed_documents"):
        vectors = embedder.embed_documents([probe_text])
    elif hasattr(embedder, "embed"):
        vectors = embedder.embed([probe_text])
    elif hasattr(embedder, "get_text_embeddings"):
        vectors = embedder.get_text_embeddings([probe_text])
    else:
        raise AttributeError("Embedder does not support embedding documents.")
    if not vectors or not vectors[0]:
        raise ValueError("Embedding probe returned an empty vector.")
    return vectors


def _validate_dimension(value: int) -> int:
    if value <= 0:
        raise ValueError("Embedding dimension must be positive.")
    return int(value)


def validate_vector_store_dimension(vector_store: object, expected_dimension: int) -> None:
    actual = _extract_vector_store_dimension(vector_store)
    if actual is None:
        return
    if actual != expected_dimension:
        raise ValueError(
            "Vector store embedding dimension mismatch: "
            f"expected {expected_dimension}, got {actual}."
        )


def _extract_vector_store_dimension(vector_store: object) -> int | None:
    for attr in ("dim", "embedding_dim", "embedding_dimension", "dimension"):
        if hasattr(vector_store, attr):
            return _coerce_dimension(getattr(vector_store, attr))
    collection = getattr(vector_store, "collection", None) or getattr(
        vector_store, "_collection", None
    )
    if collection is not None:
        dimension = _extract_dimension_from_schema(getattr(collection, "schema", None))
        if dimension is not None:
            return dimension
    client = getattr(vector_store, "client", None) or getattr(vector_store, "_client", None)
    collection_name = getattr(vector_store, "collection_name", None) or getattr(
        vector_store, "_collection_name", None
    )
    if client and collection_name and hasattr(client, "describe_collection"):
        info = client.describe_collection(collection_name)
        dimension = _extract_dimension_from_schema(info)
        if dimension is not None:
            return dimension
    return None


def _extract_dimension_from_schema(schema: object) -> int | None:
    if schema is None:
        return None
    if isinstance(schema, Mapping):
        fields = schema.get("fields") or schema.get("schema", {}).get("fields")
    else:
        fields = getattr(schema, "fields", None)
    if not fields:
        return None
    for field in fields:
        field_name = _read_field_value(field, "name")
        params = _read_field_value(field, "params") or {}
        dimension = None
        if isinstance(params, Mapping):
            dimension = params.get("dim") or params.get("dimension")
        if dimension is None:
            dimension = _read_field_value(field, "dim") or _read_field_value(field, "dimension")
        if dimension is not None and field_name in {
            None,
            "embedding",
            "vector",
            "embedding_vector",
        }:
            return _coerce_dimension(dimension)
    return None


def _read_field_value(field: object, name: str) -> Any:
    if isinstance(field, Mapping):
        return field.get(name)
    return getattr(field, name, None)


def _coerce_dimension(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


__all__ = [
    "InternalNomicEmbedding",
    "detect_embedding_dimension",
    "initialize_embedding_dimensions",
    "validate_vector_store_dimension",
]
