"""Native Milvus Lite storage adapter."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


def _load_pymilvus() -> Any:
    spec = importlib.util.find_spec("pymilvus")
    if spec is None:  # pragma: no cover - guard for missing dependency
        raise ModuleNotFoundError(
            "Critical dependency 'pymilvus' not found. "
            "Install pymilvus and milvus-lite to use native Milvus storage."
        )
    return importlib.import_module("pymilvus")


def _node_value(node: Any, attribute: str, fallback: str | None = None) -> Any:
    if hasattr(node, attribute):
        return getattr(node, attribute)
    if fallback and hasattr(node, fallback):
        return getattr(node, fallback)
    return None


def _node_text(node: Any) -> str | None:
    value = _node_value(node, "text")
    if value:
        return value
    getter = getattr(node, "get_text", None)
    if callable(getter):
        return getter()
    return None


def _node_id(node: Any) -> str | None:
    return _node_value(node, "id_", "node_id")


def _node_metadata(node: Any) -> dict[str, Any]:
    metadata = _node_value(node, "metadata") or {}
    if isinstance(metadata, dict):
        return metadata
    return {"metadata": metadata}


@dataclass
class MilvusLiteVectorStore:
    """Minimal Milvus Lite adapter for ingestion + retrieval."""

    uri: str
    collection_name: str
    dim: int
    metric_type: str = "IP"
    vector_field: str = "embedding"
    id_field: str = "id"
    content_field: str = "content"
    metadata_field: str = "metadata"
    client: Any | None = None

    def __post_init__(self) -> None:
        pymilvus = _load_pymilvus()
        client = self.client or pymilvus.MilvusClient(uri=self.uri)
        self.client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dim,
            vector_field_name=self.vector_field,
            primary_field_name=self.id_field,
            auto_id=False,
            enable_dynamic_field=True,
        )

    def add(self, nodes: Sequence[Any]) -> None:
        payloads: list[dict[str, Any]] = []
        for node in nodes:
            uid = _node_id(node)
            vector = _node_value(node, "embedding")
            if uid is None or vector is None:
                continue
            payloads.append(
                {
                    self.id_field: uid,
                    self.vector_field: vector,
                    self.content_field: _node_text(node),
                    self.metadata_field: _node_metadata(node),
                }
            )
        if payloads:
            self.client.insert(self.collection_name, payloads)

    def search(self, embedding: Sequence[float], *, top_k: int = 5) -> list[dict[str, Any]]:
        results = self.client.search(
            self.collection_name,
            data=[list(embedding)],
            limit=top_k,
            output_fields=[self.content_field, self.metadata_field, self.id_field],
        )
        return [_format_hit(hit) for hit in _iter_hits(results)]

    def get_by_id(self, uid: str) -> dict[str, Any] | None:
        results = self.client.get(
            self.collection_name,
            ids=[uid],
            output_fields=[self.content_field, self.metadata_field, self.id_field],
        )
        for hit in _iter_hits(results):
            formatted = _format_hit(hit)
            if formatted.get("uid") == uid:
                return formatted
        return None


@dataclass
class MilvusLiteVectorSearchIndex:
    """VectorSearchIndex implementation for native Milvus Lite."""

    store: MilvusLiteVectorStore
    embedder: Any
    top_k: int = 5

    def search(self, query: str) -> Iterable[dict[str, Any]]:
        embedding = _get_query_embedding(self.embedder, query)
        return self.store.search(embedding, top_k=self.top_k)

    def get_by_id(self, uid: str) -> dict[str, Any] | None:
        return self.store.get_by_id(uid)


def _get_query_embedding(embedder: Any, query: str) -> Sequence[float]:
    if hasattr(embedder, "get_query_embedding"):
        return embedder.get_query_embedding(query)
    if hasattr(embedder, "get_text_embedding"):
        return embedder.get_text_embedding(query)
    raise AttributeError("Embedder does not provide query embedding methods.")


def _iter_hits(results: Any) -> Iterable[Any]:
    if isinstance(results, list):
        if results and isinstance(results[0], list):
            for hit in results[0]:
                yield hit
        else:
            for hit in results:
                yield hit
    else:
        yield from []


def _format_hit(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        uid = hit.get("id") or hit.get("uid")
        entity = hit.get("entity", hit)
    else:
        uid = getattr(hit, "id", None) or getattr(hit, "uid", None)
        entity = getattr(hit, "entity", None) or {}
    metadata = entity.get("metadata") if isinstance(entity, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}
    return {
        "uid": uid,
        "content": entity.get("content") if isinstance(entity, dict) else None,
        "metadata": metadata,
    }


def load_milvus_lite_store(
    *, uri: str, collection_name: str, dim: int
) -> "MilvusLiteVectorStore":
    try:
        return MilvusLiteVectorStore(
            uri=uri,
            collection_name=collection_name,
            dim=dim,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - guard for missing dependency
        raise ModuleNotFoundError(
            "Vector store dependency missing. Install "
            "'pymilvus' with 'milvus-lite' to use Milvus storage."
        ) from exc


__all__ = [
    "MilvusLiteVectorSearchIndex",
    "MilvusLiteVectorStore",
    "load_milvus_lite_store",
]
