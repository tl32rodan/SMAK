"""Faiss storage adapter for SMAK."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from faiss_storage_lib.core.schema import VectorDocument

from smak.storage.faiss_engine import FaissEngine

logger = logging.getLogger(__name__)


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
    payload = _node_value(node, "payload")
    if isinstance(payload, dict) and isinstance(payload.get("content"), str):
        return payload["content"]
    getter = getattr(node, "get_text", None)
    if callable(getter):
        return getter()
    return None


def _node_id(node: Any) -> str | None:
    uid = _node_value(node, "uid")
    if isinstance(uid, str):
        return uid
    return _node_value(node, "id_", "node_id")


def _node_metadata(node: Any) -> dict[str, Any]:
    metadata = _node_value(node, "metadata")
    if isinstance(metadata, dict):
        return metadata
    payload = _node_value(node, "payload")
    if isinstance(payload, dict):
        payload_metadata = payload.get("metadata")
        if isinstance(payload_metadata, dict):
            return payload_metadata
    return {"metadata": metadata} if metadata is not None else {}


@dataclass
class FaissVectorStore:
    uri: str
    collection_name: str
    dim: int
    _engine: Any = field(init=False)
    _doc_cls: type[Any] = field(init=False)

    def __post_init__(self) -> None:
        self._doc_cls = VectorDocument
        full_path = Path(self.uri) / self.collection_name
        logger.info("Initializing FaissEngine at %s", full_path)
        self._engine = FaissEngine(str(full_path), self.dim)

    def add(self, nodes: Sequence[Any]) -> None:
        docs = []
        for node in nodes:
            uid = _node_id(node)
            vector = _node_value(node, "embedding") or _node_value(node, "vector")
            if uid is None or vector is None:
                continue
            docs.append(
                self._doc_cls(
                    uid=uid,
                    vector=vector,
                    payload={"content": _node_text(node), "metadata": _node_metadata(node)},
                )
            )
        if docs:
            self._engine.add(docs)
            self._engine.persist()

    def delete_by_metadata(self, key: str, value: Any) -> None:
        delete_method = getattr(self._engine, "delete_by_metadata", None)
        if callable(delete_method):
            delete_method(key, value)
            persist = getattr(self._engine, "persist", None)
            if callable(persist):
                persist()
            return
        logger.debug("Vector engine does not support delete_by_metadata; skipping cleanup.")

    def search(self, embedding: Sequence[float], *, top_k: int = 5) -> list[dict[str, Any]]:
        results = self._engine.search(list(embedding), top_k)
        return [
            {
                "uid": result.uid,
                "content": result.payload.get("content"),
                "metadata": result.payload.get("metadata"),
                "score": result.score,
            }
            for result in results
        ]

    def get_by_id(self, uid: str) -> dict[str, Any] | None:
        doc = self._engine.get_by_id(uid)
        if doc:
            return {
                "uid": doc.uid,
                "content": doc.payload.get("content"),
                "metadata": doc.payload.get("metadata"),
            }
        return None

    def count(self) -> int | None:
        count_method = getattr(self._engine, "count", None)
        if callable(count_method):
            return int(count_method())
        docs = getattr(self._engine, "docs", None)
        if isinstance(docs, Sequence):
            return len(docs)
        return None

    def last_update(self) -> str | None:
        value = getattr(self._engine, "last_update", None)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return None


@dataclass
class FaissVectorSearchIndex:
    """VectorSearchIndex implementation for Faiss storage."""

    store: FaissVectorStore
    embedder: Any
    top_k: int = 5

    def search(self, query: str) -> list[dict[str, Any]]:
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


def load_faiss_store(*, uri: str, collection_name: str, dim: int) -> FaissVectorStore:
    return FaissVectorStore(uri=uri, collection_name=collection_name, dim=dim)


__all__ = ["FaissVectorSearchIndex", "FaissVectorStore", "load_faiss_store"]
