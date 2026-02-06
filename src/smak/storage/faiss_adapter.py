"""Faiss storage adapter for SMAK."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def _load_faiss_dependencies() -> tuple[type[Any], type[Any]]:
    candidates = [
        ("src.engine.faiss_engine", "src.core.schema"),
        ("faiss_storage_lib.engine.faiss_engine", "faiss_storage_lib.core.schema"),
    ]
    for engine_path, schema_path in candidates:
        try:
            engine_module = importlib.import_module(engine_path)
            schema_module = importlib.import_module(schema_path)
        except ModuleNotFoundError:
            continue
        engine_cls = getattr(engine_module, "FaissEngine", None)
        doc_cls = getattr(schema_module, "VectorDocument", None)
        if engine_cls and doc_cls:
            return engine_cls, doc_cls
    raise ModuleNotFoundError(
        "Critical dependency 'faiss-storage-lib' not found. "
        "Install faiss-storage-lib to use the Faiss storage adapter."
    )


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
class FaissVectorStore:
    uri: str
    collection_name: str
    dim: int
    _engine: Any = field(init=False)
    _doc_cls: type[Any] = field(init=False)

    def __post_init__(self) -> None:
        engine_cls, doc_cls = _load_faiss_dependencies()
        self._doc_cls = doc_cls
        full_path = Path(self.uri) / self.collection_name
        logger.info("Initializing FaissEngine at %s", full_path)
        self._engine = engine_cls(str(full_path), self.dim)

    def add(self, nodes: Sequence[Any]) -> None:
        docs = []
        for node in nodes:
            uid = _node_id(node)
            vector = _node_value(node, "embedding")
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
