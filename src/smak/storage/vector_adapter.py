"""Shared vector adapter interfaces and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from faiss_storage_lib.core.schema import VectorDocument

from smak.core.domain import KnowledgeUnit


class VectorIndex(Protocol):
    def add(self, docs: Sequence[VectorDocument]) -> None: ...

    def delete_by_ids(self, uids: list[str]) -> None: ...

    def get_all_tracked_sources(self) -> dict[str, list[str]]: ...


class FaissRegistry(Protocol):
    def get_index(self, name: str) -> VectorIndex: ...


class EmbedderService(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class VectorAdapter:
    registry: FaissRegistry
    embedder: EmbedderService

    def save(self, index_name: str, units: Sequence[KnowledgeUnit]) -> None:
        vectors = self.embedder.embed_documents([unit.content for unit in units])
        docs = [
            VectorDocument(
                uid=unit.uid,
                vector=vector,
                payload={
                    "content": unit.content,
                    "relations": list(unit.relations),
                    "meta": unit.metadata,
                },
            )
            for unit, vector in zip(units, vectors)
        ]
        self.registry.get_index(index_name).add(docs)


__all__ = [
    "EmbedderService",
    "FaissRegistry",
    "VectorAdapter",
    "VectorDocument",
    "VectorIndex",
]
