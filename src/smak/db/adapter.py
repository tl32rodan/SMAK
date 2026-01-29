"""Database adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from smak.core.domain import KnowledgeUnit


class DatabaseAdapter(Protocol):
    """Protocol for database adapters."""

    def save_units(self, units: list[KnowledgeUnit]) -> None:
        """Persist knowledge units."""

    def load_units(self) -> list[KnowledgeUnit]:
        """Load all knowledge units."""


@dataclass(frozen=True)
class VectorDocument:
    """Flattened vector payload for storage."""

    uid: str
    vector: Sequence[float]
    payload: dict[str, Any]


class VectorIndex(Protocol):
    """Protocol for a vector index."""

    def add(self, docs: Sequence[VectorDocument]) -> None:
        """Add vector documents to the index."""


class FaissRegistry(Protocol):
    """Protocol for index registry."""

    def get_index(self, name: str) -> VectorIndex:
        """Return a vector index by name."""


class EmbedderService(Protocol):
    """Protocol for embedder service."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents."""


@dataclass
class VectorAdapter:
    """Adapter to persist knowledge units into vector storage."""

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


@dataclass
class InMemoryAdapter:
    """In-memory database adapter for testing."""

    _units: list[KnowledgeUnit] = field(default_factory=list)

    def save_units(self, units: list[KnowledgeUnit]) -> None:
        self._units.extend(units)

    def load_units(self) -> list[KnowledgeUnit]:
        return list(self._units)
