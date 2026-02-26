"""Database adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from smak.core.domain import KnowledgeUnit
from smak.storage.vector_adapter import VectorAdapter, VectorDocument


class DatabaseAdapter(Protocol):
    def save_units(self, units: list[KnowledgeUnit]) -> None: ...

    def load_units(self) -> list[KnowledgeUnit]: ...


@dataclass
class InMemoryAdapter:
    _units: list[KnowledgeUnit] = field(default_factory=list)

    def save_units(self, units: list[KnowledgeUnit]) -> None:
        self._units.extend(units)

    def load_units(self) -> list[KnowledgeUnit]:
        return list(self._units)


__all__ = ["DatabaseAdapter", "InMemoryAdapter", "VectorAdapter", "VectorDocument"]
