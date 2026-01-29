"""Database adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from smak.core.domain import KnowledgeUnit


class DatabaseAdapter(Protocol):
    """Protocol for database adapters."""

    def save_units(self, units: list[KnowledgeUnit]) -> None:
        """Persist knowledge units."""

    def load_units(self) -> list[KnowledgeUnit]:
        """Load all knowledge units."""


@dataclass
class InMemoryAdapter:
    """In-memory database adapter for testing."""

    _units: list[KnowledgeUnit] = field(default_factory=list)

    def save_units(self, units: list[KnowledgeUnit]) -> None:
        self._units.extend(units)

    def load_units(self) -> list[KnowledgeUnit]:
        return list(self._units)
