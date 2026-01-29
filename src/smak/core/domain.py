"""Domain objects for SMAK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class KnowledgeUnit:
    """A discrete unit of knowledge extracted from a source."""

    uid: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def with_metadata(self, updates: Mapping[str, Any]) -> "KnowledgeUnit":
        """Return a new knowledge unit with merged metadata."""

        merged = {**self.metadata, **dict(updates)}
        return KnowledgeUnit(
            uid=self.uid,
            content=self.content,
            metadata=merged,
            source=self.source,
        )
