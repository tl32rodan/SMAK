"""Domain objects for SMAK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class KnowledgeUnit:
    """A discrete unit of knowledge extracted from a source."""

    uid: str
    content: str
    source_type: str
    relations: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, updates: Mapping[str, Any]) -> "KnowledgeUnit":
        """Return a new knowledge unit with merged metadata."""

        merged = {**self.metadata, **dict(updates)}
        return KnowledgeUnit(
            uid=self.uid,
            content=self.content,
            source_type=self.source_type,
            relations=self.relations,
            metadata=merged,
        )

    def with_relations(self, relations: Sequence[str]) -> "KnowledgeUnit":
        """Return a new knowledge unit with updated relations."""

        return KnowledgeUnit(
            uid=self.uid,
            content=self.content,
            source_type=self.source_type,
            relations=tuple(relations),
            metadata=self.metadata,
        )
