"""Parsers for ingest pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from smak.core.domain import KnowledgeUnit


class Parser(Protocol):
    """Protocol for content parsers."""

    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        """Parse content into knowledge units."""


@dataclass
class SimpleLineParser:
    """Split content into knowledge units per non-empty line."""

    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        units = []
        origin = source or "content"
        for index, line in enumerate(lines, start=1):
            units.append(KnowledgeUnit(uid=f"{origin}:{index}", content=line, source=source))
        return units
