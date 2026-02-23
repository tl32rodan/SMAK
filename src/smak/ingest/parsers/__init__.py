"""Parsers for ingest pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from smak.core.domain import KnowledgeUnit
from smak.ingest.parsers.issue import IssueParser
from smak.ingest.parsers.perl import PerlParser
from smak.ingest.parsers.python import PythonParser


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
            units.append(
                KnowledgeUnit(
                    uid=f"{origin}:{index}",
                    content=line,
                    source_type="documentation",
                    metadata={"line": index, "source": source},
                )
            )
        return units


def get_parser_for_path(path: Path, root_path: Path | None = None) -> Parser:
    """Return a parser implementation based on a file path."""

    suffix = path.suffix.lower()
    if suffix == ".py":
        return PythonParser(root_path=str(root_path) if root_path else None)
    if suffix in {".pl", ".pm"}:
        return PerlParser(root_path=str(root_path) if root_path else None)
    if suffix in {".md", ".markdown"}:
        return IssueParser()
    return SimpleLineParser()


__all__ = [
    "IssueParser",
    "Parser",
    "PerlParser",
    "PythonParser",
    "SimpleLineParser",
    "get_parser_for_path",
]
