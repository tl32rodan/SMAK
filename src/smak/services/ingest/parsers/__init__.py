"""Parsers for ingest pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from smak.core.domain import KnowledgeUnit
from smak.services.ingest.parsers.issue import IssueParser
from smak.services.ingest.parsers.perl import PerlParser
from smak.services.ingest.parsers.python import PythonParser


class Parser(Protocol):
    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]: ...


@dataclass
class SimpleLineParser:
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
