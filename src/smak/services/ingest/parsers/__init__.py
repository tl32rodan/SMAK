"""Parsers for ingest pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from smak.core.domain import KnowledgeUnit
from smak.services.ingest.parsers.perl import PerlParser
from smak.services.ingest.parsers.python import PythonParser


class Parser(Protocol):
    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]: ...


@dataclass
class NullParser:
    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        _ = content, source
        return []


@dataclass
class SimpleLineParser:
    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        normalized_content = (content or "").strip()
        if not normalized_content:
            return []
        origin = str(Path(source).resolve()) if source else "content"
        symbol = "*"
        return [
            KnowledgeUnit(
                uid=f"{origin}::{symbol}",
                content=normalized_content,
                source_type="documentation",
                metadata={"symbol": symbol, "source": source},
            )
        ]


def get_parser_for_path(path: Path, root_path: Path | None = None) -> Parser:
    _ = root_path
    suffix = path.suffix.lower()
    if suffix == ".py":
        return PythonParser(root_path=str(root_path) if root_path else None)
    if suffix in {".pl", ".pm", ".t"}:
        return PerlParser(root_path=str(root_path) if root_path else None)
    if suffix in {".md", ".markdown", ".txt", ".csv", ".il"}:
        return SimpleLineParser()
    return NullParser()


__all__ = [
    "NullParser",
    "Parser",
    "PerlParser",
    "PythonParser",
    "SimpleLineParser",
    "get_parser_for_path",
]
