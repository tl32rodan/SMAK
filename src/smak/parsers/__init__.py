"""Source code parsers for extracting knowledge units."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from smak.core.domain import KnowledgeUnit
from smak.parsers.perl import PerlParser
from smak.parsers.python import PythonParser


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
        origin = str(Path(source).absolute()) if source else "content"
        symbol = "*"
        return [
            KnowledgeUnit(
                uid=f"{origin}::{symbol}",
                content=normalized_content,
                source_type="documentation",
                metadata={"symbol": symbol, "source": source},
            )
        ]


def get_parser_for_path(path: Path) -> Parser:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return PythonParser()
    if suffix in {".pl", ".pm", ".t"}:
        return PerlParser()
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
