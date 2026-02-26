"""Issue parser for markdown files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from smak.core.domain import KnowledgeUnit
from smak.utils.yaml import safe_load


@dataclass
class IssueParser:
    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        frontmatter, body = _split_frontmatter(content or "")
        metadata = safe_load(frontmatter) if frontmatter else {}
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}

        uid = _resolve_uid(metadata, body, source)

        relations = metadata.get("relations", [])
        if isinstance(relations, str):
            relations = [relations]
        if not isinstance(relations, Sequence):
            relations = [str(relations)]
        return [
            KnowledgeUnit(
                uid=uid,
                content=body.strip() or (content or "").strip(),
                source_type="issue",
                relations=tuple(str(item) for item in relations),
                metadata={"source": source, **metadata},
            )
        ]


def _resolve_uid(metadata: dict[str, object], body: str, source: str | None) -> str:
    symbol = metadata.get("symbol")
    if isinstance(symbol, str) and symbol.strip():
        return symbol.strip()
    header = _first_header(body)
    if header:
        return _slugify(header)
    if source:
        return _slugify(Path(source).stem)
    return "issue"


def _first_header(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "issue"


def _split_frontmatter(content: str) -> tuple[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", content
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return "", content
