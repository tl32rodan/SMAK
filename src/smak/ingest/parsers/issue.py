"""Issue parser for markdown files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from smak.core.domain import KnowledgeUnit
from smak.utils.yaml import load_yaml


@dataclass
class IssueParser:
    """Parse markdown issues with optional frontmatter."""

    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        frontmatter, body = _split_frontmatter(content or "")
        metadata = load_yaml(frontmatter) if frontmatter else {}
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}
        issue_id = metadata.get("id") or metadata.get("title") or source or "issue"
        uid = f"issue::{issue_id}"
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
                relations=tuple(relations),
                metadata={"source": source, **metadata},
            )
        ]


def _split_frontmatter(content: str) -> tuple[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return "", content
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return "", content
