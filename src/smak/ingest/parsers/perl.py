"""Perl source code parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from smak.core.domain import KnowledgeUnit

_SUB_REGEX = re.compile(r"^\s*sub\s+(\w+)", re.MULTILINE)


@dataclass
class PerlParser:
    """Parse Perl source code into knowledge units."""

    root_path: str | None = None

    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        rel_source = _relative_source(source, self.root_path)
        units: list[KnowledgeUnit] = []
        for line in (content or "").splitlines():
            match = _SUB_REGEX.match(line)
            if not match:
                continue
            name = match.group(1)
            units.append(
                KnowledgeUnit(
                    uid=f"{rel_source}::{name}" if rel_source else name,
                    content=line.strip(),
                    source_type="source_code",
                    metadata={"language": "perl", "symbol": name, "source": rel_source},
                )
            )
        return units


def _relative_source(source: str | None, root_path: str | None) -> str | None:
    if source is None:
        return None
    if root_path is None:
        return source
    try:
        return Path(source).resolve().relative_to(Path(root_path).resolve()).as_posix()
    except ValueError:
        return Path(source).as_posix()
