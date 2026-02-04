"""Perl source code parser."""

from __future__ import annotations

import re
from dataclasses import dataclass

from smak.core.domain import KnowledgeUnit

_SUB_REGEX = re.compile(r"^\s*sub\s+(\w+)", re.MULTILINE)


@dataclass
class PerlParser:
    """Parse Perl source code into knowledge units."""

    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        origin = f"perl:{source}" if source else "perl"
        units: list[KnowledgeUnit] = []
        for line in (content or "").splitlines():
            match = _SUB_REGEX.match(line)
            if not match:
                continue
            name = match.group(1)
            units.append(
                KnowledgeUnit(
                    uid=f"{origin}::{name}",
                    content=line.strip(),
                    source_type="source_code",
                    metadata={"language": "perl", "symbol": name, "source": source},
                )
            )
        return units
