"""Python source code parser."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from smak.core.domain import KnowledgeUnit


@dataclass
class PythonParser:
    """Parse Python source code into knowledge units."""

    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        tree = ast.parse(content or "")
        origin = source or "python"
        units: list[KnowledgeUnit] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                segment = ast.get_source_segment(content, node) or name
                units.append(
                    KnowledgeUnit(
                        uid=f"{origin}::{name}",
                        content=segment,
                        source_type="source_code",
                        metadata={
                            "language": "python",
                            "symbol": name,
                            "lineno": getattr(node, "lineno", None),
                            "source": source,
                        },
                    )
                )
        return units
