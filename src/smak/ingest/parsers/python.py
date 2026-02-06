"""Python source code parser."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from smak.core.domain import KnowledgeUnit


@dataclass
class PythonParser:
    """Parse Python source code into knowledge units."""

    root_path: str | None = None

    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        tree = ast.parse(content or "")
        rel_source = _relative_source(source, self.root_path)
        units: list[KnowledgeUnit] = []

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.class_stack: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                symbol = ".".join([*self.class_stack, node.name])
                segment = ast.get_source_segment(content, node) or node.name
                units.append(
                    KnowledgeUnit(
                        uid=f"{rel_source}::{symbol}" if rel_source else symbol,
                        content=segment,
                        source_type="source_code",
                        metadata={
                            "language": "python",
                            "symbol": symbol,
                            "source": rel_source,
                            "symbol_type": "class",
                        },
                    )
                )
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

            def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                symbol = ".".join([*self.class_stack, node.name]) if self.class_stack else node.name
                segment = ast.get_source_segment(content, node) or node.name
                units.append(
                    KnowledgeUnit(
                        uid=f"{rel_source}::{symbol}" if rel_source else symbol,
                        content=segment,
                        source_type="source_code",
                        metadata={
                            "language": "python",
                            "symbol": symbol,
                            "source": rel_source,
                            "symbol_type": "method" if self.class_stack else "function",
                            "parent_class": self.class_stack[-1] if self.class_stack else None,
                        },
                    )
                )
                self.generic_visit(node)

        Visitor().visit(tree)
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
