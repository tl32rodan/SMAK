"""Python source code parser."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from smak.core.domain import KnowledgeUnit


@dataclass
class PythonParser:

    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        tree = ast.parse(content or "")
        abs_source = str(Path(source).resolve()) if source else None
        units: list[KnowledgeUnit] = []

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.class_stack: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                symbol = ".".join([*self.class_stack, node.name])
                segment = ast.get_source_segment(content, node) or node.name
                units.append(
                    KnowledgeUnit(
                        uid=f"{abs_source}::{symbol}" if abs_source else symbol,
                        content=segment,
                        source_type="source_code",
                        metadata={
                            "language": "python",
                            "symbol": symbol,
                            "source": abs_source,
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
                        uid=f"{abs_source}::{symbol}" if abs_source else symbol,
                        content=segment,
                        source_type="source_code",
                        metadata={
                            "language": "python",
                            "symbol": symbol,
                            "source": abs_source,
                            "symbol_type": "method" if self.class_stack else "function",
                            "parent_class": self.class_stack[-1] if self.class_stack else None,
                        },
                    )
                )
                self.generic_visit(node)

        Visitor().visit(tree)
        return units
