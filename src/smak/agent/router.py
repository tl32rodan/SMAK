"""Routing utilities for agent tools and indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from smak.agent.tools import Tool, ToolRegistry


@dataclass
class ToolRouter:
    """Route tool names to tool instances."""

    registry: ToolRegistry
    default_tool: Tool | None = None

    def route(self, tool_name: str) -> Tool:
        tool = self.registry.get(tool_name)
        if tool is not None:
            return tool
        if self.default_tool is None:
            raise KeyError(f"Unknown tool: {tool_name}")
        return self.default_tool


@dataclass
class IndexRouter:
    """Route queries to relevant indices."""

    index_names: Sequence[str]
    classifier: Callable[[str, Sequence[str]], Sequence[str]]

    def route(self, query: str) -> list[str]:
        selected = list(self.classifier(query, self.index_names))
        return selected or list(self.index_names)
