"""Tooling primitives for SMAK agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence


@dataclass(frozen=True)
class Tool:
    """A callable tool that an agent can invoke."""

    name: str
    description: str
    handler: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.handler(*args, **kwargs)


class ToolRegistry:
    """Registry for tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())


class VectorSearchIndex(Protocol):
    """Protocol for vector search operations."""

    def search(self, query: str) -> Sequence[dict[str, Any]]:
        """Search the index for relevant documents."""

    def get_by_id(self, uid: str) -> dict[str, Any] | None:
        """Fetch a document by unique identifier."""


class IndexRegistry(Protocol):
    """Protocol for index registry."""

    def get_index(self, name: str) -> VectorSearchIndex:
        """Return the named index."""


@dataclass
class MeshSearchTool:
    """Search tool that expands relations in the semantic mesh."""

    registry: IndexRegistry

    def search(self, index_name: str, query: str) -> list[dict[str, Any]]:
        index = self.registry.get_index(index_name)
        results = list(index.search(query))
        expanded: list[dict[str, Any]] = []
        for result in results:
            expanded.append(result)
            payload = result.get("payload", {})
            relations = payload.get("relations", [])
            if not relations:
                relations = payload.get("meta", {}).get("relations", [])
            for relation in relations:
                related = self._lookup_relation(relation)
                if related:
                    expanded.append(related)
        return expanded

    def _lookup_relation(self, relation: str) -> dict[str, Any] | None:
        if "::" not in relation:
            return None
        index_name, _ = relation.split("::", 1)
        candidates = [index_name]
        if index_name.endswith("s"):
            candidates.append(index_name.rstrip("s"))
        else:
            candidates.append(f"{index_name}s")
        for candidate in candidates:
            try:
                index = self.registry.get_index(candidate)
            except KeyError:
                continue
            related = index.get_by_id(relation)
            if related:
                return related
        return None
