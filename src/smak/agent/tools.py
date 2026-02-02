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


def _require_llamaindex_component(module: str, component: str) -> Any:
    try:
        return __import__(module, fromlist=[component]).__dict__[component]
    except (ModuleNotFoundError, KeyError, AttributeError) as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "LlamaIndex is required for tool integration. "
            "Install 'llama-index-core' to use this feature."
        ) from exc


def _mesh_retrieval_query_engine_cls() -> type:
    base_query_engine = _require_llamaindex_component(
        "llama_index.core.query_engine", "BaseQueryEngine"
    )

    class MeshRetrievalQueryEngine(base_query_engine):
        def __init__(self, tool: "MeshRetrievalTool") -> None:
            super().__init__()
            self._tool = tool

        def _query(self, query: str) -> Any:
            return self._tool.retrieve(query)

    return MeshRetrievalQueryEngine


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


@dataclass
class MeshRetrievalTool:
    """Retrieve documents and expand mesh relations for LlamaIndex tools."""

    registry: IndexRegistry
    index_name: str
    name: str = "mesh_retrieval"
    description: str = (
        "Search the semantic mesh, expand mesh relations by ID, and return context."
    )

    def retrieve(self, query: str) -> dict[str, Any]:
        index = self.registry.get_index(self.index_name)
        results = list(index.search(query))
        context: list[dict[str, Any]] = []
        related_items: list[dict[str, Any]] = []
        for result in results:
            context.append(self._format_context(result, relation_type="primary"))
            for relation in self._extract_relations(result):
                relation_index = self._resolve_relation_index(relation, index)
                related = relation_index.get_by_id(relation)
                if related is None:
                    continue
                related_items.append(related)
                context.append(
                    self._format_context(
                        related, relation_type="related", relation_id=relation
                    )
                )
        return {
            "query": query,
            "matches": results,
            "related": related_items,
            "context": context,
        }

    def as_function_tool(self) -> Any:
        function_tool = _require_llamaindex_component(
            "llama_index.core.tools", "FunctionTool"
        )
        return function_tool.from_defaults(
            fn=self.retrieve,
            name=self.name,
            description=self.description,
        )

    def as_query_engine_tool(self) -> Any:
        query_engine_tool = _require_llamaindex_component(
            "llama_index.core.tools", "QueryEngineTool"
        )
        query_engine_cls = _mesh_retrieval_query_engine_cls()
        return query_engine_tool.from_defaults(
            query_engine=query_engine_cls(tool=self),
            name=self.name,
            description=self.description,
        )

    def _extract_relations(self, result: dict[str, Any]) -> list[str]:
        metadata = result.get("metadata") or {}
        relations = metadata.get("mesh_relations") or []
        if isinstance(relations, str):
            return [relations]
        return list(relations)

    def _resolve_relation_index(
        self, relation: str, default_index: VectorSearchIndex
    ) -> VectorSearchIndex:
        if "::" not in relation:
            return default_index
        prefix = relation.split("::", 1)[0]
        candidates = [prefix]
        if prefix.endswith("s"):
            candidates.append(prefix.rstrip("s"))
        else:
            candidates.append(f"{prefix}s")
        for candidate in candidates:
            try:
                return self.registry.get_index(candidate)
            except KeyError:
                continue
        return default_index

    def _format_context(
        self,
        result: dict[str, Any],
        *,
        relation_type: str,
        relation_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "id": result.get("uid") or result.get("id"),
            "content": self._extract_content(result),
            "metadata": result.get("metadata") or {},
            "relation_type": relation_type,
            "relation_id": relation_id,
        }

    def _extract_content(self, result: dict[str, Any]) -> str | None:
        return (
            result.get("content")
            or result.get("text")
            or result.get("payload", {}).get("content")
            or result.get("metadata", {}).get("content")
        )
