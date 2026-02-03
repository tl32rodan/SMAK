"""Agent + Gradio server orchestration for SMAK."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from smak.agent.react import build_llamaindex_react_agent
from smak.agent.tools import IndexRegistry, MeshRetrievalTool, VectorSearchIndex
from smak.bridge.models import InternalNomicEmbedding, build_internal_llm
from smak.config import SmakConfig, load_config
from smak.storage.milvus import (
    MilvusLiteVectorSearchIndex,
    MilvusLiteVectorStore,
    load_milvus_lite_store,
)


def _load_vector_store(index_name: str, config: SmakConfig) -> object:
    return load_milvus_lite_store(
        uri=config.storage.uri,
        collection_name=index_name,
        dim=config.embedding_dimensions,
    )


def _build_vector_index(vector_store: object) -> object:
    spec = importlib.util.find_spec("llama_index.core")
    if spec is None:  # pragma: no cover - guard for missing dependency
        raise ModuleNotFoundError(
            "Critical dependency 'llama-index-core' not found. "
            "Did you run pip install llama-index-core?"
        )
    module = importlib.import_module("llama_index.core")
    index_cls = getattr(module, "VectorStoreIndex")
    return index_cls.from_vector_store(vector_store)


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if isinstance(metadata, dict):
        normalized.update(metadata)
    meta = normalized.get("meta", {}) if isinstance(normalized.get("meta", {}), dict) else {}
    if "mesh_relations" not in normalized:
        if isinstance(meta, dict) and meta.get("mesh_relations"):
            normalized["mesh_relations"] = meta["mesh_relations"]
        elif normalized.get("relations"):
            normalized["mesh_relations"] = normalized["relations"]
        elif isinstance(meta, dict) and meta.get("relations"):
            normalized["mesh_relations"] = meta["relations"]
    return normalized


def _node_to_payload(node: Any) -> dict[str, Any]:
    if hasattr(node, "node"):
        node = node.node
    metadata = _normalize_metadata(getattr(node, "metadata", {}))
    return {
        "uid": getattr(node, "id_", None) or getattr(node, "node_id", None),
        "content": getattr(node, "text", None) or getattr(node, "get_text", lambda: None)(),
        "metadata": metadata,
    }


@dataclass
class LlamaIndexVectorSearchIndex:
    """Adapter that exposes LlamaIndex vector stores via VectorSearchIndex."""

    vector_store: object
    index_builder: Callable[[object], object] = _build_vector_index
    top_k: int = 5

    def __post_init__(self) -> None:
        self._index = self.index_builder(self.vector_store)

    def search(self, query: str) -> Iterable[dict[str, Any]]:
        retriever = self._index.as_retriever(similarity_top_k=self.top_k)
        results = retriever.retrieve(query)
        return [_node_to_payload(result) for result in results]

    def get_by_id(self, uid: str) -> dict[str, Any] | None:
        node = None
        if hasattr(self.vector_store, "get_nodes"):
            nodes = self.vector_store.get_nodes([uid])
            if nodes:
                node = nodes[0]
        elif hasattr(self.vector_store, "get_by_id"):
            node = self.vector_store.get_by_id(uid)
        if node is None and hasattr(self._index, "docstore"):
            docstore = getattr(self._index, "docstore", None)
            if docstore and hasattr(docstore, "get_document"):
                try:
                    node = docstore.get_document(uid)
                except KeyError:
                    node = None
        if node is None:
            return None
        return _node_to_payload(node)


@dataclass(frozen=True)
class Registry(IndexRegistry):
    """Simple index registry."""

    indices: dict[str, VectorSearchIndex]

    def get_index(self, name: str) -> VectorSearchIndex:
        return self.indices[name]


def build_index_registry(
    config: SmakConfig,
    *,
    vector_store_loader: Callable[[str, SmakConfig], object] | None = None,
    index_builder: Callable[[object], object] | None = None,
) -> Registry:
    loader = vector_store_loader or _load_vector_store
    builder = index_builder or _build_vector_index
    names = [entry.name for entry in config.indices] or ["source_code"]
    indices: dict[str, VectorSearchIndex] = {}
    for name in names:
        store = loader(name, config)
        if isinstance(store, MilvusLiteVectorStore):
            indices[name] = MilvusLiteVectorSearchIndex(
                store=store, embedder=InternalNomicEmbedding()
            )
        else:
            indices[name] = LlamaIndexVectorSearchIndex(store, index_builder=builder)
    return Registry(indices)


def _build_llm(config: SmakConfig) -> Any:
    provider = (config.llm.provider or "openai").lower()
    if provider in {"qwen", "gpt-oss", "gpt_oss"}:
        return build_internal_llm(
            provider=provider,
            model=config.llm.model,
            api_base=config.llm.api_base,
            temperature=config.llm.temperature,
        )
    module = importlib.import_module("llama_index.llms.openai_like")
    llm_cls = getattr(module, "OpenAILike")
    return llm_cls(
        model=config.llm.model,
        api_base=config.llm.api_base,
        api_key=None,
        temperature=config.llm.temperature,
    )


def _invoke_agent(agent: Any, query: str, mesh_tool: MeshRetrievalTool) -> Any:
    if hasattr(agent, "chat"):
        response = agent.chat(query)
        return getattr(response, "response", response)
    if hasattr(agent, "query"):
        return agent.query(query)
    if callable(agent):
        return agent(query)
    return mesh_tool.retrieve(query)


def build_gradio_ui(agent: Any, mesh_tool: MeshRetrievalTool) -> Any:
    spec = importlib.util.find_spec("gradio")
    if spec is None:  # pragma: no cover - guard for missing dependency
        raise ModuleNotFoundError(
            "Critical dependency 'gradio' not found. Did you run pip install gradio?"
        )
    gradio = importlib.import_module("gradio")

    def respond(message: str, history: list[dict[str, Any]]) -> str:
        result = _invoke_agent(agent, message, mesh_tool)
        return str(result)

    with gradio.Blocks() as app:
        gradio.Markdown("# SMAK Agent Server")
        gradio.ChatInterface(fn=respond)
    return app


def launch_server(
    *,
    port: int = 7860,
    config_path: str = "workspace_config.yaml",
    vector_store_loader: Callable[[str, SmakConfig], object] | None = None,
    index_builder: Callable[[object], object] | None = None,
    agent_builder: Callable[..., Any] | None = None,
    llm_loader: Callable[[SmakConfig], Any] | None = None,
    gradio_factory: Callable[[Any, MeshRetrievalTool], Any] | None = None,
) -> Any:
    config = load_config(config_path)
    registry = build_index_registry(
        config, vector_store_loader=vector_store_loader, index_builder=index_builder
    )
    primary_index = next(iter(registry.indices.keys()))
    mesh_tool = MeshRetrievalTool(registry, index_name=primary_index)
    llm = llm_loader(config) if llm_loader else _build_llm(config)
    builder = agent_builder or build_llamaindex_react_agent
    agent = builder(mesh_tool, llm=llm)
    ui_factory = gradio_factory or build_gradio_ui
    app = ui_factory(agent, mesh_tool)
    app.launch(server_name="0.0.0.0", server_port=port, share=False)
    return app


__all__ = [
    "LlamaIndexVectorSearchIndex",
    "Registry",
    "build_gradio_ui",
    "build_index_registry",
    "launch_server",
]
