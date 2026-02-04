from __future__ import annotations

import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from smak.config import IndexConfig, LLMConfig, SmakConfig, StorageConfig
from smak.server import (
    LlamaIndexVectorSearchIndex,
    Registry,
    _node_to_payload,
    build_gradio_ui,
    build_index_registry,
    launch_server,
)


class FakeNode:
    def __init__(self, uid: str, text: str, metadata: dict) -> None:
        self.id_ = uid
        self.text = text
        self.metadata = metadata


class FakeNodeWithScore:
    def __init__(self, node: FakeNode) -> None:
        self.node = node


class FakeRetriever:
    def __init__(self, nodes: list[FakeNodeWithScore]) -> None:
        self._nodes = nodes

    def retrieve(self, query: str) -> list[FakeNodeWithScore]:
        return self._nodes


class FakeIndex:
    def __init__(self, nodes: list[FakeNodeWithScore]) -> None:
        self._nodes = nodes
        self.docstore = SimpleNamespace(get_document=lambda uid: FakeNode(uid, "doc", {}))

    def as_retriever(self, similarity_top_k: int) -> FakeRetriever:
        return FakeRetriever(self._nodes)


class FakeVectorStore:
    def __init__(self, nodes: list[FakeNode]) -> None:
        self._nodes = nodes

    def get_nodes(self, ids: list[str]) -> list[FakeNode]:
        return [node for node in self._nodes if node.id_ in ids]


class FakeGradio(ModuleType):
    class Blocks:
        def __init__(self) -> None:
            self.launch_args = None

        def __enter__(self) -> "FakeGradio.Blocks":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def launch(self, **kwargs: object) -> None:
            self.launch_args = kwargs

    class Markdown:
        def __init__(self, text: str) -> None:
            self.text = text

    class ChatInterface:
        def __init__(self, fn) -> None:
            self.fn = fn


class TestServer(unittest.TestCase):
    def test_node_to_payload_normalizes_relations(self) -> None:
        node = FakeNode("unit-1", "hello", {"meta": {"mesh_relations": ["issue:1"]}})
        payload = _node_to_payload(node)

        self.assertEqual(payload["uid"], "unit-1")
        self.assertEqual(payload["content"], "hello")
        self.assertEqual(payload["metadata"]["mesh_relations"], ["issue:1"])

    def test_vector_search_index_search_and_get(self) -> None:
        node = FakeNode("unit-2", "payload", {"relations": ["doc::2"]})
        vector_store = FakeVectorStore([node])
        index = LlamaIndexVectorSearchIndex(
            vector_store,
            index_builder=lambda _: FakeIndex([FakeNodeWithScore(node)]),
        )

        results = list(index.search("query"))
        self.assertEqual(results[0]["uid"], "unit-2")
        self.assertEqual(results[0]["metadata"]["mesh_relations"], ["doc::2"])
        self.assertEqual(index.get_by_id("unit-2")["uid"], "unit-2")

    def test_build_index_registry_uses_config(self) -> None:
        config = SmakConfig(
            indices=[IndexConfig(name="code", description="Code")],
            storage=StorageConfig(uri="memory.db"),
            embedding_dimensions=3,
        )
        calls: list[str] = []

        def loader(name: str, cfg: SmakConfig) -> FakeVectorStore:
            calls.append(name)
            return FakeVectorStore([])

        registry = build_index_registry(
            config,
            vector_store_loader=loader,
            index_builder=lambda _: FakeIndex([]),
        )
        self.assertIsInstance(registry, Registry)
        self.assertEqual(calls, ["code"])
        self.assertIsNotNone(registry.get_index("code"))

    def test_build_index_registry_validates_dimensions(self) -> None:
        config = SmakConfig(
            indices=[IndexConfig(name="code", description="Code")],
            embedding_dimensions=3,
        )

        class FakeDimStore(FakeVectorStore):
            def __init__(self, nodes: list[FakeNode], dim: int) -> None:
                super().__init__(nodes)
                self.dim = dim

        def loader(name: str, cfg: SmakConfig) -> FakeVectorStore:
            return FakeDimStore([], dim=5)

        with self.assertRaises(ValueError):
            build_index_registry(
                config,
                vector_store_loader=loader,
                index_builder=lambda _: FakeIndex([]),
            )

    def test_build_gradio_ui_requires_gradio(self) -> None:
        sys.modules.pop("gradio", None)
        with patch("smak.server.importlib.util.find_spec", return_value=None):
            with self.assertRaises(ModuleNotFoundError):
                build_gradio_ui(agent=lambda _: "ok", mesh_tool=SimpleNamespace())

    def test_launch_server_wires_gradio(self) -> None:
        fake_gradio = FakeGradio("gradio")
        sys.modules["gradio"] = fake_gradio

        config = SmakConfig(
            indices=[IndexConfig(name="code", description="Code")],
            llm=LLMConfig(provider="openai"),
        )

        def fake_agent_builder(mesh_tool, llm) -> object:
            return lambda query: f"ok:{query}"

        class DummyEmbedder:
            def get_embedding_dimension(self) -> int:
                return 3

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "smak.server.load_config", return_value=config
        ), patch("smak.server.importlib.util.find_spec", return_value=object()), patch(
            "smak.server.InternalNomicEmbedding", return_value=DummyEmbedder()
        ):
            app = launch_server(
                port=5555,
                config_path=f"{tmp_dir}/workspace_config.yaml",
                vector_store_loader=lambda name, cfg: FakeVectorStore([]),
                index_builder=lambda _: FakeIndex([]),
                agent_builder=fake_agent_builder,
                llm_loader=lambda cfg: None,
                gradio_factory=build_gradio_ui,
            )

        self.assertIsInstance(app, fake_gradio.Blocks)
        self.assertEqual(app.launch_args["server_port"], 5555)


if __name__ == "__main__":
    unittest.main()
