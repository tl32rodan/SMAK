from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from smak.config import SmakConfig, StorageConfig


def _install_fake_dependencies() -> None:
    fake_requests = ModuleType("requests")

    class FakeSession:
        def post(self, url: str, json: dict, headers: dict, timeout: float) -> SimpleNamespace:
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"data": [{"index": 0, "embedding": [0.0]}]},
            )

    fake_requests.Session = FakeSession
    fake_requests.post = lambda *args, **kwargs: SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"data": [{"index": 0, "embedding": [0.0]}]},
    )

    fake_embeddings = ModuleType("llama_index.core.embeddings")

    class FakeBaseEmbedding:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

        def get_text_embedding(self, text: str) -> list[float]:
            return [0.0]

        def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    fake_embeddings.BaseEmbedding = FakeBaseEmbedding

    fake_schema = ModuleType("llama_index.core.schema")

    class FakeTextNode:
        def __init__(self, text: str, id_: str, metadata: dict) -> None:
            self.text = text
            self.id_ = id_
            self.metadata = metadata
            self.embedding: list[float] | None = None

    fake_schema.TextNode = FakeTextNode

    fake_openai_like = ModuleType("llama_index.llms.openai_like")

    class FakeOpenAILike:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    fake_openai_like.OpenAILike = FakeOpenAILike

    fake_core = ModuleType("llama_index.core")
    fake_core.embeddings = fake_embeddings
    fake_core.schema = fake_schema

    fake_llms = ModuleType("llama_index.llms")
    fake_llms.openai_like = fake_openai_like

    fake_root = ModuleType("llama_index")
    fake_root.core = fake_core
    fake_root.llms = fake_llms

    sys.modules.update(
        {
            "requests": fake_requests,
            "llama_index": fake_root,
            "llama_index.core": fake_core,
            "llama_index.core.embeddings": fake_embeddings,
            "llama_index.core.schema": fake_schema,
            "llama_index.llms": fake_llms,
            "llama_index.llms.openai_like": fake_openai_like,
        }
    )


_install_fake_dependencies()


def _load_cli():
    return importlib.import_module("smak.cli")


class FakeNode:
    def __init__(self, text: str, id_: str, metadata: dict) -> None:
        self.text = text
        self.id_ = id_
        self.metadata = metadata
        self.embedding: list[float] | None = None


class FakeVectorStore:
    def __init__(self, saved: list, index_name: str) -> None:
        self._saved = saved
        self.index_name = index_name

    def add(self, nodes: list) -> None:
        self._saved.extend(nodes)

    def delete_by_metadata(self, key: str, value: str) -> None:
        return None

    def get_by_id(self, uid: str) -> dict | None:
        for node in self._saved:
            if node.id_ == uid:
                return {"uid": uid, "metadata": node.metadata}
        return None


class TestCli(unittest.TestCase):
    class DummyEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 1.0, 2.0] for text in texts]

    def test_default_config_template_includes_storage(self) -> None:
        cli = _load_cli()
        template = cli._default_config_template()

        self.assertIn("storage:", template)
        self.assertIn("provider: faiss", template)
        self.assertIn("uri: ./smak_data", template)

    def test_ingest_folder_processes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            source = folder / "example.py"
            source.write_text("def foo():\n    return 1\n", encoding="utf-8")

            saved: list = []
            config = SmakConfig(storage=StorageConfig(uri="vault.db"))

            cli = _load_cli()
            stats = cli._ingest_folder(
                folder,
                "code",
                config,
                vector_store_loader=lambda index_name, cfg: FakeVectorStore(saved, index_name),
                node_class_loader=lambda: FakeNode,
                embedder_loader=self.DummyEmbedder,
                incremental=False,
            )

            self.assertEqual(stats.files, 1)
            self.assertEqual(stats.vectors, 1)
            self.assertEqual(stats.skipped, 0)
            self.assertIn("source_mtime", saved[0].metadata)

    def test_ingest_folder_skips_unchanged_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            source = folder / "example.py"
            source.write_text("def foo():\n    return 1\n", encoding="utf-8")

            saved: list = []
            cli = _load_cli()
            config = SmakConfig(storage=StorageConfig(uri="vault.db"))

            cli._ingest_folder(
                folder,
                "code",
                config,
                vector_store_loader=lambda index_name, cfg: FakeVectorStore(saved, index_name),
                node_class_loader=lambda: FakeNode,
                embedder_loader=self.DummyEmbedder,
                incremental=False,
            )
            stats = cli._ingest_folder(
                folder,
                "code",
                config,
                vector_store_loader=lambda index_name, cfg: FakeVectorStore(saved, index_name),
                node_class_loader=lambda: FakeNode,
                embedder_loader=self.DummyEmbedder,
                incremental=True,
            )

            self.assertEqual(stats.files, 0)
            self.assertEqual(stats.skipped, 1)

    def test_search_json_output(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "example.py"
            source.write_text("def hello():\n    return True\n", encoding="utf-8")
            cli = _load_cli()

            result = runner.invoke(cli.main, ["search", str(source), "--json-output"])

            self.assertEqual(result.exit_code, 0)
            payload = json.loads(result.output)
            self.assertEqual(len(payload), 1)
            self.assertIn("::hello", payload[0])

    def test_sidecar_update_merges_metadata(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "example.py"
            source.write_text("def hello():\n    return True\n", encoding="utf-8")
            cli = _load_cli()
            updates = json.dumps(
                [{"symbol": "example.py::hello", "intent": "greeting", "relations": ["issue:1"]}]
            )

            result = runner.invoke(
                cli.main,
                ["sidecar", "update", str(source), "--updates", updates],
            )

            self.assertEqual(result.exit_code, 0)
            sidecar = Path(tmp_dir) / "example.py.sidecar.yaml"
            self.assertTrue(sidecar.exists())
            payload = sidecar.read_text(encoding="utf-8")
            self.assertIn("greeting", payload)

    def test_query_command_outputs_json(self) -> None:
        runner = CliRunner()

        class QueryEmbedder(SimpleNamespace):
            def get_text_embedding(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

        class QueryStore(SimpleNamespace):
            def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
                return [{"uid": "x", "score": 0.8}]

        with patch("smak.cli.InternalNomicEmbedding", new=QueryEmbedder), patch(
            "smak.cli._load_vector_store_for_cli",
            new=lambda index, config: (
                SmakConfig(storage=StorageConfig(uri="memory")),
                QueryStore(),
            ),
        ):
            cli = _load_cli()
            result = runner.invoke(cli.main, ["query", "hello", "--index", "code"])

        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertEqual(payload[0]["uid"], "x")


if __name__ == "__main__":
    unittest.main()
