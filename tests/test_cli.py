from __future__ import annotations

import importlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from smak.config import IndexConfig, SmakConfig


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
            node_id = getattr(node, "id_", getattr(node, "uid", None))
            if node_id != uid:
                continue
            metadata = getattr(node, "metadata", None)
            if metadata is None and isinstance(getattr(node, "payload", None), dict):
                metadata = node.payload.get("metadata")
            return {"uid": uid, "metadata": metadata}
        return None




class ThreadAwareVectorStore(FakeVectorStore):
    def __init__(self, saved: list, index_name: str, main_thread_id: int) -> None:
        super().__init__(saved, index_name)
        self.main_thread_id = main_thread_id
        self.add_thread_ids: list[int] = []
        self.delete_thread_ids: list[int] = []

    def add(self, nodes: list) -> None:
        self.add_thread_ids.append(threading.get_ident())
        super().add(nodes)

    def delete_by_metadata(self, key: str, value: str) -> None:
        self.delete_thread_ids.append(threading.get_ident())
        super().delete_by_metadata(key, value)


class TestCli(unittest.TestCase):
    class DummyEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 1.0, 2.0] for text in texts]

    def test_default_config_template_includes_storage(self) -> None:
        cli = _load_cli()
        template = cli._default_config_template()

        self.assertNotIn("storage:", template)
        self.assertIn("indices:", template)
        self.assertIn("# optional uri override", template)

    def test_ingest_folder_processes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            source = folder / "example.py"
            source.write_text("def foo():\n    return 1\n", encoding="utf-8")

            saved: list = []
            config = SmakConfig(
                indices=[IndexConfig(name="code", description="Code", uri="vault.db")]
            )

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
            self.assertIn("source_mtime", saved[0].payload["metadata"])

    def test_ingest_folder_skips_unchanged_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            source = folder / "example.py"
            source.write_text("def foo():\n    return 1\n", encoding="utf-8")

            saved: list = []
            cli = _load_cli()
            config = SmakConfig(
                indices=[IndexConfig(name="code", description="Code", uri="vault.db")]
            )

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


    def test_ingest_folder_writes_storage_on_main_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            source = folder / "example.py"
            source.write_text("def foo():\n    return 1\n", encoding="utf-8")

            saved: list = []
            cli = _load_cli()
            config = SmakConfig(
                indices=[IndexConfig(name="code", description="Code", uri="vault.db")]
            )
            main_thread_id = threading.get_ident()
            store_holder: dict[str, ThreadAwareVectorStore] = {}

            def load_store(index_name: str, cfg: SmakConfig) -> ThreadAwareVectorStore:
                store = ThreadAwareVectorStore(saved, index_name, main_thread_id)
                store_holder["store"] = store
                return store

            stats = cli._ingest_folder(
                folder,
                "code",
                config,
                vector_store_loader=load_store,
                embedder_loader=self.DummyEmbedder,
                incremental=False,
            )

            store = store_holder["store"]
            self.assertEqual(stats.files, 1)
            self.assertEqual(stats.vectors, 1)
            self.assertTrue(store.add_thread_ids)
            self.assertTrue(store.delete_thread_ids)
            self.assertEqual(store.add_thread_ids, [main_thread_id])
            self.assertEqual(store.delete_thread_ids, [main_thread_id])

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


    def test_load_vector_store_for_cli_uses_index_uri_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "workspace.yaml"
            config_path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code\n"
                "    uri: ./brain/source_code\n",
                encoding="utf-8",
            )
            cli = _load_cli()

            with patch("smak.cli.InternalNomicEmbedding", new=self.DummyEmbedder), patch(
                "smak.cli._load_vector_store",
                new=lambda index_name, cfg, uri: {"index": index_name, "uri": uri},
            ), patch("smak.cli.validate_vector_store_dimension", new=lambda *args, **kwargs: None):
                _, vector_store = cli._load_vector_store_for_cli("source_code", str(config_path))

            self.assertEqual(vector_store["uri"], "./brain/source_code")

    def test_load_vector_store_for_cli_rejects_unknown_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "workspace.yaml"
            config_path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code\n",
                encoding="utf-8",
            )
            cli = _load_cli()

            with self.assertRaises(Exception) as ctx:
                cli._load_vector_store_for_cli("forbidden", str(config_path))

            self.assertIn("config.indices", str(ctx.exception))

    def test_query_command_rejects_unknown_index(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "workspace.yaml"
            config_path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code\n",
                encoding="utf-8",
            )
            cli = _load_cli()
            result = runner.invoke(
                cli.main,
                ["query", "hello", "--index", "forbidden", "--config", str(config_path)],
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("config.indices", result.output)

    def test_ingest_command_rejects_unknown_index(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "workspace.yaml"
            source_dir = Path(tmp_dir) / "src"
            source_dir.mkdir()
            (source_dir / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
            config_path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code\n",
                encoding="utf-8",
            )
            cli = _load_cli()
            result = runner.invoke(
                cli.main,
                [
                    "ingest",
                    "--folder",
                    str(source_dir),
                    "--index",
                    "forbidden",
                    "--config",
                    str(config_path),
                ],
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("config.indices", result.output)

    def test_stats_command_rejects_unknown_index(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "workspace.yaml"
            config_path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code\n",
                encoding="utf-8",
            )
            cli = _load_cli()
            result = runner.invoke(
                cli.main,
                ["stats", "--index", "forbidden", "--config", str(config_path)],
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("config.indices", result.output)

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
                SmakConfig(indices=[IndexConfig(name="code", description="Code", uri="memory")]),
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
