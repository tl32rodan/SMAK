from __future__ import annotations

import importlib
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
        def __init__(self, model_name: str, embed_batch_size: int) -> None:
            self.model_name = model_name
            self.embed_batch_size = embed_batch_size

        def get_text_embedding(self, text: str) -> list[float]:
            return [0.0]

        def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    fake_embeddings.BaseEmbedding = FakeBaseEmbedding

    fake_openai_like = ModuleType("llama_index.llms.openai_like")

    class FakeOpenAILike:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    fake_openai_like.OpenAILike = FakeOpenAILike

    fake_core = ModuleType("llama_index.core")
    fake_core.embeddings = fake_embeddings

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


class TestCli(unittest.TestCase):
    class DummyEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 1.0, 2.0] for text in texts]

    def test_default_config_template_includes_storage(self) -> None:
        cli = _load_cli()
        template = cli._default_config_template()

        self.assertIn("storage:", template)
        self.assertIn("provider: milvus_lite", template)
        self.assertIn("uri: ./milvus_data.db", template)
        self.assertIn("llm:", template)
        self.assertNotIn("embedding_dimensions", template)
        self.assertNotIn("llama_index:", template)

    def test_ingest_folder_processes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            source = folder / "example.py"
            source.write_text("def foo():\n    return 1\n", encoding="utf-8")
            sidecar = folder / "example.py.sidecar.yaml"
            sidecar.write_text(
                "symbols:\n  - name: foo\n    relations:\n      - issue::1\n",
                encoding="utf-8",
            )

            saved: list = []
            created: dict[str, str] = {}

            def loader(index_name: str, config: SmakConfig) -> FakeVectorStore:
                self.assertEqual(config.storage.uri, "vault.db")
                created["index"] = index_name
                return FakeVectorStore(saved, index_name)

            config = SmakConfig(storage=StorageConfig(uri="vault.db"))

            cli = _load_cli()
            stats = cli._ingest_folder(
                folder,
                "code",
                config,
                vector_store_loader=loader,
                node_class_loader=lambda: FakeNode,
                embedder_loader=self.DummyEmbedder,
            )

            self.assertEqual(stats.files, 1)
            self.assertEqual(stats.vectors, 1)
            self.assertEqual(created["index"], "code")
            self.assertEqual(saved[0].metadata["relations"], ["issue::1"])

    def test_ingest_folder_uses_embedder_dimension(self) -> None:
        class WideEmbedder(SimpleNamespace):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 5 for _ in texts]

        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            source = folder / "example.py"
            source.write_text("def foo():\n    return 1\n", encoding="utf-8")

            observed: dict[str, int] = {}

            def loader(index_name: str, config: SmakConfig) -> FakeVectorStore:
                observed["dim"] = config.embedding_dimensions
                return FakeVectorStore([], index_name)

            config = SmakConfig(storage=StorageConfig(uri="vault.db"))

            cli = _load_cli()
            cli._ingest_folder(
                folder,
                "code",
                config,
                vector_store_loader=loader,
                node_class_loader=lambda: FakeNode,
                embedder_loader=WideEmbedder,
            )

            self.assertEqual(observed["dim"], 5)

    def test_cli_init_and_ingest(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            folder = tmp_path / "repo"
            folder.mkdir()
            source = folder / "note.md"
            source.write_text("---\nid: issue-1\n---\nbody\n", encoding="utf-8")
            config_path = tmp_path / "workspace_config.yaml"
            config_path.write_text("storage:\n  uri: vault.db\n", encoding="utf-8")

            saved: list = []
            with patch(
                "smak.cli._load_vector_store",
                new=lambda index, config: FakeVectorStore(saved, index),
            ), patch("smak.cli._load_text_node_class", new=lambda: FakeNode):
                cli = _load_cli()
                init_result = runner.invoke(
                    cli.main,
                    [
                        "init",
                        "--path",
                        str(tmp_path / "generated.yaml"),
                    ],
                )

                self.assertEqual(init_result.exit_code, 0)
                self.assertIn("workspace config", init_result.output)
                self.assertTrue((tmp_path / "generated.yaml").exists())

                result = runner.invoke(
                    cli.main,
                    [
                        "ingest",
                        "--folder",
                        str(folder),
                        "--index",
                        "issues",
                        "--config",
                        str(config_path),
                    ],
                )

                self.assertEqual(result.exit_code, 0)
                self.assertIn("Ingestion Complete", result.output)
                self.assertTrue(saved)

    def test_load_vector_store_missing_dependency(self) -> None:
        from smak import cli

        with patch(
            "smak.storage.milvus.load_milvus_lite_store",
            side_effect=ModuleNotFoundError("Vector store dependency missing."),
        ):
            with self.assertRaises(Exception) as exc:
                cli._load_vector_store("code", SmakConfig())
            self.assertIn("Vector store dependency missing", str(exc.exception))

    def test_load_text_node_missing_dependency(self) -> None:
        from smak import cli

        with patch.object(cli.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(Exception) as exc:
                cli._load_text_node_class()
            self.assertIn("llama-index-core", str(exc.exception))

    def test_server_command_invokes_launcher(self) -> None:
        runner = CliRunner()
        with patch("smak.cli.launch_server") as launcher:
            cli = _load_cli()
            result = runner.invoke(cli.main, ["server", "--port", "7777", "--config", "cfg.yaml"])

        self.assertEqual(result.exit_code, 0)
        launcher.assert_called_once_with(port=7777, config_path="cfg.yaml")


if __name__ == "__main__":
    unittest.main()
