from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from smak.cli import _default_config_template, _ingest_folder, main
from smak.config import SmakConfig, StorageConfig


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
    def test_default_config_template_includes_storage(self) -> None:
        template = _default_config_template()

        self.assertIn("storage:", template)
        self.assertIn("provider: milvus_lite", template)
        self.assertIn("uri: ./milvus_data.db", template)

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

            stats = _ingest_folder(
                folder,
                "code",
                config,
                vector_store_loader=loader,
                node_class_loader=lambda: FakeNode,
            )

            self.assertEqual(stats.files, 1)
            self.assertEqual(stats.vectors, 1)
            self.assertEqual(created["index"], "code")
            self.assertEqual(saved[0].metadata["relations"], ["issue::1"])

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
                init_result = runner.invoke(
                    main,
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
                    main,
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

        with patch.object(cli.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(Exception) as exc:
                cli._load_vector_store("code", SmakConfig())
            self.assertIn("llama-index-vector-stores-milvus", str(exc.exception))

    def test_load_text_node_missing_dependency(self) -> None:
        from smak import cli

        with patch.object(cli.importlib.util, "find_spec", return_value=None):
            with self.assertRaises(Exception) as exc:
                cli._load_text_node_class()
            self.assertIn("llama-index-core", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
