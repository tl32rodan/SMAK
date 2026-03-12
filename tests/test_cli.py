from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import click
from click.testing import CliRunner

from smak.config import EmbeddingConfig, IndexConfig, SmakConfig


def _install_fake_dependencies() -> None:
    fake_requests = ModuleType("requests")

    class FakeSession:
        def post(self, url: str, json: dict, headers: dict, timeout: float) -> SimpleNamespace:
            return SimpleNamespace(
                raise_for_status=lambda: None, json=lambda: {"data": [{"embedding": [0.0]}]}
            )

    fake_requests.Session = FakeSession

    fake_embeddings = ModuleType("llama_index.core.embeddings")

    class FakeBaseEmbedding:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

        def get_text_embedding(self, text: str) -> list[float]:
            return [0.0]

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
    fake_openai_like.OpenAILike = object

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


class TestCli(unittest.TestCase):
    def test_default_config_template_includes_index_uri(self) -> None:
        cli = importlib.import_module("smak.cli")
        template = cli._default_config_template()
        self.assertIn("uri: ./smak_data/source_code", template)
        self.assertIn("Customize uri", template)
        self.assertNotIn("storage:", template)
        self.assertNotIn("llm:", template)
        self.assertIn("paths:", template)

    def test_load_vector_store_for_cli_raises_for_unknown_index(self) -> None:
        cli = importlib.import_module("smak.cli")
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "workspace.yaml"
            config_path.write_text(
                "indices:\n  - name: source_code\n    description: source\n",
                encoding="utf-8",
            )
            with self.assertRaises(click.ClickException) as context:
                cli._load_vector_store_for_cli("missing", str(config_path))
            self.assertIn("not found", str(context.exception))


    def test_load_vector_store_for_cli_uses_index_uri_fallback(self) -> None:
        cli = importlib.import_module("smak.cli")
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "workspace.yaml"
            config_path.write_text(
                "indices:\n  - name: docs\n    description: docs index\n",
                encoding="utf-8",
            )
            captured: dict[str, object] = {}

            def fake_loader(
                index_config: object,
                config: object,
            ) -> object:
                captured["index_name"] = index_config.name
                captured["index_uri"] = index_config.uri
                return SimpleNamespace(dimension=1)

            with (
                patch("smak.cli.load_and_validate_vector_store", new=fake_loader),
                patch("smak.cli.init_config", new=lambda cfg, **kw: cfg),
            ):
                cli._load_vector_store_for_cli("docs", str(config_path))

            self.assertEqual(captured["index_name"], "docs")
            self.assertEqual(
                captured["index_uri"],
                str((Path(tmp_dir).resolve() / "smak_data" / "docs").resolve()),
            )


    def test_ingest_command_forwards_follow_symlinks_option_and_sync(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            config_path = Path(tmp_dir) / "workspace.yaml"
            config_path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: source\n"
                f"    path: {str(folder)}\n",
                encoding="utf-8",
            )
            captured: dict[str, object] = {}

            class FakeIngestService:
                def __init__(self, vector_store: object) -> None:
                    self.vector_store = vector_store

                def ingest_paths(self, *args: object, **kwargs: object) -> object:
                    captured.update(kwargs)
                    return SimpleNamespace(files=0, skipped=0, vectors=0, deleted=0)

            with (
                patch(
                    "smak.cli._load_vector_store_for_cli",
                    new=lambda index, config, embedding_config=None: (
                        SmakConfig(),
                        IndexConfig(name="source_code", description="source", paths=[str(folder)]),
                        object(),
                    ),
                ),
                patch("smak.cli.IngestService", new=FakeIngestService),
            ):
                cli = importlib.import_module("smak.cli")
                result = runner.invoke(
                    cli.main,
                    [
                        "ingest",
                        "--index",
                        "source_code",
                        "--config",
                        str(config_path),
                        "--no-follow-symlinks",
                        "--sync",
                    ],
                )

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(captured.get("follow_symlinks"), False)
            self.assertEqual(captured.get("sync"), True)
            self.assertIn("Ghost Files Pruned: 0", result.output)
            self.assertIn("embedder_loader", captured)
            self.assertTrue(callable(captured["embedder_loader"]))

    def test_ingest_command_accepts_embedding_setup_option(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            emb_yaml = Path(tmp_dir) / "custom_emb.yaml"
            emb_yaml.write_text(
                "api_base: http://custom:7777\nmodel: test-model\n",
                encoding="utf-8",
            )
            captured_emb: list[EmbeddingConfig] = []

            class FakeIngestService:
                def __init__(self, vector_store: object) -> None:
                    pass

                def ingest_paths(self, *args: object, **kwargs: object) -> object:
                    loader = kwargs.get("embedder_loader")
                    if loader:
                        captured_emb.append(loader)
                    return SimpleNamespace(files=0, skipped=0, vectors=0, deleted=0)

            with (
                patch(
                    "smak.cli._load_vector_store_for_cli",
                    new=lambda index, config, embedding_config=None: (
                        SmakConfig(),
                        IndexConfig(name="src", description="s", paths=[str(folder)]),
                        object(),
                    ),
                ),
                patch("smak.cli.IngestService", new=FakeIngestService),
            ):
                cli = importlib.import_module("smak.cli")
                result = runner.invoke(
                    cli.main,
                    [
                        "ingest",
                        "--index", "src",
                        "--embedding-setup", str(emb_yaml),
                    ],
                )

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(len(captured_emb), 1)

    def test_query_command_accepts_embedding_setup_option(self) -> None:
        runner = CliRunner()
        cli = importlib.import_module("smak.cli")
        result = runner.invoke(cli.main, ["query", "--help"])
        self.assertIn("--embedding-setup", result.output)

    def test_doctor_command_accepts_embedding_setup_option(self) -> None:
        runner = CliRunner()
        cli = importlib.import_module("smak.cli")
        result = runner.invoke(cli.main, ["doctor", "--help"])
        self.assertIn("--embedding-setup", result.output)

    def test_sidecar_inspect_json_output(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "example.py"
            source.write_text("def hello():\n    return True\n", encoding="utf-8")
            cli = importlib.import_module("smak.cli")
            result = runner.invoke(cli.main, ["sidecar", "inspect", str(source), "--json-output"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("[\n    ", result.output)
            payload = json.loads(result.output)
            self.assertEqual(payload[0], "hello")

    def test_sidecar_update_full_sync_creates_sidecar(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "example.py"
            source.write_text("def hello():\n    return True\n", encoding="utf-8")
            cli = importlib.import_module("smak.cli")
            result = runner.invoke(cli.main, ["sidecar", "update", str(source)])
            self.assertEqual(result.exit_code, 0)
            self.assertIn('"total_symbols"', result.output)
            self.assertTrue((Path(tmp_dir) / ".example.py.sidecar.yaml").exists())

    def test_sidecar_update_single_symbol(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "example.py"
            source.write_text("def hello():\n    return True\n", encoding="utf-8")
            cli = importlib.import_module("smak.cli")
            result = runner.invoke(
                cli.main,
                [
                    "sidecar", "update", str(source),
                    "--symbol", "example.py::hello",
                    "--intent", "greeting",
                    "--relations", "issue:1",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn('"total_symbols"', result.output)
            self.assertTrue((Path(tmp_dir) / ".example.py.sidecar.yaml").exists())

    def test_sidecar_clear_removes_symbol(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "example.py"
            source.write_text("def hello():\n    return True\n", encoding="utf-8")
            cli = importlib.import_module("smak.cli")
            runner.invoke(
                cli.main,
                [
                    "sidecar", "update", str(source),
                    "--symbol", "example.py::hello",
                    "--intent", "greeting",
                ],
            )
            result = runner.invoke(
                cli.main,
                ["sidecar", "clear", str(source), "--symbol", "example.py::hello"],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn('"cleared_symbol"', result.output)

    def test_query_command_outputs_structured_json(self) -> None:
        runner = CliRunner()
        captured_top_k: list[int] = []

        class QueryStore(SimpleNamespace):
            def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
                captured_top_k.append(top_k)
                return [
                    {
                        "uid": "func_A",
                        "score": 0.8,
                        "content": "A",
                        "metadata": {"source": "src/main.py"},
                    }
                ]

            def get_by_id(self, uid: str) -> dict | None:
                return {"uid": uid, "content": "related"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = workspace / "workspace_config.yaml"
            config_path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: source\n",
                encoding="utf-8",
            )
            source = workspace / "src" / "main.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("def a():\n    pass\n", encoding="utf-8")
            source.with_name(".main.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: func_A\n"
                "    relations:\n"
                "      - issue_1\n",
                encoding="utf-8",
            )

            with (
                patch(
                    "smak.cli._load_vector_store_for_cli",
                    new=lambda index, config, embedding_config=None: (
                        SmakConfig(
                            indices=[IndexConfig(name="source_code", description="source")]
                        ),
                        IndexConfig(name="source_code", description="source", paths=[str(workspace)]),
                        QueryStore(),
                    ),
                ),
                patch(
                    "smak.utils.embedding.InternalNomicEmbedding",
                    new=lambda **kw: SimpleNamespace(get_text_embedding=lambda text: [0.1]),
                ),
            ):
                cli = importlib.import_module("smak.cli")
                result = runner.invoke(
                    cli.main,
                    [
                        "query",
                        "hello",
                        "--index",
                        "code",
                        "--config",
                        str(config_path),
                    ],
                )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(captured_top_k, [1])
        self.assertIn('{\n    "hits"', result.output)
        payload = json.loads(result.output)
        self.assertIn("hits", payload)
        self.assertIn("related_context", payload)

    def test_sidecar_update_has_no_reingest_options(self) -> None:
        runner = CliRunner()
        cli = importlib.import_module("smak.cli")
        result = runner.invoke(cli.main, ["sidecar", "update", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("--reingest", result.output)
        self.assertNotIn("--index", result.output)
        self.assertNotIn("--config", result.output)


if __name__ == "__main__":
    unittest.main()
