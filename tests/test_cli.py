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

from smak.config import IndexConfig, SmakConfig


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
                base_path: object = None,
            ) -> object:
                captured["index_name"] = index_config.name
                captured["index_uri"] = cli._resolve_index_uri(
                    index_config,
                    base_path=base_path,
                )
                return SimpleNamespace(dimension=1)

            with (
                patch("smak.cli._load_vector_store", new=fake_loader),
                patch("smak.cli.validate_vector_store_dimension", new=lambda store, dim: None),
                patch("smak.cli.initialize_embedding_dimensions", new=lambda cfg, emb: cfg),
            ):
                cli._load_vector_store_for_cli("docs", str(config_path))

            self.assertEqual(captured["index_name"], "docs")
            self.assertEqual(
                captured["index_uri"],
                str((Path(tmp_dir).resolve() / "smak_data" / "docs").resolve()),
            )


    def test_ingest_command_forwards_follow_symlinks_option(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir) / "src"
            folder.mkdir()
            config_path = Path(tmp_dir) / "workspace.yaml"
            config_path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: source\n",
                encoding="utf-8",
            )
            captured: dict[str, object] = {}

            class FakeIngestService:
                def __init__(self, vector_store: object) -> None:
                    self.vector_store = vector_store

                def ingest_folder(self, *args: object, **kwargs: object) -> object:
                    captured.update(kwargs)
                    return SimpleNamespace(files=0, skipped=0, vectors=0)

            with (
                patch(
                    "smak.cli._load_vector_store_for_cli",
                    new=lambda index, config: (SmakConfig(), object()),
                ),
                patch("smak.cli.IngestService", new=FakeIngestService),
            ):
                cli = importlib.import_module("smak.cli")
                result = runner.invoke(
                    cli.main,
                    [
                        "ingest",
                        "--folder",
                        str(folder),
                        "--index",
                        "source_code",
                        "--config",
                        str(config_path),
                        "--no-follow-symlinks",
                    ],
                )

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(captured.get("follow_symlinks"), False)

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
            self.assertIn("::hello", payload[0])

    def test_sidecar_update_merges_metadata(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "example.py"
            source.write_text("def hello():\n    return True\n", encoding="utf-8")
            cli = importlib.import_module("smak.cli")
            updates = json.dumps(
                [{"symbol": "example.py::hello", "intent": "greeting", "relations": ["issue:1"]}]
            )
            result = runner.invoke(
                cli.main, ["sidecar", "update", str(source), "--updates", updates]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn('{\n    "file_path"', result.output)
            self.assertTrue((Path(tmp_dir) / "example.py.sidecar.yaml").exists())

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
            source.with_name("main.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: func_A\n"
                "    relations:\n"
                "      - issue_1\n",
                encoding="utf-8",
            )

            with (
                patch(
                    "smak.cli._load_vector_store_for_cli",
                    new=lambda index, config: (
                        SmakConfig(
                            indices=[IndexConfig(name="source_code", description="source")]
                        ),
                        QueryStore(),
                    ),
                ),
                patch(
                    "smak.services.query.InternalNomicEmbedding",
                    new=lambda: SimpleNamespace(get_text_embedding=lambda text: [0.1]),
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
