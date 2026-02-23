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
    def test_default_config_template_includes_storage(self) -> None:
        cli = importlib.import_module("smak.cli")
        template = cli._default_config_template()
        self.assertIn("storage:", template)

    def test_sidecar_inspect_json_output(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "example.py"
            source.write_text("def hello():\n    return True\n", encoding="utf-8")
            cli = importlib.import_module("smak.cli")
            result = runner.invoke(cli.main, ["sidecar", "inspect", str(source), "--json-output"])
            self.assertEqual(result.exit_code, 0)
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
            self.assertTrue((Path(tmp_dir) / "example.py.sidecar.yaml").exists())

    def test_query_command_outputs_structured_json(self) -> None:
        runner = CliRunner()

        class QueryStore(SimpleNamespace):
            def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
                return [
                    {
                        "uid": "func_A",
                        "score": 0.8,
                        "content": "A",
                        "metadata": {"relations": ["issue_1"]},
                    }
                ]

            def get_by_id(self, uid: str) -> dict | None:
                return {"uid": uid, "content": "related"}

        with (
            patch(
                "smak.cli._load_vector_store_for_cli",
                new=lambda index, config: (object(), QueryStore()),
            ),
            patch(
                "smak.services.query.InternalNomicEmbedding",
                new=lambda: SimpleNamespace(get_text_embedding=lambda text: [0.1]),
            ),
        ):
            cli = importlib.import_module("smak.cli")
            result = runner.invoke(cli.main, ["query", "hello", "--index", "code"])

        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertIn("hits", payload)
        self.assertIn("related_context", payload)


if __name__ == "__main__":
    unittest.main()
