from __future__ import annotations

import importlib
import io
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner


class TestVerifyModels(unittest.TestCase):
    def test_verify_models_function_runs(self) -> None:
        sys.modules.pop("smak.verify_models", None)
        verify_models_module = importlib.import_module("smak.verify_models")

        class DummyEmbedder:
            def __init__(self, **_kwargs: Any) -> None:
                return None

            def get_text_embedding(self, _text: str) -> list[float]:
                return [0.0, 1.0]

        class DummyLLM:
            def chat(self, _messages: list[Any]) -> Any:
                return SimpleNamespace(message=SimpleNamespace(content="ok"))

        class DummyChatMessage:
            def __init__(self, role: str, content: str) -> None:
                self.role = role
                self.content = content

        embedding_length, response_text = verify_models_module.verify_models(
            text="ping",
            provider="qwen",
            llm_model=None,
            llm_api_base=None,
            embedding_model=None,
            embedding_api_base=None,
            embedder_factory=DummyEmbedder,
            llm_factory=lambda **_kwargs: DummyLLM(),
            chat_message_factory=DummyChatMessage,
        )

        self.assertEqual(embedding_length, 2)
        self.assertEqual(response_text, "ok")

    def test_verify_models_script_runs(self) -> None:
        sys.modules.pop("scripts.verify_models", None)
        verify_models_script = importlib.import_module("scripts.verify_models")

        stream = io.StringIO()
        with patch.object(verify_models_script, "verify_models", return_value=(2, "ok")):
            with redirect_stdout(stream):
                exit_code = verify_models_script.main(["--text", "ping", "--provider", "qwen"])

        self.assertEqual(exit_code, 0)
        output = stream.getvalue()
        self.assertIn("Embedding vector length: 2", output)
        self.assertIn("LLM response: ok", output)

    def test_verify_models_cli_runs(self) -> None:
        sys.modules.pop("smak.cli", None)
        cli = importlib.import_module("smak.cli")

        runner = CliRunner()
        with patch.object(cli, "verify_models", return_value=(3, "pong")):
            result = runner.invoke(cli.main, ["verify-models", "--text", "ping"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Embedding vector length: 3", result.output)
        self.assertIn("LLM response: pong", result.output)


if __name__ == "__main__":
    unittest.main()
