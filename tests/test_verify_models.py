from __future__ import annotations

import importlib
import io
import sys
import unittest
from contextlib import contextmanager, redirect_stdout
from types import ModuleType, SimpleNamespace
from typing import Any, Generator
from unittest.mock import patch


@contextmanager
def install_fake_dependencies() -> Generator[None, None, None]:
    fake_llms = ModuleType("llama_index.core.llms")

    class FakeChatMessage:
        def __init__(self, role: str, content: str) -> None:
            self.role = role
            self.content = content

    fake_llms.ChatMessage = FakeChatMessage

    fake_core = ModuleType("llama_index.core")
    fake_core.llms = fake_llms

    fake_root = ModuleType("llama_index")
    fake_root.core = fake_core

    fake_requests = ModuleType("requests")

    class FakeSession:
        def post(
            self, url: str, json: dict[str, Any], headers: dict[str, str], timeout: float
        ) -> Any:
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

    fake_embeddings.BaseEmbedding = FakeBaseEmbedding

    fake_openai_like = ModuleType("llama_index.llms.openai_like")

    class FakeOpenAILike:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

        def chat(self, _messages: list[Any]) -> Any:
            return SimpleNamespace(message=SimpleNamespace(content="ok"))

    fake_openai_like.OpenAILike = FakeOpenAILike

    fake_llms_pkg = ModuleType("llama_index.llms")
    fake_llms_pkg.openai_like = fake_openai_like

    with patch.dict(
        sys.modules,
        {
            "requests": fake_requests,
            "llama_index": fake_root,
            "llama_index.core": fake_core,
            "llama_index.core.llms": fake_llms,
            "llama_index.core.embeddings": fake_embeddings,
            "llama_index.llms": fake_llms_pkg,
            "llama_index.llms.openai_like": fake_openai_like,
        },
    ):
        yield


class TestVerifyModels(unittest.TestCase):
    def test_verify_models_script_runs(self) -> None:
        with install_fake_dependencies():
            sys.modules.pop("scripts.verify_models", None)
            verify_models = importlib.import_module("scripts.verify_models")

            class DummyEmbedder:
                def __init__(self, **_kwargs: Any) -> None:
                    return None

                def get_text_embedding(self, _text: str) -> list[float]:
                    return [0.0, 1.0]

            class DummyLLM:
                def chat(self, _messages: list[Any]) -> Any:
                    return SimpleNamespace(message=SimpleNamespace(content="ok"))

            with patch.object(verify_models, "InternalNomicEmbedding", DummyEmbedder), patch.object(
                verify_models, "build_internal_llm", lambda **_kwargs: DummyLLM()
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    exit_code = verify_models.main(["--text", "ping", "--provider", "qwen"])

                self.assertEqual(exit_code, 0)
                output = stream.getvalue()
                self.assertIn("Embedding vector length: 2", output)
                self.assertIn("LLM response: ok", output)


if __name__ == "__main__":
    unittest.main()
