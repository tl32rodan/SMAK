from __future__ import annotations

import importlib
import sys
import unittest
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Generator
from unittest.mock import patch


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class DummySession:
    def __init__(
        self,
        expected_payload: dict[str, Any],
        response_payload: dict[str, Any],
        expected_url: str | None = None,
    ) -> None:
        self.expected_payload = expected_payload
        self.response_payload = response_payload
        self.expected_url = expected_url
        self.calls: list[dict[str, Any]] = []

    def post(
        self, url: str, json: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> DummyResponse:
        if self.expected_url and url != self.expected_url:
            raise AssertionError(f"Unexpected url: {url}")
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        self.assert_payload(json)
        return DummyResponse(self.response_payload)

    def assert_payload(self, payload: dict[str, Any]) -> None:
        if payload != self.expected_payload:
            raise AssertionError(f"Unexpected payload: {payload}")


@contextmanager
def install_fake_dependencies() -> Generator[dict[str, Any], None, None]:
    fake_requests = ModuleType("requests")

    class FakeSession:
        def post(
            self, url: str, json: dict[str, Any], headers: dict[str, str], timeout: float
        ) -> DummyResponse:
            return DummyResponse({"data": [{"index": 0, "embedding": [0.0]}]})

    def fake_post(
        url: str, json: dict[str, Any], headers: dict[str, str], timeout: float
    ) -> DummyResponse:
        return DummyResponse({"data": [{"index": 0, "embedding": [0.0]}]})

    fake_requests.Session = FakeSession
    fake_requests.post = fake_post

    fake_embeddings = ModuleType("llama_index.core.embeddings")

    class FakeBaseEmbedding:
        def __init__(self, model_name: str, embed_batch_size: int) -> None:
            self.model_name = model_name
            self.embed_batch_size = embed_batch_size

        def get_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embedding(text)

        def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return self._get_text_embeddings(texts)

    fake_embeddings.BaseEmbedding = FakeBaseEmbedding

    fake_openai_like = ModuleType("llama_index.llms.openai_like")

    class FakeOpenAILike:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    fake_openai_like.OpenAILike = FakeOpenAILike

    fake_core = ModuleType("llama_index.core")
    fake_core.embeddings = fake_embeddings

    fake_llms = ModuleType("llama_index.llms")
    fake_llms.openai_like = fake_openai_like

    fake_root = ModuleType("llama_index")
    fake_root.core = fake_core
    fake_root.llms = fake_llms

    with patch.dict(
        sys.modules,
        {
            "requests": fake_requests,
            "llama_index": fake_root,
            "llama_index.core": fake_core,
            "llama_index.core.embeddings": fake_embeddings,
            "llama_index.llms": fake_llms,
            "llama_index.llms.openai_like": fake_openai_like,
        },
    ):
        yield {"FakeOpenAILike": FakeOpenAILike}


class TestInternalNomicEmbedding(unittest.TestCase):
    def _load_models(self) -> Any:
        sys.modules.pop("smak.bridge.models", None)
        return importlib.import_module("smak.bridge.models")

    def test_internal_nomic_embedding_posts_request(self) -> None:
        with install_fake_dependencies():
            models = self._load_models()
            session = DummySession(
                expected_payload={"model": "nomic-test", "input": ["hello"]},
                response_payload={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
            )
            embedder = models.InternalNomicEmbedding(
                api_base="http://nomic.test",
                model="nomic-test",
                session=session,
            )

            vector = embedder.get_text_embedding("hello")

            self.assertEqual(vector, [0.1, 0.2, 0.3])

    def test_internal_nomic_embedding_batches_response_in_order(self) -> None:
        with install_fake_dependencies():
            models = self._load_models()
            session = DummySession(
                expected_payload={"model": "nomic-test", "input": ["a", "b"]},
                response_payload={
                    "data": [
                        {"index": 1, "embedding": [2.0]},
                        {"index": 0, "embedding": [1.0]},
                    ]
                },
            )
            embedder = models.InternalNomicEmbedding(
                api_base="http://nomic.test",
                model="nomic-test",
                session=session,
            )

            vectors = embedder._get_text_embeddings(["a", "b"])

            self.assertEqual(vectors, [[1.0], [2.0]])

    def test_internal_nomic_embedding_uses_api_embed_endpoint(self) -> None:
        with install_fake_dependencies():
            models = self._load_models()
            session = DummySession(
                expected_payload={"model": "nomic-test", "input": ["hello"]},
                response_payload={"data": [{"index": 0, "embedding": [0.1]}]},
                expected_url="http://nomic.test/api/embed",
            )
            embedder = models.InternalNomicEmbedding(
                api_base="http://nomic.test",
                model="nomic-test",
                session=session,
            )

            vector = embedder.get_text_embedding("hello")

            self.assertEqual(vector, [0.1])

    def test_build_internal_llm_uses_internal_defaults(self) -> None:
        with install_fake_dependencies() as fake:
            models = self._load_models()
            llm = models.build_internal_llm(
                provider="gpt-oss",
                model="gpt-oss-turbo",
                api_base="http://x",
            )

            self.assertIsInstance(llm, fake["FakeOpenAILike"])
            self.assertEqual(llm.model, "gpt-oss-turbo")
            self.assertEqual(llm.api_base, "http://x")

    def test_build_internal_llm_defaults_to_configured_models(self) -> None:
        with install_fake_dependencies() as fake, patch.dict("os.environ", {}, clear=True):
            models = self._load_models()

            with (
                patch.object(models, "_DEFAULT_QWEN_LLM_MODEL", "qwen-default"),
                patch.object(models, "_DEFAULT_QWEN_API_BASE", "http://qwen-default.test/v1"),
                patch.object(models, "_DEFAULT_GPT_OSS_LLM_MODEL", "gpt-oss-default"),
                patch.object(models, "_DEFAULT_GPT_API_BASE", "http://gpt-default.test/v1"),
            ):
                qwen_llm = models.build_internal_llm(provider="qwen")
                gpt_llm = models.build_internal_llm(provider="gpt-oss")

            self.assertIsInstance(qwen_llm, fake["FakeOpenAILike"])
            self.assertEqual(qwen_llm.model, "qwen-default")
            self.assertEqual(qwen_llm.api_base, "http://qwen-default.test/v1")
            self.assertEqual(gpt_llm.model, "gpt-oss-default")
            self.assertEqual(gpt_llm.api_base, "http://gpt-default.test/v1")


if __name__ == "__main__":
    unittest.main()
