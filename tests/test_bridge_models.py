import json
from types import SimpleNamespace

import httpx

from smak.bridge import models
from smak.bridge.models import InternalNomicEmbedding, build_internal_llm


def test_internal_nomic_embedding_posts_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload == {"model": "nomic-test", "input": ["hello"]}
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    embedder = InternalNomicEmbedding(
        api_base="http://nomic.test",
        model="nomic-test",
        client=client,
    )

    vector = embedder.get_text_embedding("hello")

    assert vector == [0.1, 0.2, 0.3]


def test_internal_nomic_embedding_batches_response_in_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["input"] == ["a", "b"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [2.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    embedder = InternalNomicEmbedding(
        api_base="http://nomic.test",
        model="nomic-test",
        client=client,
    )

    vectors = embedder._get_text_embeddings(["a", "b"])

    assert vectors == [[1.0], [2.0]]


def test_build_internal_llm_uses_internal_defaults(monkeypatch) -> None:
    captured = {}

    def fake_openai_like(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(models, "OpenAILike", fake_openai_like)
    llm = build_internal_llm(provider="gpt-oss", model="gpt-oss-turbo", api_base="http://x")

    assert llm.model == "gpt-oss-turbo"
    assert captured["api_base"] == "http://x"

