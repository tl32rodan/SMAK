from types import SimpleNamespace

from smak.bridge import models
from smak.bridge.models import InternalNomicEmbedding, build_internal_llm


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, expected_payload, response_payload):
        self.expected_payload = expected_payload
        self.response_payload = response_payload
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        assert json == self.expected_payload
        return DummyResponse(self.response_payload)


def test_internal_nomic_embedding_posts_request() -> None:
    session = DummySession(
        expected_payload={"model": "nomic-test", "input": ["hello"]},
        response_payload={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
    )
    embedder = InternalNomicEmbedding(
        api_base="http://nomic.test",
        model="nomic-test",
        session=session,
    )

    vector = embedder.get_text_embedding("hello")

    assert vector == [0.1, 0.2, 0.3]


def test_internal_nomic_embedding_batches_response_in_order() -> None:
    session = DummySession(
        expected_payload={"model": "nomic-test", "input": ["a", "b"]},
        response_payload={
            "data": [
                {"index": 1, "embedding": [2.0]},
                {"index": 0, "embedding": [1.0]},
            ]
        },
    )
    embedder = InternalNomicEmbedding(
        api_base="http://nomic.test",
        model="nomic-test",
        session=session,
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
