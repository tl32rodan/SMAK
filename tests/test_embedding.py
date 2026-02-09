from __future__ import annotations

import unittest

from smak.config import SmakConfig
from smak.embedding import (
    InternalNomicEmbedding,
    detect_embedding_dimension,
    initialize_embedding_dimensions,
    validate_vector_store_dimension,
)


class DummyEmbedderWithDimension:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def get_embedding_dimension(self) -> int:
        return self._dimension


class DummyEmbedderWithEmbeddings:
    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimension for _ in texts]


class DummyVectorStore:
    def __init__(self, dimension: int) -> None:
        self.dim = dimension


class DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class DummySession:
    def __init__(self) -> None:
        self.last_json: dict | None = None

    def post(self, _url: str, json: dict, headers: dict, timeout: float) -> DummyResponse:
        _ = headers, timeout
        self.last_json = json
        return DummyResponse({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})


class TestEmbeddingHelpers(unittest.TestCase):
    def test_detect_embedding_dimension_from_embedder_method(self) -> None:
        embedder = DummyEmbedderWithDimension(7)

        dimension = detect_embedding_dimension(embedder)

        self.assertEqual(dimension, 7)

    def test_detect_embedding_dimension_from_embeddings(self) -> None:
        embedder = DummyEmbedderWithEmbeddings(4)

        dimension = detect_embedding_dimension(embedder)

        self.assertEqual(dimension, 4)

    def test_initialize_embedding_dimensions_updates_config(self) -> None:
        config = SmakConfig()
        embedder = DummyEmbedderWithDimension(8)

        updated = initialize_embedding_dimensions(config, embedder)

        self.assertEqual(updated.embedding_dimensions, 8)

    def test_validate_vector_store_dimension_raises_on_mismatch(self) -> None:
        vector_store = DummyVectorStore(2)

        with self.assertRaises(ValueError):
            validate_vector_store_dimension(vector_store, 3)

    def test_internal_nomic_embedding_initializes_all_members(self) -> None:
        session = DummySession()

        embedder = InternalNomicEmbedding(
            api_base="http://localhost:1234",
            model="nomic-test",
            timeout=5.0,
            headers={"x-test": "1"},
            session=session,
            embed_batch_size=16,
        )

        self.assertEqual(embedder.api_base, "http://localhost:1234")
        self.assertEqual(embedder.model, "nomic-test")
        self.assertEqual(embedder.timeout, 5.0)
        self.assertEqual(embedder.headers, {"x-test": "1"})
        self.assertIs(embedder.session, session)
        self.assertEqual(embedder.model_name, "nomic-test")
        self.assertEqual(embedder.embed_batch_size, 16)

    def test_internal_nomic_embedding_gets_embeddings(self) -> None:
        session = DummySession()
        embedder = InternalNomicEmbedding(session=session)

        vectors = embedder._get_text_embeddings(["hello"])

        self.assertEqual(vectors, [[0.1, 0.2, 0.3]])
        self.assertIsNotNone(session.last_json)
        self.assertEqual(session.last_json["input"], ["hello"])


if __name__ == "__main__":
    unittest.main()
