from __future__ import annotations

import unittest

from smak.config import SmakConfig
from smak.embedding import (
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


if __name__ == "__main__":
    unittest.main()
