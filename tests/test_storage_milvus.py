from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import patch

from smak.storage.milvus import (
    MilvusLiteVectorSearchIndex,
    MilvusLiteVectorStore,
    load_milvus_lite_store,
)


@dataclass
class FakeNode:
    id_: str
    text: str
    metadata: dict
    embedding: list[float]


class FakeMilvusClient:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.collections: dict[str, list[dict]] = {}

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_collection(
        self,
        *,
        collection_name: str,
        dimension: int,
        vector_field_name: str,
        primary_field_name: str,
        auto_id: bool,
        enable_dynamic_field: bool,
    ) -> None:
        self.collections[collection_name] = []

    def insert(self, collection_name: str, payloads: list[dict]) -> None:
        self.collections[collection_name].extend(payloads)

    def search(self, collection_name: str, data: list[list[float]], limit: int, **kwargs):
        hits = []
        for row in self.collections.get(collection_name, [])[:limit]:
            hits.append({"id": row["id"], "entity": row})
        return [hits]

    def get(self, collection_name: str, ids: list[str], **kwargs):
        hits = []
        for row in self.collections.get(collection_name, []):
            if row["id"] in ids:
                hits.append({"id": row["id"], "entity": row})
        return hits


class FakePymilvus(ModuleType):
    def __init__(self) -> None:
        super().__init__("pymilvus")
        self.MilvusClient = FakeMilvusClient


class FakeEmbedder:
    def get_query_embedding(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class TestMilvusLiteStorage(unittest.TestCase):
    def _install_fake_pymilvus(self) -> None:
        sys.modules["pymilvus"] = FakePymilvus()

    def test_store_add_search_and_get(self) -> None:
        self._install_fake_pymilvus()
        with patch("smak.storage.milvus.importlib.util.find_spec", return_value=object()):
            store = MilvusLiteVectorStore(uri="memory.db", collection_name="code", dim=3)
            node = FakeNode(
                id_="unit-1",
                text="hello",
                metadata={"mesh_relations": ["issue::1"]},
                embedding=[0.1, 0.2, 0.3],
            )
            store.add([node])

            results = store.search([0.1, 0.2, 0.3], top_k=1)
            self.assertEqual(results[0]["uid"], "unit-1")
            self.assertEqual(results[0]["content"], "hello")
            self.assertEqual(results[0]["metadata"]["mesh_relations"], ["issue::1"])

            fetched = store.get_by_id("unit-1")
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched["uid"], "unit-1")

    def test_search_index_uses_embedder(self) -> None:
        self._install_fake_pymilvus()
        with patch("smak.storage.milvus.importlib.util.find_spec", return_value=object()):
            store = MilvusLiteVectorStore(uri="memory.db", collection_name="docs", dim=3)
            store.add(
                [
                    FakeNode(
                        id_="doc-1",
                        text="payload",
                        metadata={},
                        embedding=[0.1, 0.2, 0.3],
                    )
                ]
            )
            index = MilvusLiteVectorSearchIndex(store=store, embedder=FakeEmbedder())
            results = list(index.search("query"))
            self.assertEqual(results[0]["uid"], "doc-1")
            self.assertEqual(index.get_by_id("doc-1")["uid"], "doc-1")

    def test_load_milvus_lite_store_raises_clear_error(self) -> None:
        with patch(
            "smak.storage.milvus.MilvusLiteVectorStore",
            side_effect=ModuleNotFoundError("pymilvus missing"),
        ):
            with self.assertRaises(ModuleNotFoundError) as exc:
                load_milvus_lite_store(uri="memory.db", collection_name="code", dim=3)
        self.assertIn("pymilvus", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
