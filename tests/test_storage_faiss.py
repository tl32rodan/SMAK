from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import patch

from smak.storage.faiss_adapter import (
    FaissVectorSearchIndex,
    FaissVectorStore,
    load_faiss_store,
)


@dataclass
class FakeVectorDocument:
    uid: str
    vector: list[float]
    payload: dict


@dataclass
class FakeResult:
    uid: str
    payload: dict
    score: float


class FakeFaissEngine:
    def __init__(self, path: str, dim: int) -> None:
        self.path = path
        self.dim = dim
        self.docs: list[FakeVectorDocument] = []
        self.persisted = False
        self.deleted: list[tuple[str, object]] = []

    def add(self, docs: list[FakeVectorDocument]) -> None:
        self.docs.extend(docs)

    def persist(self) -> None:
        self.persisted = True

    def delete_by_metadata(self, key: str, value: object) -> None:
        self.deleted.append((key, value))

    def search(self, embedding: list[float], top_k: int) -> list[FakeResult]:
        return [
            FakeResult(uid=doc.uid, payload=doc.payload, score=0.9)
            for doc in self.docs[:top_k]
        ]

    def get_by_id(self, uid: str) -> FakeVectorDocument | None:
        for doc in self.docs:
            if doc.uid == uid:
                return doc
        return None


@dataclass
class FakeNode:
    id_: str
    text: str
    metadata: dict
    embedding: list[float]


class FakeEmbedder:
    def get_query_embedding(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class TestFaissAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self._installed_modules: list[str] = []
        self._install_fake_faiss()

    def tearDown(self) -> None:
        for name in self._installed_modules:
            sys.modules.pop(name, None)

    def _install_fake_faiss(self) -> None:
        fake_root = ModuleType("faiss_storage_lib")
        fake_engine_pkg = ModuleType("faiss_storage_lib.engine")
        fake_schema_pkg = ModuleType("faiss_storage_lib.core")
        fake_engine = ModuleType("faiss_storage_lib.engine.faiss_engine")
        fake_schema = ModuleType("faiss_storage_lib.core.schema")
        fake_engine.FaissEngine = FakeFaissEngine
        fake_schema.VectorDocument = FakeVectorDocument
        for module in (
            fake_root,
            fake_engine_pkg,
            fake_schema_pkg,
            fake_engine,
            fake_schema,
        ):
            sys.modules[module.__name__] = module
            self._installed_modules.append(module.__name__)

    def test_store_add_search_and_get(self) -> None:
        store = FaissVectorStore(uri="memory", collection_name="code", dim=3)
        node = FakeNode(
            id_="unit-1",
            text="hello",
            metadata={"mesh_relations": ["issue:1"]},
            embedding=[0.1, 0.2, 0.3],
        )
        store.add([node])

        results = store.search([0.1, 0.2, 0.3], top_k=1)
        self.assertEqual(results[0]["uid"], "unit-1")
        self.assertEqual(results[0]["content"], "hello")
        self.assertEqual(results[0]["metadata"]["mesh_relations"], ["issue:1"])
        self.assertTrue(store._engine.persisted)

        fetched = store.get_by_id("unit-1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["uid"], "unit-1")

    def test_search_index_uses_embedder(self) -> None:
        store = FaissVectorStore(uri="memory", collection_name="docs", dim=3)
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
        index = FaissVectorSearchIndex(store=store, embedder=FakeEmbedder())
        results = index.search("query")
        self.assertEqual(results[0]["uid"], "doc-1")
        self.assertEqual(index.get_by_id("doc-1")["uid"], "doc-1")

    def test_load_faiss_store_raises_clear_error(self) -> None:
        with patch(
            "smak.storage.faiss_adapter.importlib.import_module",
            side_effect=ModuleNotFoundError("faiss missing"),
        ):
            with self.assertRaises(ModuleNotFoundError) as exc:
                load_faiss_store(uri="memory", collection_name="code", dim=3)
        self.assertIn("faiss-storage-lib", str(exc.exception))

    def test_delete_by_metadata_calls_engine(self) -> None:
        store = FaissVectorStore(uri="memory", collection_name="code", dim=3)

        store.delete_by_metadata("source", "src/a.py")

        self.assertEqual(store._engine.deleted, [("source", "src/a.py")])
        self.assertTrue(store._engine.persisted)


if __name__ == "__main__":
    unittest.main()
