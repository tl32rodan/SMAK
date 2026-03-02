from __future__ import annotations

import importlib
import sys
import unittest
from dataclasses import dataclass
from types import ModuleType

from smak.core.domain import KnowledgeUnit


@dataclass
class FakeVectorDocument:
    uid: str
    vector: list[float]
    payload: dict


class TestVectorAdapter(unittest.TestCase):
    def setUp(self) -> None:
        fake_root = ModuleType("faiss_storage_lib")
        fake_core = ModuleType("faiss_storage_lib.core")
        fake_schema = ModuleType("faiss_storage_lib.core.schema")
        fake_schema.VectorDocument = FakeVectorDocument

        self._original_modules = {
            name: sys.modules.get(name)
            for name in (
                "faiss_storage_lib",
                "faiss_storage_lib.core",
                "faiss_storage_lib.core.schema",
                "smak.storage.vector_adapter",
            )
        }
        sys.modules["faiss_storage_lib"] = fake_root
        sys.modules["faiss_storage_lib.core"] = fake_core
        sys.modules["faiss_storage_lib.core.schema"] = fake_schema
        sys.modules.pop("smak.storage.vector_adapter", None)

    def tearDown(self) -> None:
        for name, module in self._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_vector_adapter_saves_documents_with_expected_payload(self) -> None:
        vector_adapter_module = importlib.import_module("smak.storage.vector_adapter")
        VectorAdapter = vector_adapter_module.VectorAdapter

        saved: list[FakeVectorDocument] = []

        @dataclass
        class FakeIndex:
            def add(self, docs: list[FakeVectorDocument]) -> None:
                saved.extend(docs)

        @dataclass
        class FakeRegistry:
            index_name: str | None = None

            def get_index(self, name: str) -> FakeIndex:
                self.index_name = name
                return FakeIndex()

        @dataclass
        class FakeEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[float(len(text))] for text in texts]

        registry = FakeRegistry()
        adapter = VectorAdapter(registry=registry, embedder=FakeEmbedder())
        units = [
            KnowledgeUnit(
                uid="code::login",
                content="def login(): pass",
                source_type="source_code",
                relations=("issue:1",),
                metadata={"language": "python"},
            ),
            KnowledgeUnit(
                uid="code::logout",
                content="def logout(): pass",
                source_type="source_code",
                metadata={"language": "python"},
            ),
        ]

        adapter.save("code", units)

        self.assertEqual(registry.index_name, "code")
        self.assertEqual([doc.uid for doc in saved], ["code::login", "code::logout"])
        self.assertEqual(saved[0].payload["relations"], ["issue:1"])
        self.assertEqual(saved[1].payload["meta"]["language"], "python")


if __name__ == "__main__":
    unittest.main()
