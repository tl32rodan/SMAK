from __future__ import annotations

import unittest
from dataclasses import dataclass

from smak.core.domain import KnowledgeUnit
from smak.db.adapter import InMemoryAdapter, VectorAdapter, VectorDocument


class TestAdapters(unittest.TestCase):
    def test_in_memory_adapter_saves_and_loads(self) -> None:
        adapter = InMemoryAdapter()
        unit = KnowledgeUnit(uid="u1", content="content", source_type="issue")

        adapter.save_units([unit])

        self.assertEqual(adapter.load_units(), [unit])

    def test_vector_adapter_saves_documents(self) -> None:
        saved: list[VectorDocument] = []

        @dataclass
        class FakeIndex:
            def add(self, docs: list[VectorDocument]) -> None:
                saved.extend(docs)

        @dataclass
        class FakeRegistry:
            def get_index(self, name: str) -> FakeIndex:
                return FakeIndex()

        @dataclass
        class FakeEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[float(len(texts[0]))]]

        adapter = VectorAdapter(registry=FakeRegistry(), embedder=FakeEmbedder())
        unit = KnowledgeUnit(
            uid="code::login",
            content="def login(): pass",
            source_type="source_code",
            relations=("issue:1",),
            metadata={"language": "python"},
        )

        adapter.save("code", [unit])

        self.assertEqual(saved[0].uid, "code::login")
        self.assertEqual(saved[0].payload["relations"], ["issue:1"])


if __name__ == "__main__":
    unittest.main()
