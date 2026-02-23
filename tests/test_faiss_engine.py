from __future__ import annotations

import importlib
import threading
import unittest
from dataclasses import dataclass


@dataclass
class FakeVectorDocument:
    uid: str
    vector: list[float]
    payload: dict


class FakeConnection:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def execute(self, sql: str) -> None:
        self.commands.append(sql)


class FakeBaseEngine:
    created_threads: list[int] = []
    instances: list["FakeBaseEngine"] = []

    def __init__(self, _path: str, _dim: int) -> None:
        self.conn = FakeConnection()
        self.docs: list[FakeVectorDocument] = []
        self.persisted = False
        FakeBaseEngine.created_threads.append(threading.get_ident())
        FakeBaseEngine.instances.append(self)

    def add(self, docs: list[FakeVectorDocument]) -> None:
        self.docs.extend(docs)

    def persist(self) -> None:
        self.persisted = True

    def search(self, _embedding: list[float], top_k: int) -> list[FakeVectorDocument]:
        return self.docs[:top_k]

    def get_by_id(self, uid: str) -> FakeVectorDocument | None:
        for doc in self.docs:
            if doc.uid == uid:
                return doc
        return None

    def delete_by_metadata(self, key: str, value: str) -> None:
        self.docs = [d for d in self.docs if d.payload.get("metadata", {}).get(key) != value]


class TestThreadLocalFaissEngine(unittest.TestCase):
    def setUp(self) -> None:
        FakeBaseEngine.created_threads.clear()
        FakeBaseEngine.instances.clear()
        self.mod = importlib.import_module("smak.storage.faiss_engine")

    def test_reuses_engine_within_same_thread(self) -> None:
        engine = self.mod.FaissEngine(path="memory", dim=3, _factory=FakeBaseEngine)

        engine.add([FakeVectorDocument("a", [0.1], {"metadata": {"source": "x"}})])
        engine.add([FakeVectorDocument("b", [0.2], {"metadata": {"source": "y"}})])

        self.assertEqual(len(FakeBaseEngine.instances), 1)
        self.assertEqual(engine.count(), 2)
        self.assertIn("PRAGMA journal_mode=NA;", FakeBaseEngine.instances[0].conn.commands)

    def test_creates_dedicated_engine_per_thread(self) -> None:
        engine = self.mod.FaissEngine(path="memory", dim=3, _factory=FakeBaseEngine)

        def worker() -> None:
            engine.add([FakeVectorDocument("w", [0.1], {"metadata": {"source": "w"}})])

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        engine.add([FakeVectorDocument("m", [0.2], {"metadata": {"source": "m"}})])

        self.assertEqual(len(FakeBaseEngine.instances), 2)
        self.assertEqual(len(set(FakeBaseEngine.created_threads)), 2)


if __name__ == "__main__":
    unittest.main()
