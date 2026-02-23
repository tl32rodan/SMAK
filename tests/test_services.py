from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from smak.services.doctor import DoctorService
from smak.services.ingest import IngestService
from smak.services.query import QueryService
from smak.services.sidecar import SidecarService


class FakeNode:
    def __init__(self, text: str, id_: str, metadata: dict) -> None:
        self.text = text
        self.id_ = id_
        self.metadata = metadata
        self.embedding = None


class FakeVectorStore:
    def __init__(self) -> None:
        self.saved: list = []

    def add(self, nodes: list) -> None:
        self.saved.extend(nodes)

    def delete_by_metadata(self, key: str, value: str) -> None:
        return None

    def get_by_id(self, uid: str):
        for node in self.saved:
            if node.id_ == uid:
                return {"uid": uid, "metadata": node.metadata, "content": node.text}
        return None


class TestServices(unittest.TestCase):
    class DummyEmbedder:
        def get_text_embedding(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    def test_ingest_service_processes_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "a.py"
            src.write_text("def foo():\n    return 1\n", encoding="utf-8")
            store = FakeVectorStore()
            service = IngestService(vector_store=store)
            stats = service.ingest_folder(
                Path(tmp_dir),
                incremental=False,
                node_class_loader=lambda: FakeNode,
                embedder_loader=self.DummyEmbedder,
            )
            self.assertEqual(stats.files, 1)
            self.assertGreaterEqual(stats.vectors, 1)

    def test_query_service_expands_one_hop_relations(self) -> None:
        store = SimpleNamespace(
            search=lambda vector, top_k=5: [
                {
                    "uid": "func_A",
                    "score": 0.9,
                    "content": "A",
                    "metadata": {"relations": ["issue_12"]},
                }
            ],
            get_by_id=lambda uid: {"uid": uid, "content": "Issue body"},
        )
        payload = QueryService(store, embedder=self.DummyEmbedder()).search("query", top_k=1)
        self.assertEqual(payload["hits"][0]["match_type"], "semantic")
        self.assertEqual(payload["related_context"][0]["source_hit"], "func_A")

    def test_sidecar_service_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()
            result = service.update(
                source, json.dumps([{"symbol": "main.py::hello", "relations": ["issue-1"]}])
            )
            self.assertEqual(result["applied_updates"], 1)

    def test_doctor_service_detects_dangling_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            sidecar = Path(tmp_dir) / "main.py.sidecar.yaml"
            sidecar.write_text(
                "symbols:\n  - name: main.py::hello\n    relations:\n      - missing_uid\n",
                encoding="utf-8",
            )
            service = DoctorService(vector_store=SimpleNamespace(get_by_id=lambda uid: None))
            warnings = service.validate_mesh_integrity(Path(tmp_dir))
            self.assertEqual(len(warnings), 1)


if __name__ == "__main__":
    unittest.main()
