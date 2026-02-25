from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from smak.config import IndexConfig, SmakConfig
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
        config = SmakConfig(indices=[IndexConfig(name="source_code", description="source")])
        payload = QueryService(
            store,
            config=config,
            vector_store_loader=lambda index_config, cfg: store,
            embedder=self.DummyEmbedder(),
        ).search("query", top_k=1)
        self.assertEqual(payload["hits"][0]["match_type"], "semantic")
        self.assertEqual(payload["related_context"][0]["source_hit"], "func_A")
        self.assertEqual(payload["hits"][0]["content"], "A")

    def test_query_service_resolves_relations_across_indices_without_extra_search(self) -> None:
        class PrimaryStore:
            def __init__(self) -> None:
                self.search_calls = 0
                self.get_by_id_calls: list[str] = []

            def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
                self.search_calls += 1
                return [
                    {
                        "uid": "func_A",
                        "score": 0.95,
                        "content": "def A(): pass",
                        "metadata": {"relations": ["issue:42"]},
                    }
                ]

            def get_by_id(self, uid: str) -> dict | None:
                self.get_by_id_calls.append(uid)
                return None

        class SecondaryStore:
            def __init__(self) -> None:
                self.get_by_id_calls: list[str] = []

            def get_by_id(self, uid: str) -> dict | None:
                self.get_by_id_calls.append(uid)
                if uid == "issue:42":
                    return {"uid": uid, "content": "Fix login bug"}
                return None

        primary = PrimaryStore()
        secondary = SecondaryStore()
        loader_calls: list[str] = []

        def loader(index_config: IndexConfig, config: SmakConfig) -> object:
            loader_calls.append(index_config.name)
            if index_config.name == "issues":
                return secondary
            return primary

        config = SmakConfig(
            indices=[
                IndexConfig(name="source_code", description="source"),
                IndexConfig(name="issues", description="issues"),
            ]
        )

        payload = QueryService(
            primary,
            config=config,
            vector_store_loader=loader,
            embedder=self.DummyEmbedder(),
        ).search("query", top_k=1)

        self.assertEqual(primary.search_calls, 1)
        self.assertEqual(loader_calls, ["source_code", "issues"])
        self.assertEqual(payload["hits"][0]["content"], "def A(): pass")
        self.assertEqual(payload["related_context"][0]["content"], "Fix login bug")
        self.assertEqual(payload["related_context"][0]["match_type"], "relation")


    def test_query_service_uses_loader_with_index_config(self) -> None:
        class PrimaryStore:
            def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
                return [{"uid": "func_A", "content": "A", "metadata": {"relations": ["issue:42"]}}]

            def get_by_id(self, uid: str) -> dict | None:
                return None

        class SecondaryStore:
            def get_by_id(self, uid: str) -> dict | None:
                return {"uid": uid, "content": "Issue"} if uid == "issue:42" else None

        loader_calls: list[str] = []

        def loader(index_config: IndexConfig, config: SmakConfig) -> object:
            loader_calls.append(index_config.name)
            return SecondaryStore() if index_config.name == "issues" else PrimaryStore()

        config = SmakConfig(
            indices=[
                IndexConfig(name="source_code", description="source"),
                IndexConfig(name="issues", description="issues"),
            ]
        )

        payload = QueryService(
            PrimaryStore(),
            config=config,
            vector_store_loader=loader,
            embedder=self.DummyEmbedder(),
        ).search("query", top_k=1)

        self.assertEqual(loader_calls, ["source_code", "issues"])
        self.assertEqual(payload["related_context"][0]["content"], "Issue")

    def test_shared_sidecar_suffixes_work_across_yaml_extensions(self) -> None:
        from smak.services.sidecar_paths import iter_sidecar_files

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.py").write_text("print('a')\n", encoding="utf-8")
            (root / "a.py.sidecar.yaml").write_text("symbols: []\n", encoding="utf-8")
            (root / "b.py").write_text("print('b')\n", encoding="utf-8")
            (root / "b.py.sidecar.yml").write_text("symbols: []\n", encoding="utf-8")

            doctor = DoctorService()
            issues = doctor.validate_sidecars(root)

            sidecar_files = sorted(path.name for path in iter_sidecar_files(root))

            self.assertEqual(issues, [])
            self.assertEqual(sidecar_files, ["a.py.sidecar.yaml", "b.py.sidecar.yml"])


    def test_iter_source_files_can_toggle_symlink_following(self) -> None:
        from smak.services import ingest as ingest_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            real_dir = root / "real"
            real_dir.mkdir()
            (real_dir / "linked.py").write_text("print('x')\n", encoding="utf-8")
            link_dir = root / "link"
            try:
                link_dir.symlink_to(real_dir, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation not supported in this environment")

            without_follow = list(
                ingest_module._iter_source_files(root, follow_symlinks=False)
            )
            with_follow = list(
                ingest_module._iter_source_files(root, follow_symlinks=True)
            )

            without_paths = {str(path.relative_to(root)) for path in without_follow}
            with_paths = {str(path.relative_to(root)) for path in with_follow}

            self.assertNotIn("link/linked.py", without_paths)
            self.assertIn("link/linked.py", with_paths)

    def test_sidecar_service_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()
            result = service.update(
                source, json.dumps([{"symbol": "main.py::hello", "relations": ["issue-1"]}])
            )
            self.assertEqual(result["applied_updates"], 1)


    def test_ingest_read_text_fallback_replaces_invalid_bytes(self) -> None:
        from smak.services import ingest as ingest_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.py"
            path.write_bytes(b"ok\x80")
            self.assertEqual(ingest_module._read_text_with_fallback(path), "ok�")

    def test_sidecar_read_text_fallback_replaces_invalid_bytes(self) -> None:
        from smak.services import sidecar as sidecar_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.py"
            path.write_bytes(b"ok\x80")
            self.assertEqual(sidecar_module._read_text_with_fallback(path), "ok�")


    def test_ingest_read_text_fallback_handles_unicode_decode_error(self) -> None:
        from smak.services import ingest as ingest_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "any.py"
            path.write_text("ok", encoding="utf-8")
            with patch.object(
                Path,
                "read_text",
                side_effect=[UnicodeDecodeError("utf-8", b"", 0, 1, "boom"), "ok"],
            ):
                self.assertEqual(ingest_module._read_text_with_fallback(path), "ok")

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
