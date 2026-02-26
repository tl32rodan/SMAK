from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smak.services.ingest import IngestService


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


class DummyEmbedder:
    def get_text_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class TestIngestService(unittest.TestCase):
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
                embedder_loader=DummyEmbedder,
            )
            self.assertEqual(stats.files, 1)
            self.assertGreaterEqual(stats.vectors, 1)

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

            without_follow = list(ingest_module._iter_source_files(root, follow_symlinks=False))
            with_follow = list(ingest_module._iter_source_files(root, follow_symlinks=True))

            without_paths = {str(path.relative_to(root)) for path in without_follow}
            with_paths = {str(path.relative_to(root)) for path in with_follow}

            self.assertNotIn("link/linked.py", without_paths)
            self.assertIn("link/linked.py", with_paths)

    def test_ingest_read_text_fallback_replaces_invalid_bytes(self) -> None:
        from smak.services import ingest as ingest_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.py"
            path.write_bytes(b"ok\x80")
            self.assertEqual(ingest_module._read_text_with_fallback(path), "ok�")

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


if __name__ == "__main__":
    unittest.main()
