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
        self.deleted_ids: list[list[str]] = []
        self.tracked_sources: dict[str, list[str]] = {}

    def add(self, nodes: list) -> None:
        self.saved.extend(nodes)

    def delete_by_metadata(self, key: str, value: str) -> None:
        return None

    def get_all_tracked_sources(self) -> dict[str, list[str]]:
        return self.tracked_sources

    def delete_by_ids(self, uids: list[str]) -> None:
        self.deleted_ids.append(uids)


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
            stats = service.ingest_paths(
                [Path(tmp_dir)],
                incremental=False,
                node_class_loader=lambda: FakeNode,
                embedder_loader=DummyEmbedder,
            )
            self.assertEqual(stats.files, 1)
            self.assertGreaterEqual(stats.vectors, 1)

    def test_ingest_service_skips_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "code.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
            (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
            store = FakeVectorStore()
            service = IngestService(vector_store=store)
            stats = service.ingest_paths(
                [root],
                incremental=False,
                node_class_loader=lambda: FakeNode,
                embedder_loader=DummyEmbedder,
            )
            self.assertEqual(stats.files, 1)
            self.assertEqual(stats.skipped, 1)
            for node in store.saved:
                self.assertNotIn("image.png", node.id_)

    def test_ingest_service_processes_unknown_text_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "script.sh").write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
            (root / "diff.patch").write_text("--- a\n+++ b\n@@ -1 +1 @@\n", encoding="utf-8")
            store = FakeVectorStore()
            service = IngestService(vector_store=store)
            stats = service.ingest_paths(
                [root],
                incremental=False,
                node_class_loader=lambda: FakeNode,
                embedder_loader=DummyEmbedder,
            )
            self.assertEqual(stats.files, 2)
            saved_sources = {node.metadata.get("source") for node in store.saved}
            self.assertEqual(len(store.saved), 2)
            self.assertTrue(any("script.sh" in (s or "") for s in saved_sources))
            self.assertTrue(any("diff.patch" in (s or "") for s in saved_sources))


    def test_ingest_service_stores_absolute_uid_for_python_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src = root / "pkg" / "a.py"
            src.parent.mkdir(parents=True)
            src.write_text("def foo():\n    return 1\n", encoding="utf-8")

            store = FakeVectorStore()
            service = IngestService(vector_store=store)
            service.ingest_paths(
                [root],
                incremental=False,
                node_class_loader=lambda: FakeNode,
                embedder_loader=DummyEmbedder,
            )

            self.assertGreaterEqual(len(store.saved), 1)
            self.assertTrue(store.saved[0].id_.startswith(str(src.resolve())))
            self.assertIn("::foo", store.saved[0].id_)

    def test_iter_source_files_can_toggle_symlink_following(self) -> None:
        from smak.services.ingest import service as ingest_module

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


    def test_skip_file_predicate_filters_files(self) -> None:
        from smak.services.ingest import service as ingest_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "a.py").write_text("x = 1\n", encoding="utf-8")
            (root / "b.sidecar.yaml").write_text("symbols: []\n", encoding="utf-8")

            all_files = list(ingest_module._iter_source_files(root))
            filtered = list(
                ingest_module._iter_source_files(
                    root, skip_file=lambda p: p.name.endswith(".sidecar.yaml")
                )
            )
            self.assertEqual(len(all_files), 2)
            self.assertEqual(len(filtered), 1)
            self.assertTrue(filtered[0].name.endswith(".py"))

    def test_ingest_read_text_fallback_replaces_invalid_bytes(self) -> None:
        from smak.services.ingest import service as ingest_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.py"
            path.write_bytes(b"ok\x80")
            self.assertEqual(ingest_module._read_text_with_fallback(path), "ok�")

    def test_ingest_read_text_fallback_handles_unicode_decode_error(self) -> None:
        from smak.services.ingest import service as ingest_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "any.py"
            path.write_text("ok", encoding="utf-8")
            with patch.object(
                Path,
                "read_text",
                side_effect=[UnicodeDecodeError("utf-8", b"", 0, 1, "boom"), "ok"],
            ):
                self.assertEqual(ingest_module._read_text_with_fallback(path), "ok")

    def test_ingest_service_sync_prunes_ghost_sources_and_calls_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "a.py"
            source.write_text("def foo():\n    return 1\n", encoding="utf-8")

            ghost_sidecar = root / ".ghost.py.sidecar.yaml"
            ghost_sidecar.write_text("symbols: []\n", encoding="utf-8")

            store = FakeVectorStore()
            store.tracked_sources = {"ghost.py": ["ghost::1", "ghost::2"], "a.py": ["a::1"]}

            ghost_callbacks: list[tuple[str, list[Path]]] = []

            def on_ghost(source_key: str, folders: list[Path]) -> None:
                ghost_callbacks.append((source_key, folders))
                # Simulate sidecar cleanup
                for folder in folders:
                    sidecar = folder / f".{Path(source_key).name}.sidecar.yaml"
                    if sidecar.exists():
                        sidecar.unlink()

            service = IngestService(vector_store=store)
            stats = service.ingest_paths(
                [root],
                incremental=False,
                sync=True,
                node_class_loader=lambda: FakeNode,
                embedder_loader=DummyEmbedder,
                on_ghost_source=on_ghost,
            )

            self.assertEqual(stats.deleted, 1)
            self.assertEqual(store.deleted_ids, [["ghost::1", "ghost::2"]])
            self.assertEqual(len(ghost_callbacks), 1)
            self.assertEqual(ghost_callbacks[0][0], "ghost.py")
            self.assertFalse(ghost_sidecar.exists())


    def test_ingest_paths_processes_multiple_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src = root / "src"
            src.mkdir()
            lib = root / "lib"
            lib.mkdir()
            (src / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
            (lib / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")

            store = FakeVectorStore()
            service = IngestService(vector_store=store)
            stats = service.ingest_paths(
                [src, lib],
                incremental=False,
                node_class_loader=lambda: FakeNode,
                embedder_loader=DummyEmbedder,
            )

            self.assertEqual(stats.files, 2)
            self.assertGreaterEqual(stats.vectors, 2)

    def test_ingest_paths_sync_prunes_only_true_ghosts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src = root / "src"
            src.mkdir()
            lib = root / "lib"
            lib.mkdir()
            (src / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
            (lib / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")

            # 'ghost.py' is tracked but does not exist in either folder
            store = FakeVectorStore()
            store.tracked_sources = {
                "a.py": ["src_a::1"],
                "b.py": ["lib_b::1"],
                "ghost.py": ["ghost::1"],
            }

            service = IngestService(vector_store=store)
            stats = service.ingest_paths(
                [src, lib],
                incremental=False,
                sync=True,
                node_class_loader=lambda: FakeNode,
                embedder_loader=DummyEmbedder,
            )

            self.assertEqual(stats.deleted, 1)
            self.assertEqual(store.deleted_ids, [["ghost::1"]])


    def test_ingest_stores_env_var_uid_when_env_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src = root / "pkg" / "a.py"
            src.parent.mkdir(parents=True)
            src.write_text("def foo():\n    return 1\n", encoding="utf-8")

            store = FakeVectorStore()
            service = IngestService(vector_store=store)
            service.ingest_paths(
                [root],
                incremental=False,
                node_class_loader=lambda: FakeNode,
                embedder_loader=DummyEmbedder,
                env={"TEST_DDI_ROOT": str(root)},
            )

            self.assertGreaterEqual(len(store.saved), 1)
            self.assertTrue(store.saved[0].id_.startswith("$TEST_DDI_ROOT/"))
            self.assertIn("::foo", store.saved[0].id_)

    def test_ingest_preserves_absolute_uid_when_no_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src = root / "a.py"
            src.write_text("def foo():\n    return 1\n", encoding="utf-8")

            store = FakeVectorStore()
            service = IngestService(vector_store=store)
            service.ingest_paths(
                [root],
                incremental=False,
                node_class_loader=lambda: FakeNode,
                embedder_loader=DummyEmbedder,
            )

            self.assertGreaterEqual(len(store.saved), 1)
            self.assertTrue(store.saved[0].id_.startswith(str(src.resolve())))


if __name__ == "__main__":
    unittest.main()
