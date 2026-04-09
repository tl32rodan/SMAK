"""Tests for smak.core_ops — shared operation layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from smak.config import EmbeddingConfig, SmakConfig, IndexConfig
from smak.core_ops import (
    do_describe,
    do_enrich_batch,
    do_enrich_file,
    do_enrich_symbol,
    do_graph_stats,
    do_health,
    do_ingest,
    do_lookup,
    do_search,
    do_search_all,
    resolve_source_path,
    setup_config,
)


def _patch_init_config():
    return patch("smak.core_ops.init_config", side_effect=lambda cfg, **kw: cfg)


def _patch_query():
    return (
        patch("smak.core_ops.load_and_validate_vector_store", return_value=object()),
        patch("smak.core_ops.create_query_service"),
    )


def _patch_sidecar():
    return patch("smak.core_ops.create_sidecar_service")


def _patch_ingest():
    return (
        patch("smak.core_ops.load_and_validate_vector_store"),
        patch("smak.core_ops.IngestService"),
    )


def _patch_doctor():
    return patch("smak.core_ops.create_doctor_service")


def _make_config(tmp_dir: str) -> SmakConfig:
    """Create a SmakConfig with a source_code index pointing to tmp_dir/src."""
    src = Path(tmp_dir) / "src"
    src.mkdir(exist_ok=True)
    return SmakConfig(indices=[
        IndexConfig(name="source_code", description="src", uri=str(src / "data"), paths=[str(src)]),
    ])


def _write_source(tmp_dir: str, name: str = "a.py",
                   content: str = "def hello():\n    pass\n") -> Path:
    source = Path(tmp_dir) / "src" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(content, encoding="utf-8")
    return source


class TestSetupConfig(unittest.TestCase):
    def test_setup_config_returns_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "workspace_config.yaml"
            config_path.write_text(
                "indices:\n  - name: src\n    description: d\n    uri: ./data\n    paths:\n      - .\n",
                encoding="utf-8",
            )
            with _patch_init_config():
                cfg, emb_cfg = setup_config(str(config_path))
            self.assertIsInstance(cfg, SmakConfig)
            self.assertIsInstance(emb_cfg, EmbeddingConfig)

    def test_setup_config_raises_on_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            setup_config("/nonexistent/path.yaml")


class TestResolveSourcePath(unittest.TestCase):
    def test_resolves_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = _write_source(tmp_dir)
            ic = SimpleNamespace(paths=[str(Path(tmp_dir) / "src")])
            result = resolve_source_path("source_code", ic, "a.py")
            self.assertEqual(result, source)

    def test_resolves_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = _write_source(tmp_dir)
            ic = SimpleNamespace(paths=[str(Path(tmp_dir) / "src")])
            result = resolve_source_path("source_code", ic, str(source))
            self.assertEqual(result, source)

    def test_raises_on_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for sub in ("sub1", "sub2"):
                d = Path(tmp_dir) / "src" / sub
                d.mkdir(parents=True)
                (d / "dup.py").write_text("x = 1\n", encoding="utf-8")
            ic = SimpleNamespace(paths=[str(Path(tmp_dir) / "src")])
            with self.assertRaises(ValueError) as cm:
                resolve_source_path("source_code", ic, "dup.py")
            self.assertIn("Ambiguous", str(cm.exception))

    def test_raises_on_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "src").mkdir()
            ic = SimpleNamespace(paths=[str(Path(tmp_dir) / "src")])
            with self.assertRaises(FileNotFoundError):
                resolve_source_path("source_code", ic, "missing.py")


class TestDoDescribe(unittest.TestCase):
    def test_lists_indices(self) -> None:
        cfg = SmakConfig(indices=[
            IndexConfig(name="src", description="code", uri="./data/src", paths=["./src"]),
            IndexConfig(name="docs", description="docs", uri="./data/docs", paths=["./docs"]),
        ])
        result = do_describe(cfg)
        self.assertEqual(len(result["indices"]), 2)
        self.assertEqual(result["indices"][0]["name"], "src")


class TestDoSearch(unittest.TestCase):
    def test_search_delegates_to_query_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            p1, p2 = _patch_query()
            with p1, p2 as qs_factory:
                qs_factory.return_value.search.return_value = {"hits": []}
                result = do_search(cfg, "test query", index="source_code")
                self.assertEqual(result, {"hits": []})
                qs_factory.return_value.search.assert_called_once_with("test query", top_k=5)

    def test_search_all_queries_multiple_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            src = Path(tmp_dir) / "src"
            src.mkdir()
            cfg = SmakConfig(indices=[
                IndexConfig(name="source_code", description="src", uri=str(src / "data/src"), paths=[str(src)]),
                IndexConfig(name="tests", description="tests", uri=str(src / "data/tests"), paths=[str(src)]),
            ])
            p1, p2 = _patch_query()
            with p1, p2 as qs_factory:
                qs_factory.return_value.search.return_value = {"hits": []}
                result = do_search_all(cfg, "query")
                self.assertIn("source_code", result)
                self.assertIn("tests", result)


class TestDoLookup(unittest.TestCase):
    def test_lookup_delegates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            p1, p2 = _patch_query()
            with p1, p2 as qs_factory:
                qs_factory.return_value.lookup.return_value = {"found": True, "uid": "a::b"}
                result = do_lookup(cfg, "a::b", index="source_code")
                self.assertTrue(result["found"])


class TestDoIngest(unittest.TestCase):
    def test_ingest_returns_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            p1, p2 = _patch_ingest()
            with p1, p2 as ingest_cls:
                ingest_cls.return_value.ingest_paths.return_value = MagicMock(
                    files=3, skipped=1, vectors=10,
                )
                result = do_ingest(cfg, index="source_code")
                self.assertEqual(result["files"], 3)
                self.assertEqual(result["vectors"], 10)


class TestDoEnrichSymbol(unittest.TestCase):
    def test_enrich_symbol_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            _write_source(tmp_dir)
            with _patch_sidecar() as svc_factory:
                svc = svc_factory.return_value
                svc.inspect.return_value = ["hello"]
                svc.update.return_value = {"total_symbols": 1}
                result = do_enrich_symbol(
                    cfg, "a.py", "hello", intent="greets", index="source_code",
                )
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["intent"], "greets")

    def test_enrich_symbol_rejects_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            _write_source(tmp_dir)
            with _patch_sidecar() as svc_factory:
                svc_factory.return_value.inspect.return_value = ["hello"]
                result = do_enrich_symbol(
                    cfg, "a.py", "nonexistent", intent="x", index="source_code",
                )
                self.assertEqual(result["status"], "error")
                self.assertIn("valid_symbols", result)


class TestDoEnrichFile(unittest.TestCase):
    def test_enrich_file_syncs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = _make_config(tmp_dir)
            _write_source(tmp_dir)
            with _patch_sidecar() as svc_factory:
                svc_factory.return_value.update.return_value = {"total_symbols": 1}
                result = do_enrich_file(cfg, "a.py", index="source_code")
                self.assertEqual(result["total_symbols"], 1)


class TestDoHealth(unittest.TestCase):
    def test_healthy(self) -> None:
        cfg = SmakConfig()
        with _patch_doctor() as df:
            df.return_value.validate_all.return_value = None
            result = do_health(cfg)
            self.assertEqual(result["status"], "healthy")

    def test_unhealthy(self) -> None:
        cfg = SmakConfig()
        with _patch_doctor() as df:
            df.return_value.validate_all.side_effect = RuntimeError("bad\nworse")
            result = do_health(cfg)
            self.assertEqual(result["status"], "unhealthy")
            self.assertEqual(len(result["issues"]), 2)


class TestDoGraphStats(unittest.TestCase):
    def test_delegates(self) -> None:
        cfg = SmakConfig()
        with _patch_doctor() as df:
            df.return_value.graph_stats.return_value = {"by_index": {}}
            result = do_graph_stats(cfg)
            self.assertIn("by_index", result)


if __name__ == "__main__":
    unittest.main()
