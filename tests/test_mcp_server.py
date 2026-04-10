from __future__ import annotations

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    from smak.config import EmbeddingConfig
    from smak.mcp_server import SmakMcpServer, build_mcp_server
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependency guard
    EmbeddingConfig = None
    SmakMcpServer = None
    build_mcp_server = None
    _MCP_IMPORT_ERROR = exc
else:
    _MCP_IMPORT_ERROR = None


# ------------------------------------------------------------------
# Shared patch stacks — eliminates repeated @patch decorators
# ------------------------------------------------------------------

def _patch_init_config():
    return patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)


def _patch_query_stack():
    """Patch stack for search / lookup / search_all tests."""
    return contextlib.ExitStack()


def _query_patches():
    """Return three patch context managers for query-related tests."""
    return (
        _patch_init_config(),
        patch("smak.core_ops.load_and_validate_vector_store", return_value=object()),
        patch("smak.core_ops.create_query_service"),
    )


def _sidecar_patches():
    """Return two patch context managers for sidecar-related tests."""
    return (
        _patch_init_config(),
        patch("smak.core_ops.create_sidecar_service"),
    )


def _ingest_patches():
    """Return three patch context managers for ingest-related tests."""
    return (
        _patch_init_config(),
        patch("smak.core_ops.load_and_validate_vector_store"),
        patch("smak.core_ops.IngestService"),
    )


@unittest.skipIf(_MCP_IMPORT_ERROR is not None, f"Missing dependency: {_MCP_IMPORT_ERROR}")
class TestMcpServer(unittest.TestCase):
    """Tests for the intent-based MCP tool set."""

    def _write_config(self, tmp_dir: str, extra_yaml: str = "") -> str:
        tmp_path = Path(tmp_dir)
        (tmp_path / "src").mkdir(exist_ok=True)
        config_path = tmp_path / "workspace_config.yaml"
        config_path.write_text(
            "indices:\n"
            "  - name: source_code\n"
            "    description: src\n"
            "    uri: /tmp/smak_test_data/source_code\n"
            "    paths:\n"
            "      - ./src\n"
            + extra_yaml,
            encoding="utf-8",
        )
        return str(config_path)

    def _server_and_config(self, tmp_dir: str, extra_yaml: str = "") -> tuple:
        """Create server + config in one call."""
        return SmakMcpServer(), self._write_config(tmp_dir, extra_yaml)

    def _write_source(self, tmp_dir: str, name: str = "a.py", content: str = "def hello():\n    pass\n") -> Path:
        """Write a source file under src/ and return its path."""
        source = Path(tmp_dir) / "src" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")
        return source

    # ------------------------------------------------------------------
    # Core: init, config caching
    # ------------------------------------------------------------------

    def test_init_no_args(self) -> None:
        self.assertIsNotNone(SmakMcpServer())

    def test_load_config_raises_on_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            SmakMcpServer()._load_config("/nonexistent/path.yaml")

    def test_load_config_caches_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            self.assertIs(server._load_config(config), server._load_config(config))

    def test_server_stores_custom_embedding_config(self) -> None:
        emb = EmbeddingConfig(api_base="http://custom:8888")
        self.assertEqual(SmakMcpServer(embedding_config=emb).embedding_config.api_base, "http://custom:8888")

    # ------------------------------------------------------------------
    # describe_workspace
    # ------------------------------------------------------------------

    def test_describe_workspace_returns_indices(self) -> None:
        with _patch_init_config(), tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            result = server.describe_workspace(config=config)

            self.assertIn("indices", result)
            self.assertEqual(result["indices"][0]["name"], "source_code")
            self.assertIn("config_path", result)

    # ------------------------------------------------------------------
    # search / search_all / lookup
    # ------------------------------------------------------------------

    def test_search_calls_query_service(self) -> None:
        p1, p2, p3 = _query_patches()
        with p1, p2, p3 as query_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            query_factory.return_value.search.return_value = {"hits": [], "related_context": []}

            result = server.search(config=config, query="auth", index="source_code")

            self.assertEqual(result, {"hits": [], "related_context": []})
            query_factory.return_value.search.assert_called_once_with("auth", top_k=5)

    def test_search_forwards_embedding_config(self) -> None:
        p1, p2, p3 = _query_patches()
        with p1, p2, p3 as query_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            query_factory.return_value.search.return_value = {}

            server.search(config=config, query="test", index="source_code")

            self.assertIn("embedding_config", query_factory.call_args[1])

    def test_search_all_queries_all_indices(self) -> None:
        p1, p2, p3 = _query_patches()
        with p1, p2, p3 as query_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(
                tmp_dir, "  - name: issues\n    description: bugs\n    uri: /tmp/smak_test_data/issues\n    paths:\n      - ./src\n",
            )
            query_factory.return_value.search.return_value = {"hits": [], "related_context": []}

            result = server.search_all(config=config, query="auth logic")

            self.assertIn("source_code", result)
            self.assertIn("issues", result)

    def test_search_all_filters_by_indices(self) -> None:
        p1, p2, p3 = _query_patches()
        with p1, p2, p3 as query_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(
                tmp_dir, "  - name: issues\n    description: bugs\n    uri: /tmp/smak_test_data/issues\n    paths:\n      - ./src\n",
            )
            query_factory.return_value.search.return_value = {"hits": [], "related_context": []}

            result = server.search_all(config=config, query="auth", indices=["source_code"])

            self.assertIn("source_code", result)
            self.assertNotIn("issues", result)

    def test_lookup_delegates_to_query_service(self) -> None:
        p1, p2, p3 = _query_patches()
        with p1, p2, p3 as query_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            query_factory.return_value.lookup.return_value = {"found": True, "uid": "a::b"}

            result = server.lookup(config=config, uid="a::b", index="source_code")

            self.assertTrue(result["found"])
            query_factory.return_value.lookup.assert_called_once_with("a::b")

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------

    def test_ingest_uses_ingest_service(self) -> None:
        p1, p2, p3 = _ingest_patches()
        with p1, p2 as load_store, p3 as ingest_cls, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            load_store.return_value = object()
            ingest_cls.return_value.ingest_paths.return_value = MagicMock(files=1, skipped=0, vectors=2)

            output = server.ingest(config=config, index="source_code")

            self.assertIn("Ingestion Complete", output)

    def test_ingest_passes_embedder_loader(self) -> None:
        p1, p2, p3 = _ingest_patches()
        with p1, p2 as load_store, p3 as ingest_cls, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            load_store.return_value = object()
            ingest_cls.return_value.ingest_paths.return_value = MagicMock(files=0, skipped=0, vectors=0)

            server.ingest(config=config)

            call_kwargs = ingest_cls.return_value.ingest_paths.call_args[1]
            self.assertIn("embedder_loader", call_kwargs)
            self.assertTrue(callable(call_kwargs["embedder_loader"]))

    # ------------------------------------------------------------------
    # enrich_symbol
    # ------------------------------------------------------------------

    def test_enrich_symbol_updates_intent_and_relations(self) -> None:
        p1, p2 = _sidecar_patches()
        with p1, p2 as sidecar_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            source = self._write_source(tmp_dir)
            svc = sidecar_factory.return_value
            svc.inspect.return_value = ["hello"]
            svc.update.return_value = {"total_symbols": 1}

            result = server.enrich_symbol(
                config=config, file_path="a.py", symbol="hello",
                intent="greets user", relations=["issue:1"], index="source_code",
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(svc.update.call_count, 2)
            svc.update.assert_any_call(source)
            svc.update.assert_any_call(source, symbol="hello", intent="greets user", relations=["issue:1"])

    def test_enrich_symbol_rejects_unknown_symbol(self) -> None:
        p1, p2 = _sidecar_patches()
        with p1, p2 as sidecar_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            self._write_source(tmp_dir)
            sidecar_factory.return_value.inspect.return_value = ["hello"]

            result = server.enrich_symbol(
                config=config, file_path="a.py", symbol="nonexistent",
                intent="oops", index="source_code",
            )

            self.assertEqual(result["status"], "error")
            self.assertIn("nonexistent", result["message"])
            self.assertIn("valid_symbols", result)

    def test_enrich_symbol_auto_clears_stale_on_retry(self) -> None:
        p1, p2 = _sidecar_patches()
        with p1, p2 as sidecar_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            self._write_source(tmp_dir)
            svc = sidecar_factory.return_value
            svc.inspect.return_value = ["hello"]
            svc.update.side_effect = [
                ValueError('Cannot remove symbols with existing relations. Clear them first:\n  smak sidecar clear a.py --symbol "old_func"'),
                {"total_symbols": 1},
                {"total_symbols": 1},
            ]
            svc.clear_symbol.return_value = {"cleared_symbol": "old_func"}

            result = server.enrich_symbol(
                config=config, file_path="a.py", symbol="hello",
                intent="greets", index="source_code",
            )

            self.assertEqual(result["status"], "ok")
            svc.clear_symbol.assert_called()

    # ------------------------------------------------------------------
    # enrich_file / enrich_batch
    # ------------------------------------------------------------------

    def test_enrich_file_does_full_sync(self) -> None:
        p1, p2 = _sidecar_patches()
        with p1, p2 as sidecar_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            source = self._write_source(tmp_dir)
            sidecar_factory.return_value.update.return_value = {"total_symbols": 1, "added": 1, "removed": 0}

            result = server.enrich_file(config=config, file_path="a.py", index="source_code")

            self.assertEqual(result["total_symbols"], 1)
            sidecar_factory.return_value.update.assert_called_once_with(source)

    def test_enrich_batch_processes_multiple_files(self) -> None:
        p1, p2 = _sidecar_patches()
        with p1, p2 as sidecar_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            sidecar_factory.return_value.update.return_value = {"total_symbols": 1}
            for name in ("a.py", "b.py"):
                self._write_source(tmp_dir, name=name, content="x = 1\n")

            result = server.enrich_batch(config=config, file_paths=["a.py", "b.py"], index="source_code")

            self.assertEqual(len(result), 2)

    def test_enrich_batch_continues_on_error(self) -> None:
        p1, p2 = _sidecar_patches()
        with p1, p2 as sidecar_factory, tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            sidecar_factory.return_value.update.side_effect = [ValueError("fail"), {"total_symbols": 1}]
            for name in ("a.py", "b.py"):
                self._write_source(tmp_dir, name=name, content="x = 1\n")

            result = server.enrich_batch(config=config, file_paths=["a.py", "b.py"], index="source_code")

            self.assertEqual(len(result), 2)
            self.assertIn("error", result[0])
            self.assertIn("total_symbols", result[1])

    # ------------------------------------------------------------------
    # check_health
    # ------------------------------------------------------------------

    def test_check_health_passes(self) -> None:
        with _patch_init_config(), patch("smak.core_ops.create_doctor_service") as doctor_factory, \
                tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            doctor_factory.return_value.validate_all.return_value = None

            result = server.check_health(config=config)

            self.assertEqual(result["status"], "healthy")

    def test_check_health_reports_issues(self) -> None:
        with _patch_init_config(), patch("smak.core_ops.create_doctor_service") as doctor_factory, \
                tempfile.TemporaryDirectory() as tmp_dir:
            server, config = self._server_and_config(tmp_dir)
            doctor_factory.return_value.validate_all.side_effect = RuntimeError("Orphaned sidecar: x.yaml")

            result = server.check_health(config=config)

            self.assertEqual(result["status"], "unhealthy")
            self.assertIn("Orphaned sidecar", result["issues"][0])

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def test_resolve_source_path_falls_back_to_unique_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            candidate = Path(tmp_dir) / "src" / "csv_editor.py"
            candidate.parent.mkdir(parents=True)
            candidate.write_text("# mock\n", encoding="utf-8")

            resolved = SmakMcpServer()._resolve_source_path(
                "source_code", SimpleNamespace(paths=[str(Path(tmp_dir))]), "csv_editor.py",
            )
            self.assertEqual(resolved, candidate)

    def test_resolve_source_path_raises_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for sub in ("src", "tests"):
                d = Path(tmp_dir) / sub
                d.mkdir(parents=True)
                (d / "csv_editor.py").write_text("# x\n", encoding="utf-8")

            with self.assertRaises(ValueError) as cm:
                SmakMcpServer()._resolve_source_path(
                    "source_code", SimpleNamespace(paths=[str(Path(tmp_dir))]), "csv_editor.py",
                )
            self.assertIn("Ambiguous", str(cm.exception))

    def test_resolve_source_path_raises_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError) as cm:
                SmakMcpServer()._resolve_source_path(
                    "source_code", SimpleNamespace(paths=[str(Path(tmp_dir))]), "missing.py",
                )
            self.assertIn("semantic_search", str(cm.exception))

    # ------------------------------------------------------------------
    # build_mcp_server
    # ------------------------------------------------------------------

    def test_build_mcp_server_returns_sdk_server(self) -> None:
        self.assertEqual(build_mcp_server().name, "SMAK")

    def test_build_mcp_server_accepts_embedding_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            emb_yaml = Path(tmp_dir) / "custom_emb.yaml"
            emb_yaml.write_text("api_base: http://custom:7777\n", encoding="utf-8")
            self.assertEqual(build_mcp_server(embedding_setup=emb_yaml).name, "SMAK")


if __name__ == "__main__":
    unittest.main()
