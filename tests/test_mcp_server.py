from __future__ import annotations

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
            "    paths:\n"
            "      - ./src\n"
            + extra_yaml,
            encoding="utf-8",
        )
        return str(config_path)

    # ------------------------------------------------------------------
    # Core: init, config caching
    # ------------------------------------------------------------------

    def test_init_no_args(self) -> None:
        server = SmakMcpServer()
        self.assertIsNotNone(server)

    def test_load_config_raises_on_missing_file(self) -> None:
        server = SmakMcpServer()
        with self.assertRaises(FileNotFoundError):
            server._load_config("/nonexistent/path.yaml")

    def test_load_config_caches_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            cfg1 = server._load_config(config)
            cfg2 = server._load_config(config)
            self.assertIs(cfg1, cfg2)

    def test_server_stores_custom_embedding_config(self) -> None:
        emb_cfg = EmbeddingConfig(api_base="http://custom:8888")
        server = SmakMcpServer(embedding_config=emb_cfg)
        self.assertEqual(server.embedding_config.api_base, "http://custom:8888")

    # ------------------------------------------------------------------
    # describe_workspace
    # ------------------------------------------------------------------

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    def test_describe_workspace_returns_indices(self, _: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()

            result = server.describe_workspace(config=config)

            self.assertIn("indices", result)
            self.assertEqual(len(result["indices"]), 1)
            self.assertEqual(result["indices"][0]["name"], "source_code")
            self.assertEqual(result["indices"][0]["description"], "src")
            self.assertIn("config_path", result)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store", return_value=object())
    @patch("smak.mcp_server.create_query_service")
    def test_search_calls_query_service(
        self,
        query_factory: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            query_factory.return_value.search.return_value = {
                "hits": [], "related_context": [],
            }

            result = server.search(config=config, query="auth", index="source_code")

            self.assertEqual(result, {"hits": [], "related_context": []})
            query_factory.return_value.search.assert_called_once_with("auth", top_k=5)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store", return_value=object())
    @patch("smak.mcp_server.create_query_service")
    def test_search_forwards_embedding_config(
        self,
        query_factory: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            query_factory.return_value.search.return_value = {}

            server.search(config=config, query="test", index="source_code")

            call_kwargs = query_factory.call_args[1]
            self.assertIn("embedding_config", call_kwargs)

    # ------------------------------------------------------------------
    # search_all
    # ------------------------------------------------------------------

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store", return_value=object())
    @patch("smak.mcp_server.create_query_service")
    def test_search_all_queries_all_indices(
        self,
        query_factory: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(
                tmp_dir,
                "  - name: issues\n    description: bugs\n    paths:\n      - ./src\n",
            )
            server = SmakMcpServer()
            query_factory.return_value.search.return_value = {
                "hits": [], "related_context": [],
            }

            result = server.search_all(config=config, query="auth logic")

            self.assertIn("source_code", result)
            self.assertIn("issues", result)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store", return_value=object())
    @patch("smak.mcp_server.create_query_service")
    def test_search_all_filters_by_indices(
        self,
        query_factory: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(
                tmp_dir,
                "  - name: issues\n    description: bugs\n    paths:\n      - ./src\n",
            )
            server = SmakMcpServer()
            query_factory.return_value.search.return_value = {
                "hits": [], "related_context": [],
            }

            result = server.search_all(
                config=config, query="auth", indices=["source_code"],
            )

            self.assertIn("source_code", result)
            self.assertNotIn("issues", result)

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store", return_value=object())
    @patch("smak.mcp_server.create_query_service")
    def test_lookup_delegates_to_query_service(
        self,
        query_factory: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            query_factory.return_value.lookup.return_value = {
                "found": True, "uid": "a::b",
            }

            result = server.lookup(config=config, uid="a::b", index="source_code")

            self.assertTrue(result["found"])
            query_factory.return_value.lookup.assert_called_once_with("a::b")

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store")
    @patch("smak.mcp_server.IngestService")
    def test_ingest_uses_ingest_service(
        self,
        ingest_cls: MagicMock,
        load_store: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            load_store.return_value = object()
            ingest_instance = ingest_cls.return_value
            ingest_instance.ingest_paths.return_value = MagicMock(
                files=1, skipped=0, vectors=2,
            )

            output = server.ingest(config=config, index="source_code")

            self.assertIn("Ingestion Complete", output)
            ingest_cls.assert_called_once()

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store")
    @patch("smak.mcp_server.IngestService")
    def test_ingest_passes_embedder_loader(
        self,
        ingest_cls: MagicMock,
        load_store: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            load_store.return_value = object()
            ingest_instance = ingest_cls.return_value
            ingest_instance.ingest_paths.return_value = MagicMock(
                files=0, skipped=0, vectors=0,
            )

            server.ingest(config=config)

            call_kwargs = ingest_instance.ingest_paths.call_args[1]
            self.assertIn("embedder_loader", call_kwargs)
            self.assertTrue(callable(call_kwargs["embedder_loader"]))

    # ------------------------------------------------------------------
    # enrich_symbol — the composite sidecar tool
    # ------------------------------------------------------------------

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_enrich_symbol_updates_intent_and_relations(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            source = Path(tmp_dir) / "src" / "a.py"
            source.write_text("def hello():\n    pass\n", encoding="utf-8")

            svc = sidecar_factory.return_value
            svc.inspect.return_value = ["hello"]
            svc.update.return_value = {"total_symbols": 1}

            result = server.enrich_symbol(
                config=config,
                file_path="a.py",
                symbol="hello",
                intent="greets user",
                relations=["issue:1"],
                index="source_code",
            )

            self.assertEqual(result["status"], "ok")
            # enrich_symbol calls update twice: full sync then single-symbol
            self.assertEqual(svc.update.call_count, 2)
            svc.update.assert_any_call(source)  # full sync
            svc.update.assert_any_call(
                source,
                symbol="hello",
                intent="greets user",
                relations=["issue:1"],
            )

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_enrich_symbol_rejects_unknown_symbol(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            source = Path(tmp_dir) / "src" / "a.py"
            source.write_text("def hello():\n    pass\n", encoding="utf-8")

            svc = sidecar_factory.return_value
            svc.inspect.return_value = ["hello"]

            result = server.enrich_symbol(
                config=config,
                file_path="a.py",
                symbol="nonexistent",
                intent="oops",
                index="source_code",
            )

            self.assertEqual(result["status"], "error")
            self.assertIn("nonexistent", result["message"])
            self.assertIn("valid_symbols", result)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_enrich_symbol_auto_clears_stale_on_retry(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        """When update fails because of stale symbols, enrich_symbol clears them and retries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            source = Path(tmp_dir) / "src" / "a.py"
            source.write_text("def hello():\n    pass\n", encoding="utf-8")

            svc = sidecar_factory.return_value
            svc.inspect.return_value = ["hello"]
            # First call to full sync fails, second succeeds
            svc.update.side_effect = [
                ValueError('Cannot remove symbols with existing relations. Clear them first:\n  smak sidecar clear a.py --symbol "old_func"'),
                {"total_symbols": 1},  # full sync after clear
                {"total_symbols": 1},  # single symbol update
            ]
            svc.clear_symbol.return_value = {"cleared_symbol": "old_func"}

            result = server.enrich_symbol(
                config=config,
                file_path="a.py",
                symbol="hello",
                intent="greets",
                index="source_code",
            )

            self.assertEqual(result["status"], "ok")
            svc.clear_symbol.assert_called()

    # ------------------------------------------------------------------
    # enrich_file
    # ------------------------------------------------------------------

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_enrich_file_does_full_sync(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            source = Path(tmp_dir) / "src" / "a.py"
            source.write_text("def hello():\n    pass\n", encoding="utf-8")

            svc = sidecar_factory.return_value
            svc.update.return_value = {
                "total_symbols": 1, "added": 1, "removed": 0,
            }

            result = server.enrich_file(
                config=config,
                file_path="a.py",
                index="source_code",
            )

            self.assertEqual(result["total_symbols"], 1)
            svc.update.assert_called_once_with(source)

    # ------------------------------------------------------------------
    # enrich_batch
    # ------------------------------------------------------------------

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_enrich_batch_processes_multiple_files(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            sidecar_factory.return_value.update.return_value = {"total_symbols": 1}

            for name in ("a.py", "b.py"):
                f = Path(tmp_dir) / "src" / name
                f.write_text("x = 1\n", encoding="utf-8")

            result = server.enrich_batch(
                config=config,
                file_paths=["a.py", "b.py"],
                index="source_code",
            )

            self.assertEqual(len(result), 2)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_enrich_batch_continues_on_error(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            sidecar_factory.return_value.update.side_effect = [
                ValueError("fail"), {"total_symbols": 1},
            ]

            for name in ("a.py", "b.py"):
                f = Path(tmp_dir) / "src" / name
                f.write_text("x = 1\n", encoding="utf-8")

            result = server.enrich_batch(
                config=config,
                file_paths=["a.py", "b.py"],
                index="source_code",
            )

            self.assertEqual(len(result), 2)
            self.assertIn("error", result[0])
            self.assertIn("total_symbols", result[1])

    # ------------------------------------------------------------------
    # check_health
    # ------------------------------------------------------------------

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_doctor_service")
    def test_check_health_passes(
        self,
        doctor_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            doctor_factory.return_value.validate_all.return_value = None

            result = server.check_health(config=config)

            self.assertEqual(result["status"], "healthy")
            doctor_factory.return_value.validate_all.assert_called_once()

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_doctor_service")
    def test_check_health_reports_issues(
        self,
        doctor_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._write_config(tmp_dir)
            server = SmakMcpServer()
            doctor_factory.return_value.validate_all.side_effect = RuntimeError(
                "Orphaned sidecar: x.yaml"
            )

            result = server.check_health(config=config)

            self.assertEqual(result["status"], "unhealthy")
            self.assertIn("Orphaned sidecar", result["issues"][0])

    # ------------------------------------------------------------------
    # Path resolution (internal, reused by tools)
    # ------------------------------------------------------------------

    def test_resolve_source_path_falls_back_to_unique_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            candidate = root / "src" / "csv_editor.py"
            candidate.parent.mkdir(parents=True)
            candidate.write_text("# mock\n", encoding="utf-8")
            index_config = SimpleNamespace(paths=[str(root)])
            server = SmakMcpServer()

            resolved = server._resolve_source_path(
                index="source_code",
                index_config=index_config,
                file_path="csv_editor.py",
            )
            self.assertEqual(resolved, candidate)

    def test_resolve_source_path_raises_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for sub in ("src", "tests"):
                d = root / sub
                d.mkdir(parents=True)
                (d / "csv_editor.py").write_text("# x\n", encoding="utf-8")
            index_config = SimpleNamespace(paths=[str(root)])
            server = SmakMcpServer()

            with self.assertRaises(ValueError) as cm:
                server._resolve_source_path(
                    index="source_code",
                    index_config=index_config,
                    file_path="csv_editor.py",
                )
            self.assertIn("Ambiguous", str(cm.exception))

    def test_resolve_source_path_raises_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_config = SimpleNamespace(paths=[str(Path(tmp_dir))])
            server = SmakMcpServer()

            with self.assertRaises(FileNotFoundError) as cm:
                server._resolve_source_path(
                    index="source_code",
                    index_config=index_config,
                    file_path="missing.py",
                )
            self.assertIn("semantic_search", str(cm.exception))

    # ------------------------------------------------------------------
    # build_mcp_server
    # ------------------------------------------------------------------

    def test_build_mcp_server_returns_sdk_server(self) -> None:
        mcp = build_mcp_server()
        self.assertEqual(mcp.name, "SMAK")

    def test_build_mcp_server_accepts_embedding_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            emb_yaml = Path(tmp_dir) / "custom_emb.yaml"
            emb_yaml.write_text("api_base: http://custom:7777\n", encoding="utf-8")
            mcp = build_mcp_server(embedding_setup=emb_yaml)
            self.assertEqual(mcp.name, "SMAK")


if __name__ == "__main__":
    unittest.main()
