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
    def _create_server(self, tmp_dir: str, extra_yaml: str = "") -> SmakMcpServer:
        """Create a server with direct config path (no registry)."""
        tmp_path = Path(tmp_dir)
        src_dir = tmp_path / "src"
        src_dir.mkdir(exist_ok=True)
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
        return SmakMcpServer(config_path=config_path)

    def test_init_requires_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.yaml"
            with self.assertRaises(FileNotFoundError):
                SmakMcpServer(config_path=missing)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store")
    @patch("smak.mcp_server.IngestService")
    def test_refresh_knowledge_uses_ingest_service(
        self,
        ingest_cls: MagicMock,
        load_store: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            load_store.return_value = object()
            ingest_instance = ingest_cls.return_value
            ingest_instance.ingest_paths.return_value = MagicMock(files=1, skipped=0, vectors=2)

            output = server.refresh_knowledge(index="source_code")

            self.assertIn("Ingestion Complete", output)
            ingest_cls.assert_called_once()
            ingest_instance.ingest_paths.assert_called_once()

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    def test_list_available_indices_returns_name_and_description(
        self,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)

            indices = server.list_available_indices()

            self.assertEqual(
                indices,
                [{"name": "source_code", "description": "src"}],
            )

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store", return_value=object())
    @patch("smak.mcp_server.create_query_service")
    def test_semantic_search_calls_query_service(
        self,
        query_factory: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            query_factory.return_value.search.return_value = {"hits": [], "related_context": []}

            result = server.semantic_search(query="auth")

            self.assertEqual(result, {"hits": [], "related_context": []})
            query_factory.return_value.search.assert_called_once_with("auth", top_k=5)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_inspect_sidecar_uses_resolver(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            sidecar_factory.return_value.inspect.return_value = ["a.py::A"]
            source_file = Path(tmp_dir) / "src" / "a.py"
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text("print('ok')\n", encoding="utf-8")

            symbols = server.inspect_sidecar(file_path="a.py")

            self.assertEqual(symbols, ["a.py::A"])
            sidecar_factory.return_value.inspect.assert_called_once_with(source_file)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_update_sidecar_uses_resolver(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            sidecar_factory.return_value.update.return_value = {"total_symbols": 1}
            source_file = Path(tmp_dir) / "src" / "a.py"
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text("print('ok')\n", encoding="utf-8")

            result = server.update_sidecar(
                file_path="a.py",
                symbol="x",
                intent="test intent",
            )

            self.assertEqual(result["total_symbols"], 1)
            sidecar_factory.return_value.update.assert_called_once_with(
                source_file,
                symbol="x",
                intent="test intent",
                relations=None,
            )

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_clear_sidecar_symbol_uses_resolver(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            sidecar_factory.return_value.clear_symbol.return_value = {
                "cleared_symbol": "x",
                "remaining_symbols": 0,
            }
            source_file = Path(tmp_dir) / "src" / "a.py"
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_text("print('ok')\n", encoding="utf-8")

            result = server.clear_sidecar_symbol(
                file_path="a.py",
                symbol="x",
            )

            self.assertEqual(result["cleared_symbol"], "x")
            sidecar_factory.return_value.clear_symbol.assert_called_once_with(
                source_file, "x"
            )

    def test_resolve_source_path_falls_back_to_unique_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            candidate = root / "src" / "csv_editor.py"
            candidate.parent.mkdir(parents=True)
            candidate.write_text("# mock\n", encoding="utf-8")
            index_config = SimpleNamespace(paths=[str(root)])
            server = self._create_server(tmp_dir)

            resolved = server._resolve_source_path(
                index="source_code",
                index_config=index_config,
                file_path="csv_editor.py",
            )

            self.assertEqual(resolved, candidate)

    def test_resolve_source_path_raises_ambiguous_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "src" / "csv_editor.py"
            second = root / "tests" / "test_csv_editor.py"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text("# first\n", encoding="utf-8")
            second.write_text("# second\n", encoding="utf-8")
            index_config = SimpleNamespace(paths=[str(root)])
            server = self._create_server(tmp_dir)

            with self.assertRaises(ValueError) as cm:
                server._resolve_source_path(
                    index="source_code",
                    index_config=index_config,
                    file_path="csv_editor.py",
                )

            msg = str(cm.exception)
            self.assertIn("Ambiguous file path 'csv_editor.py'", msg)

    def test_resolve_source_path_raises_actionable_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            index_config = SimpleNamespace(paths=[str(root)])
            server = self._create_server(tmp_dir)

            with self.assertRaises(FileNotFoundError) as cm:
                server._resolve_source_path(
                    index="source_code",
                    index_config=index_config,
                    file_path="missing.py",
                )

            msg = str(cm.exception)
            self.assertIn("index='source_code'", msg)
            self.assertIn("file_path='missing.py'", msg)
            self.assertIn("semantic_search", msg)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_doctor_service")
    def test_validate_mesh_uses_doctor_service(
        self,
        doctor_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            doctor_instance = doctor_factory.return_value

            output = server.validate_mesh()

            self.assertEqual(output, "Mesh diagnostics passed.")
            doctor_instance.validate_all.assert_called_once()

    def test_build_mcp_server_returns_sdk_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            mcp = build_mcp_server(config_path=server.config_path)
            self.assertEqual(mcp.name, "SMAK")

    def test_server_stores_custom_embedding_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "src").mkdir()
            config_path = tmp_path / "workspace_config.yaml"
            config_path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: src\n"
                "    paths:\n"
                "      - ./src\n",
                encoding="utf-8",
            )
            emb_cfg = EmbeddingConfig(api_base="http://custom:8888")
            server = SmakMcpServer(
                config_path=config_path,
                embedding_config=emb_cfg,
            )
            self.assertEqual(server.embedding_config.api_base, "http://custom:8888")

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store")
    @patch("smak.mcp_server.IngestService")
    def test_refresh_knowledge_passes_embedder_loader(
        self,
        ingest_cls: MagicMock,
        load_store: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            load_store.return_value = object()
            ingest_instance = ingest_cls.return_value
            ingest_instance.ingest_paths.return_value = MagicMock(
                files=0, skipped=0, vectors=0,
            )

            server.refresh_knowledge()

            call_kwargs = ingest_instance.ingest_paths.call_args[1]
            self.assertIn("embedder_loader", call_kwargs)
            self.assertTrue(callable(call_kwargs["embedder_loader"]))

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store", return_value=object())
    @patch("smak.mcp_server.create_query_service")
    def test_semantic_search_forwards_embedding_config(
        self,
        query_factory: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            query_factory.return_value.search.return_value = {}

            server.semantic_search(query="test")

            call_kwargs = query_factory.call_args[1]
            self.assertIn("embedding_config", call_kwargs)

    def test_build_mcp_server_accepts_embedding_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            emb_yaml = Path(tmp_dir) / "custom_emb.yaml"
            emb_yaml.write_text(
                "api_base: http://custom:7777\n",
                encoding="utf-8",
            )
            server = self._create_server(tmp_dir)
            mcp = build_mcp_server(server.config_path, embedding_setup=emb_yaml)
            self.assertEqual(mcp.name, "SMAK")

    # --- Composite tool tests ---

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store", return_value=object())
    @patch("smak.mcp_server.create_query_service")
    def test_multi_index_search_queries_all_indices(
        self,
        query_factory: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(
                tmp_dir,
                "  - name: issues\n    description: bugs\n    paths:\n      - ./src\n",
            )
            query_factory.return_value.search.return_value = {
                "hits": [],
                "related_context": [],
            }

            result = server.multi_index_search(query="auth logic")

            self.assertIn("source_code", result)
            self.assertIn("issues", result)
            self.assertIn("hits", result["source_code"])

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.load_and_validate_vector_store", return_value=object())
    @patch("smak.mcp_server.create_query_service")
    def test_multi_index_search_filters_by_indices_param(
        self,
        query_factory: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(
                tmp_dir,
                "  - name: issues\n    description: bugs\n    paths:\n      - ./src\n",
            )
            query_factory.return_value.search.return_value = {
                "hits": [],
                "related_context": [],
            }

            result = server.multi_index_search(query="auth", indices=["source_code"])

            self.assertIn("source_code", result)
            self.assertNotIn("issues", result)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    def test_workspace_status_returns_per_index_stats(
        self,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)

            result = server.workspace_status()

            self.assertIn("indices", result)
            self.assertEqual(len(result["indices"]), 1)
            self.assertEqual(result["indices"][0]["name"], "source_code")

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_batch_update_sidecars_syncs_multiple_files(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            sidecar_factory.return_value.update.return_value = {"total_symbols": 1}

            for name in ("a.py", "b.py"):
                f = Path(tmp_dir) / "src" / name
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("x = 1\n", encoding="utf-8")

            result = server.batch_update_sidecars(
                file_paths=["a.py", "b.py"],
                index="source_code",
            )

            self.assertEqual(len(result), 2)
            self.assertEqual(sidecar_factory.return_value.update.call_count, 2)

    @patch("smak.mcp_server.init_config", side_effect=lambda cfg, **kw: cfg)
    @patch("smak.mcp_server.create_sidecar_service")
    def test_batch_update_sidecars_continues_on_error(
        self,
        sidecar_factory: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            sidecar_factory.return_value.update.side_effect = [
                ValueError("parse error"),
                {"total_symbols": 1},
            ]

            for name in ("a.py", "b.py"):
                f = Path(tmp_dir) / "src" / name
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("x = 1\n", encoding="utf-8")

            result = server.batch_update_sidecars(
                file_paths=["a.py", "b.py"],
                index="source_code",
            )

            self.assertEqual(len(result), 2)
            self.assertIn("error", result[0])
            self.assertIn("total_symbols", result[1])


if __name__ == "__main__":
    unittest.main()
