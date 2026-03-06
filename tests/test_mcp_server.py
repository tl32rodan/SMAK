from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    from smak.mcp_server import SmakMcpServer, build_mcp_server
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependency guard
    SmakMcpServer = None
    build_mcp_server = None
    _MCP_IMPORT_ERROR = exc
else:
    _MCP_IMPORT_ERROR = None


@unittest.skipIf(_MCP_IMPORT_ERROR is not None, f"Missing dependency: {_MCP_IMPORT_ERROR}")
class TestMcpServer(unittest.TestCase):
    def _create_server(self, tmp_dir: str) -> SmakMcpServer:
        tmp_path = Path(tmp_dir)
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "workspace_config.yaml").write_text(
            "indices:\n"
            "  - name: source_code\n"
            "    description: src\n",
            encoding="utf-8",
        )
        registry_path = tmp_path / "registry.yaml"
        registry_path.write_text(
            "\n".join(
                [
                    "configs:",
                    "  mock_config:",
                    '    config_path: "./workspace/workspace_config.yaml"',
                    '    description: "Mock config"',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return SmakMcpServer(registry_path=registry_path)

    def test_init_requires_registry_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.yaml"
            with self.assertRaises(FileNotFoundError):
                SmakMcpServer(registry_path=missing)

    def test_init_requires_non_empty_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir) / "registry.yaml"
            registry_path.write_text("configs:\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                SmakMcpServer(registry_path=registry_path)

    def test_resolve_config_path_returns_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            config_path = server._resolve_config_path("mock_config")
            self.assertTrue(config_path.is_absolute())
            self.assertEqual(
                config_path,
                Path(tmp_dir).resolve() / "workspace" / "workspace_config.yaml",
            )

    def test_resolve_config_path_rejects_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            with self.assertRaises(ValueError):
                server._resolve_config_path("unknown")

    @patch("smak.mcp_server.initialize_embedding_dimensions", side_effect=lambda cfg, _: cfg)
    @patch("smak.mcp_server._load_vector_store")
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

            output = server.refresh_knowledge(
                config="mock_config",
                index="source_code",
            )

            self.assertIn("Ingestion Complete", output)
            ingest_cls.assert_called_once()
            ingest_instance.ingest_paths.assert_called_once()

    @patch("smak.mcp_server.initialize_embedding_dimensions", side_effect=lambda cfg, _: cfg)
    def test_list_available_indices_returns_name_and_description(
        self,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)

            indices = server.list_available_indices(config="mock_config")

            self.assertEqual(
                indices,
                [{"name": "source_code", "description": "src"}],
            )

    @patch("smak.mcp_server.initialize_embedding_dimensions", side_effect=lambda cfg, _: cfg)
    @patch("smak.mcp_server._load_vector_store", return_value=object())
    @patch("smak.mcp_server.QueryService")
    def test_semantic_search_calls_query_service(
        self,
        query_cls: MagicMock,
        _: MagicMock,
        __: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            query_cls.return_value.search.return_value = {"hits": [], "related_context": []}

            result = server.semantic_search(config="mock_config", query="auth")

            self.assertEqual(result, {"hits": [], "related_context": []})
            query_cls.return_value.search.assert_called_once_with("auth", top_k=5)

    @patch("smak.mcp_server.initialize_embedding_dimensions", side_effect=lambda cfg, _: cfg)
    @patch("smak.mcp_server.SidecarService")
    def test_inspect_sidecar_uses_resolver(
        self,
        sidecar_cls: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            sidecar_cls.return_value.inspect.return_value = ["a.py::A"]
            source_file = Path(tmp_dir) / "workspace" / "src" / "a.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("print('ok')\n", encoding="utf-8")

            symbols = server.inspect_sidecar(
                config="mock_config",
                file_path="a.py",
            )

            self.assertEqual(symbols, ["a.py::A"])
            sidecar_cls.return_value.inspect.assert_called_once_with(source_file)

    @patch("smak.mcp_server.initialize_embedding_dimensions", side_effect=lambda cfg, _: cfg)
    @patch("smak.mcp_server.SidecarService")
    def test_update_sidecar_uses_resolver(
        self,
        sidecar_cls: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            sidecar_cls.return_value.update.return_value = {"applied_updates": 1}
            source_file = Path(tmp_dir) / "workspace" / "src" / "a.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("print('ok')\n", encoding="utf-8")
            updates = [{"symbol": "x"}]

            result = server.update_sidecar(
                config="mock_config",
                file_path="a.py",
                updates=updates,
            )

            self.assertEqual(result["applied_updates"], 1)
            sidecar_cls.return_value.update.assert_called_once_with(
                source_file,
                '[{"symbol": "x"}]',
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
                config_name="mock_config",
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
                    config_name="mock_config",
                    index="source_code",
                    index_config=index_config,
                    file_path="csv_editor.py",
                )

            msg = str(cm.exception)
            self.assertIn("Ambiguous file path 'csv_editor.py'", msg)
            self.assertIn("src/csv_editor.py", msg)
            self.assertIn("tests/test_csv_editor.py", msg)

    def test_resolve_source_path_raises_actionable_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            index_config = SimpleNamespace(paths=[str(root)])
            server = self._create_server(tmp_dir)

            with self.assertRaises(FileNotFoundError) as cm:
                server._resolve_source_path(
                    config_name="mock_config",
                    index="source_code",
                    index_config=index_config,
                    file_path="missing.py",
                )

            msg = str(cm.exception)
            self.assertIn("config='mock_config'", msg)
            self.assertIn("index='source_code'", msg)
            self.assertIn("file_path='missing.py'", msg)
            self.assertIn(str(root), msg)
            self.assertIn("semantic_search", msg)

    @patch("smak.mcp_server.initialize_embedding_dimensions", side_effect=lambda cfg, _: cfg)
    @patch("smak.mcp_server.DoctorService")
    def test_validate_mesh_uses_doctor_service(
        self,
        doctor_cls: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            doctor_instance = doctor_cls.return_value

            output = server.validate_mesh(config="mock_config")

            self.assertEqual(output, "Mesh diagnostics passed.")
            doctor_instance.validate_all.assert_called_once()

    def test_build_mcp_server_returns_sdk_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            mcp = build_mcp_server(server.registry_path)
            self.assertEqual(mcp.name, "SMAK")


if __name__ == "__main__":
    unittest.main()
