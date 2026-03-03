from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
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
            ingest_instance.ingest_folder.return_value = MagicMock(files=1, skipped=0, vectors=2)

            output = server.refresh_knowledge(
                config="mock_config",
                index="source_code",
            )

            self.assertIn("Ingestion Complete", output)
            ingest_cls.assert_called_once()
            ingest_instance.ingest_folder.assert_called_once()

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
    def test_manage_sidecar_inspect(
        self,
        sidecar_cls: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            sidecar_cls.return_value.inspect.return_value = ["a.py::A"]

            symbols = server.manage_sidecar(
                config="mock_config",
                action="inspect",
                file_path="a.py",
            )

            self.assertEqual(symbols, ["a.py::A"])

    @patch("smak.mcp_server.initialize_embedding_dimensions", side_effect=lambda cfg, _: cfg)
    @patch("smak.mcp_server.SidecarService")
    def test_manage_sidecar_update(
        self,
        sidecar_cls: MagicMock,
        _: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            sidecar_cls.return_value.update.return_value = {"applied_updates": 1}

            result = server.manage_sidecar(
                config="mock_config",
                action="update",
                file_path="a.py",
                updates=[{"symbol": "x"}],
            )

            self.assertEqual(result["applied_updates"], 1)

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
