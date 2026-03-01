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
        registry_path = tmp_path / "registry.yaml"
        registry_path.write_text(
            """
workspaces:
  mock_workspace:
    path: "./workspace"
    description: "Mock workspace"
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return SmakMcpServer(registry_path=registry_path)

    def test_init_requires_registry_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.yaml"
            with self.assertRaises(FileNotFoundError):
                SmakMcpServer(registry_path=missing)

    def test_init_requires_non_empty_workspaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir) / "registry.yaml"
            registry_path.write_text("workspaces: {}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                SmakMcpServer(registry_path=registry_path)

    def test_get_workspace_cwd_resolves_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            cwd = server._get_workspace_cwd("mock_workspace")
            self.assertTrue(cwd.is_absolute())
            self.assertEqual(cwd, Path(tmp_dir).resolve() / "workspace")

    def test_get_workspace_cwd_rejects_unknown_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            with self.assertRaises(ValueError):
                server._get_workspace_cwd("unknown")

    def test_refresh_knowledge_uses_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            with patch.object(server, "_run_cli", return_value="ok") as run_cli:
                output = server.refresh_knowledge(
                    workspace="mock_workspace", folder="src", index="source_code"
                )
            self.assertEqual(output, "ok")
            self.assertIn("ingest", run_cli.call_args.args[0])
            self.assertEqual(run_cli.call_args.kwargs["workspace"], "mock_workspace")

    def test_refresh_knowledge_can_disable_follow_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            with patch.object(server, "_run_cli", return_value="ok") as run_cli:
                server.refresh_knowledge(
                    workspace="mock_workspace",
                    folder="src",
                    index="source_code",
                    follow_symlinks=False,
                )
            self.assertIn("--no-follow-symlinks", run_cli.call_args.args[0])

    def test_semantic_search_parses_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            with patch.object(server, "_run_cli", return_value='{"hits":[],"related_context":[]}'):
                result = server.semantic_search(workspace="mock_workspace", query="auth")
            self.assertIn("hits", result)

    def test_manage_sidecar_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            with patch.object(server, "_run_cli", return_value='["a.py::A"]'):
                symbols = server.manage_sidecar(
                    workspace="mock_workspace", action="inspect", file_path="a.py"
                )
            self.assertEqual(symbols, ["a.py::A"])

    def test_manage_sidecar_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            with patch.object(server, "_run_cli", return_value='{"applied_updates":1}'):
                result = server.manage_sidecar(
                    workspace="mock_workspace",
                    action="update",
                    file_path="src/a.py",
                    updates=[{"symbol": "x"}],
                )
            self.assertEqual(result["applied_updates"], 1)

    def test_validate_mesh_uses_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            with patch.object(server, "_run_cli", return_value="ok") as run_cli:
                output = server.validate_mesh(workspace="mock_workspace", path=".")
            self.assertEqual(output, "ok")
            self.assertEqual(run_cli.call_args.kwargs["workspace"], "mock_workspace")

    def test_run_cli_forces_utf8_environment_and_workspace_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            completed = MagicMock(returncode=0, stdout="ok", stderr="")
            with patch("smak.mcp_server.subprocess.run", return_value=completed) as run_mock:
                output = server._run_cli(["query", "hello"], workspace="mock_workspace")
            self.assertEqual(output, "ok")
            kwargs = run_mock.call_args.kwargs
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
            self.assertEqual(kwargs["cwd"], Path(tmp_dir).resolve() / "workspace")

    def test_build_mcp_server_returns_sdk_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = self._create_server(tmp_dir)
            mcp = build_mcp_server(server.registry_path)
            self.assertEqual(mcp.name, "SMAK")


if __name__ == "__main__":
    unittest.main()
