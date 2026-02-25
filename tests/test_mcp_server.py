from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from smak.mcp_server import SmakMcpServer, build_mcp_server


class TestMcpServer(unittest.TestCase):
    def test_refresh_knowledge_uses_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = SmakMcpServer(workspace_root=Path(tmp_dir))
            with patch.object(server, "_run_cli", return_value="ok") as run_cli:
                output = server.refresh_knowledge(folder="src", index="source_code")
            self.assertEqual(output, "ok")
            self.assertIn("ingest", run_cli.call_args.args[0])

    def test_semantic_search_parses_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = SmakMcpServer(workspace_root=Path(tmp_dir))
            with patch.object(server, "_run_cli", return_value='{"hits":[],"related_context":[]}'):
                result = server.semantic_search("auth")
            self.assertIn("hits", result)

    def test_manage_sidecar_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = SmakMcpServer(workspace_root=Path(tmp_dir))
            with patch.object(server, "_run_cli", return_value='["a.py::A"]'):
                symbols = server.manage_sidecar(action="inspect", file_path="a.py")
            self.assertEqual(symbols, ["a.py::A"])

    def test_manage_sidecar_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = SmakMcpServer(workspace_root=Path(tmp_dir))
            with patch.object(server, "_run_cli", return_value='{"applied_updates":1}'):
                result = server.manage_sidecar(
                    action="update", file_path="src/a.py", updates=[{"symbol": "x"}]
                )
            self.assertEqual(result["applied_updates"], 1)


    def test_run_cli_forces_utf8_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = SmakMcpServer(workspace_root=Path(tmp_dir))
            completed = MagicMock(returncode=0, stdout="ok", stderr="")
            with patch("smak.mcp_server.subprocess.run", return_value=completed) as run_mock:
                output = server._run_cli(["query", "hello"])
            self.assertEqual(output, "ok")
            kwargs = run_mock.call_args.kwargs
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")

    def test_build_mcp_server_returns_sdk_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mcp = build_mcp_server(tmp_dir)
            self.assertEqual(mcp.name, "SMAK")


if __name__ == "__main__":
    unittest.main()
