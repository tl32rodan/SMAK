from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smak.mcp_server import SmakMcpServer, build_mcp_server


class TestMcpServer(unittest.TestCase):
    def test_refresh_knowledge_uses_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = SmakMcpServer(workspace_root=Path(tmp_dir))
            with patch.object(server, "_run_cli", return_value="ok") as run_cli:
                output = server.refresh_knowledge(folder="src", index="source_code")

            self.assertEqual(output, "ok")
            run_cli.assert_called_once()
            self.assertIn("ingest", run_cli.call_args.args[0])

    def test_inspect_file_symbols_parses_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = SmakMcpServer(workspace_root=Path(tmp_dir))
            with patch.object(server, "_run_cli", return_value='["a.py::A"]'):
                symbols = server.inspect_file_symbols("a.py")

            self.assertEqual(symbols, ["a.py::A"])

    def test_semantic_search_parses_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = SmakMcpServer(workspace_root=Path(tmp_dir))
            with patch.object(server, "_run_cli", return_value='[{"uid":"1","score":0.9}]'):
                result = server.semantic_search("auth")

            self.assertEqual(result[0]["uid"], "1")

    def test_update_sidecar_metadata_parses_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            server = SmakMcpServer(workspace_root=Path(tmp_dir))
            payload = '{"applied_updates":1,"total_symbols":2}'
            with patch.object(server, "_run_cli", return_value=payload):
                result = server.update_sidecar_metadata(
                    file_path="src/a.py",
                    updates=[{"symbol": "src/a.py::foo", "intent": "x", "relations": []}],
                )

            self.assertEqual(result["applied_updates"], 1)

    def test_build_mcp_server_returns_sdk_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mcp = build_mcp_server(tmp_dir)
            self.assertEqual(mcp.name, "SMAK")


if __name__ == "__main__":
    unittest.main()
