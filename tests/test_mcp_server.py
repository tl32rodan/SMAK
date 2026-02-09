import tempfile
import unittest
from pathlib import Path

from smak.mcp_server import SmakMcpServer, build_mcp_server


class TestMcpServer(unittest.TestCase):
    def test_get_symbol_context_and_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src = root / "src"
            src.mkdir()
            file_path = src / "csv_editor.py"
            file_path.write_text(
                "class CsvEditor:\n    def read_rows(self):\n        return []\n",
                encoding="utf-8",
            )
            server = SmakMcpServer(workspace_root=root)
            server.upsert_sidecar(
                file_path="src/csv_editor.py",
                symbol="CsvEditor.read_rows",
                intent="read csv rows",
                relations=["issue:101"],
            )

            context = server.get_symbol_context("src/csv_editor.py", "CsvEditor.read_rows")

            self.assertIn("Inherited Issue Links", context)
            self.assertIn("issue:101", context)
            self.assertIn("read csv rows", context)

    def test_diagnose_mesh_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            orphan = root / "ghost.py.sidecar.yaml"
            orphan.write_text("symbols: []\n", encoding="utf-8")
            server = SmakMcpServer(workspace_root=root)

            problems = server.diagnose_mesh()

            self.assertEqual(len(problems), 1)
            self.assertIn("Orphaned sidecar", problems[0])

    def test_build_mcp_server_returns_sdk_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mcp = build_mcp_server(tmp_dir)

            self.assertEqual(mcp.name, "SMAK")


if __name__ == "__main__":
    unittest.main()
