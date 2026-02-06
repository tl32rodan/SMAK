import tempfile
import unittest
from pathlib import Path

from smak.mcp_server import SmakMcpServer


class TestMcpServer(unittest.TestCase):
    def test_get_symbol_context_and_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src = root / "src"
            src.mkdir()
            file_path = src / "auth.py"
            file_path.write_text(
                "class Auth:\n    def login(self):\n        return True\n",
                encoding="utf-8",
            )
            server = SmakMcpServer(workspace_root=root)
            server.upsert_sidecar(
                file_path="src/auth.py",
                symbol="Auth",
                intent="auth class",
                relations=["issue:101"],
            )

            context = server.get_symbol_context("src/auth.py", "Auth.login")

            self.assertIn("Inherited Issue Links", context)
            self.assertIn("issue:101", context)
            self.assertIn("(missing)", context)

    def test_diagnose_mesh_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            orphan = root / "ghost.py.sidecar.yaml"
            orphan.write_text("symbols: []\n", encoding="utf-8")
            server = SmakMcpServer(workspace_root=root)

            problems = server.diagnose_mesh()

            self.assertEqual(len(problems), 1)
            self.assertIn("Orphaned sidecar", problems[0])


if __name__ == "__main__":
    unittest.main()
