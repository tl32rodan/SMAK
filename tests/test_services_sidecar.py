from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smak.services.sidecar import SidecarService


class TestSidecarService(unittest.TestCase):
    def test_sidecar_service_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()
            result = service.update(
                source, json.dumps([{"symbol": "main.py::hello", "relations": ["issue-1"]}])
            )
            self.assertEqual(result["applied_updates"], 1)


    def test_sidecar_service_init_directory_writes_hidden_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "a.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()

            output = service.init(root)

            self.assertEqual(output, root / ".sidecar.yaml")
            self.assertTrue(output.exists())


    def test_sidecar_service_init_for_markdown_uses_wildcard_symbol_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "note.md"
            source.write_text("# title\ncontent\n", encoding="utf-8")
            service = SidecarService()

            sidecar_path = service.init(source)
            payload = sidecar_path.read_text(encoding="utf-8")

            self.assertIn("- name: *", payload)

    def test_sidecar_read_text_fallback_replaces_invalid_bytes(self) -> None:
        from smak.services import sidecar as sidecar_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.py"
            path.write_bytes(b"ok\x80")
            self.assertEqual(sidecar_module._read_text_with_fallback(path), "ok�")


if __name__ == "__main__":
    unittest.main()
