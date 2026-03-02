from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smak.sidecar.store import SidecarStore


class TestSidecarStore(unittest.TestCase):
    def test_merge_symbols_for_source_resolves_workspace_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            source = workspace_root / "src" / "a.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("print('x')\n", encoding="utf-8")

            store = SidecarStore()
            sidecar_path, total_symbols = store.merge_symbols_for_source(
                source,
                [{"name": "src/a.py::foo", "relations": ["issue:1"]}],
            )

            self.assertEqual(sidecar_path, source.with_name("a.py.sidecar.yaml"))
            self.assertEqual(total_symbols, 1)
            symbols = store.load_symbols_for_source(source)
            self.assertEqual(symbols[0]["name"], "src/a.py::foo")


if __name__ == "__main__":
    unittest.main()
