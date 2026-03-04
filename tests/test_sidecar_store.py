from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smak.sidecar import CentralizedYAMLSidecarStore, SidecarStore, YAMLSidecarStore


class TestSidecarStore(unittest.TestCase):
    def test_yaml_merge_symbols_for_source_uses_hidden_sidecar_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            source = workspace_root / "src" / "a.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("print('x')\n", encoding="utf-8")

            store = YAMLSidecarStore()
            sidecar_path, total_symbols = store.merge_symbols_for_source(
                source,
                [{"name": "src/a.py::foo", "relations": ["issue:1"]}],
            )

            self.assertEqual(sidecar_path, source.with_name(".a.py.sidecar.yaml"))
            self.assertEqual(total_symbols, 1)
            symbols = store.load_symbols_for_source(source)
            self.assertEqual(symbols[0]["name"], "src/a.py::foo")

    def test_yaml_store_delete_sidecar_for_source_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "src" / "a.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("print('x')\n", encoding="utf-8")
            sidecar = source.with_name(".a.py.sidecar.yaml")
            sidecar.write_text("symbols: []\n", encoding="utf-8")

            store = YAMLSidecarStore()
            store.delete_sidecar_for_source(source)
            store.delete_sidecar_for_source(source)

            self.assertFalse(sidecar.exists())

    def test_yaml_store_runtime_protocol_compliance(self) -> None:
        self.assertIsInstance(YAMLSidecarStore(), SidecarStore)

    def test_centralized_store_runtime_protocol_compliance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertIsInstance(CentralizedYAMLSidecarStore(Path(tmp_dir)), SidecarStore)

    def test_centralized_store_roundtrip_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "src" / "a.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("print('x')\n", encoding="utf-8")

            store = CentralizedYAMLSidecarStore(root)
            sidecar_path = store.save_symbols_for_source(
                source,
                [{"name": "src/a.py::foo", "intent": "demo", "relations": []}],
            )

            self.assertEqual(sidecar_path, root / ".smak" / "sidecars.yaml")
            merged_path, total = store.merge_symbols_for_source(
                source,
                [{"name": "src/a.py::foo", "relations": ["issue:1"]}],
            )
            self.assertEqual(merged_path, sidecar_path)
            self.assertEqual(total, 1)
            loaded = store.load_symbols_for_source(source)
            self.assertEqual(loaded[0]["name"], "src/a.py::foo")
            self.assertIn("issue:1", str(loaded[0].get("relations")))


if __name__ == "__main__":
    unittest.main()
