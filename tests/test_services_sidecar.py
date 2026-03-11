from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smak.services.sidecar import SidecarService
from smak.sidecar.store import YAMLSidecarStore
from smak.utils.yaml import safe_dump


class TestSidecarService(unittest.TestCase):
    def test_update_creates_sidecar_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()

            result = service.update(source)

            self.assertEqual(result["total_symbols"], 1)
            self.assertEqual(result["added"], 1)
            self.assertEqual(result["removed"], 0)
            sidecar = Path(tmp_dir) / ".main.py.sidecar.yaml"
            self.assertTrue(sidecar.exists())

    def test_update_preserves_existing_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text(
                "def hello():\n  return True\ndef world():\n  pass\n",
                encoding="utf-8",
            )
            # First, find out the actual UIDs the parser produces
            service = SidecarService()
            uids = service.inspect(source)
            hello_uid = [u for u in uids if "hello" in u][0]

            store = YAMLSidecarStore()
            store.save_symbols_for_source(
                source,
                [
                    {"name": hello_uid, "intent": "greet", "relations": ["issue-1"]},
                ],
            )
            service = SidecarService(sidecar_store=store)

            result = service.update(source)

            self.assertEqual(result["total_symbols"], 2)
            self.assertEqual(result["added"], 1)
            symbols = store.load_symbols_for_source(source)
            hello = next(s for s in symbols if s["name"] == hello_uid)
            self.assertEqual(hello["intent"], "greet")
            self.assertEqual(hello["relations"], ["issue-1"])

    def test_update_blocks_removal_of_symbol_with_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()
            uids = service.inspect(source)
            hello_uid = uids[0]

            store = YAMLSidecarStore()
            store.save_symbols_for_source(
                source,
                [
                    {"name": hello_uid, "intent": "", "relations": []},
                    {"name": "old_func", "intent": "old", "relations": ["issue-2"]},
                ],
            )
            service = SidecarService(sidecar_store=store)

            with self.assertRaises(ValueError) as ctx:
                service.update(source)

            msg = str(ctx.exception)
            self.assertIn("Cannot remove symbols with existing relations", msg)
            self.assertIn('--symbol "old_func"', msg)

    def test_update_removes_symbol_without_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()
            uids = service.inspect(source)
            hello_uid = uids[0]

            store = YAMLSidecarStore()
            store.save_symbols_for_source(
                source,
                [
                    {"name": hello_uid, "intent": "", "relations": []},
                    {"name": "old_func", "intent": "old", "relations": []},
                ],
            )
            service = SidecarService(sidecar_store=store)

            result = service.update(source)

            self.assertEqual(result["removed"], 1)
            self.assertEqual(result["total_symbols"], 1)
            symbols = store.load_symbols_for_source(source)
            names = [s["name"] for s in symbols]
            self.assertNotIn("old_func", names)

    def test_update_single_symbol_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()

            result = service.update(
                source, symbol="hello", intent="greeting function"
            )

            self.assertEqual(result["total_symbols"], 1)
            store = YAMLSidecarStore()
            symbols = store.load_symbols_for_source(source)
            self.assertEqual(symbols[0]["intent"], "greeting function")

    def test_update_single_symbol_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            store = YAMLSidecarStore()
            store.save_symbols_for_source(
                source,
                [{"name": "hello", "intent": "greet", "relations": []}],
            )
            service = SidecarService(sidecar_store=store)

            service.update(source, symbol="hello", relations=["issue-1"])

            symbols = store.load_symbols_for_source(source)
            self.assertEqual(symbols[0]["relations"], ["issue-1"])
            self.assertEqual(symbols[0]["intent"], "greet")

    def test_update_single_symbol_requires_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()

            with self.assertRaises(ValueError) as ctx:
                service.update(source, symbol="hello")

            self.assertIn("--intent or --relations", str(ctx.exception))

    def test_clear_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            store = YAMLSidecarStore()
            store.save_symbols_for_source(
                source,
                [
                    {"name": "hello", "intent": "", "relations": ["issue-1"]},
                    {"name": "world", "intent": "", "relations": []},
                ],
            )
            service = SidecarService(sidecar_store=store)

            result = service.clear_symbol(source, "hello")

            self.assertEqual(result["cleared_symbol"], "hello")
            self.assertEqual(result["remaining_symbols"], 1)
            symbols = store.load_symbols_for_source(source)
            self.assertEqual(len(symbols), 1)
            self.assertEqual(symbols[0]["name"], "world")

    def test_clear_symbol_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            store = YAMLSidecarStore()
            store.save_symbols_for_source(
                source,
                [{"name": "hello", "intent": "", "relations": []}],
            )
            service = SidecarService(sidecar_store=store)

            with self.assertRaises(ValueError) as ctx:
                service.clear_symbol(source, "nonexistent")

            self.assertIn("not found", str(ctx.exception))

    def test_validate_passes_for_valid_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()
            uids = service.inspect(source)

            store = YAMLSidecarStore()
            store.save_symbols_for_source(
                source, [{"name": uids[0], "intent": "greet", "relations": []}]
            )
            service = SidecarService(sidecar_store=store)

            result = service.validate(source)
            self.assertTrue(result["valid"])
            self.assertEqual(result["issues"], [])

    def test_validate_fails_for_stale_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")

            store = YAMLSidecarStore()
            store.save_symbols_for_source(
                source, [{"name": "nonexistent_func", "intent": "", "relations": []}]
            )
            service = SidecarService(sidecar_store=store)

            result = service.validate(source)
            self.assertFalse(result["valid"])
            self.assertEqual(len(result["issues"]), 1)
            self.assertIn("nonexistent_func", result["issues"][0])

    def test_validate_passes_when_no_sidecar_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def hello():\n  return True\n", encoding="utf-8")
            service = SidecarService()

            result = service.validate(source)
            self.assertTrue(result["valid"])
            self.assertEqual(result["issues"], [])

    def test_sidecar_read_text_fallback_replaces_invalid_bytes(self) -> None:
        from smak.services import sidecar as sidecar_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.py"
            path.write_bytes(b"ok\x80")
            self.assertEqual(sidecar_module._read_text_with_fallback(path), "ok\ufffd")


if __name__ == "__main__":
    unittest.main()
