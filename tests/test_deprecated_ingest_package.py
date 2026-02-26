from __future__ import annotations

import importlib
import unittest


class TestDeprecatedIngestPackage(unittest.TestCase):
    def test_smak_ingest_package_is_removed(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("smak.ingest")


if __name__ == "__main__":
    unittest.main()
