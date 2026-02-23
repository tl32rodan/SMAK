from __future__ import annotations

import sys
import unittest
from pathlib import Path


class TestSitecustomizeScope(unittest.TestCase):
    def test_test_runtime_uses_tests_sitecustomize(self) -> None:
        module = sys.modules.get("sitecustomize")

        self.assertIsNotNone(module)
        assert module is not None
        self.assertTrue(str(module.__file__).endswith("tests/sitecustomize.py"))

    def test_repo_no_global_sitecustomize_under_src(self) -> None:
        self.assertFalse(Path("src/sitecustomize.py").exists())


if __name__ == "__main__":
    unittest.main()
