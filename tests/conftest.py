"""Bootstrap test dependency shims before test collection."""

import importlib
import sys
from pathlib import Path

_tests_dir = str(Path(__file__).resolve().parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# Force-load the test sitecustomize even if the system one is already loaded.
_expected_suffix = "tests/sitecustomize.py"
_current = sys.modules.get("sitecustomize")
if _current is None or not str(getattr(_current, "__file__", "")).endswith(_expected_suffix):
    sys.modules.pop("sitecustomize", None)
    importlib.import_module("sitecustomize")
