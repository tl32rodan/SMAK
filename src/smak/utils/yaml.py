"""YAML parsing helpers."""

from __future__ import annotations

import importlib
from typing import Any


def safe_load(text: str) -> Any:
    """Parse YAML content using PyYAML."""

    module = importlib.import_module("yaml")
    return module.safe_load(text)


__all__ = ["safe_load"]
