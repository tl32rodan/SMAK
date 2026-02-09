"""YAML parsing helpers."""

from __future__ import annotations

from typing import Any

import yaml


def safe_load(text: str) -> Any:
    """Parse YAML content with PyYAML."""

    return yaml.safe_load(text)


def safe_dump(data: Any) -> str:
    """Serialize YAML content with PyYAML."""

    return yaml.safe_dump(data, sort_keys=False)


__all__ = ["safe_load", "safe_dump"]
