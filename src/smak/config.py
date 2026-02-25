"""Configuration loader for SMAK."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from smak.utils.yaml import safe_load


@dataclass(frozen=True)
class IndexConfig:
    """Configuration for an index."""

    name: str
    description: str
    uri: str | None = None


@dataclass(frozen=True)
class SmakConfig:
    """Typed configuration container."""

    indices: list[IndexConfig] = field(default_factory=list)
    embedding_dimensions: int | None = None


def load_config(path: str | Path) -> SmakConfig:
    """Load configuration from a YAML file."""

    raw = Path(path).read_text(encoding="utf-8")
    data: Any = safe_load(raw) or {}
    return _coerce_config(data)


def _coerce_config(data: Mapping[str, Any]) -> SmakConfig:
    indices_data = data.get("indices", []) if isinstance(data, Mapping) else []
    indices: list[IndexConfig] = []
    if isinstance(indices_data, list):
        for entry in indices_data:
            if isinstance(entry, Mapping):
                indices.append(
                    IndexConfig(
                        name=str(entry.get("name", "")),
                        description=str(entry.get("description", "")),
                        uri=(
                            str(entry["uri"])
                            if entry.get("uri") is not None
                            else None
                        ),
                    )
                )
    return SmakConfig(
        indices=indices,
    )


__all__ = ["IndexConfig", "SmakConfig", "load_config"]
