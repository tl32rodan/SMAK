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
    path: str = "."
    uri: str | None = None


@dataclass(frozen=True)
class SmakConfig:
    """Typed configuration container."""

    indices: list[IndexConfig] = field(default_factory=list)
    embedding_dimensions: int | None = None


def _resolve_config(cfg: SmakConfig, config_path: str | Path) -> SmakConfig:
    config_file = Path(config_path)
    base_path = config_file.resolve().parent if config_file.exists() else Path.cwd().resolve()
    resolved_indices = []
    for index in cfg.indices:
        # Resolve path
        resolved_path = str((base_path / Path(index.path).expanduser()).resolve()) if not Path(index.path).expanduser().is_absolute() else str(Path(index.path).expanduser().resolve())
        # Resolve uri
        if index.uri:
            uri_path = Path(index.uri).expanduser()
            resolved_uri = str((base_path / uri_path).resolve()) if not uri_path.is_absolute() else str(uri_path.resolve())
        else:
            resolved_uri = str((base_path / "./smak_data" / index.name).resolve())
        resolved_indices.append(
            IndexConfig(
                name=index.name,
                description=index.description,
                path=resolved_path,
                uri=resolved_uri,
            )
        )
    return SmakConfig(indices=resolved_indices, embedding_dimensions=cfg.embedding_dimensions)


def load_config(path: str | Path) -> SmakConfig:
    """Load configuration from a YAML file."""

    raw = Path(path).read_text(encoding="utf-8")
    data: Any = safe_load(raw) or {}
    cfg = _coerce_config(data)
    return _resolve_config(cfg, path)


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
                        path=str(entry.get("path", ".")),
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
