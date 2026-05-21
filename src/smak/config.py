"""Configuration loader for SMAK."""

from __future__ import annotations

import glob as _glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from smak.utils.path_env import contains_env_var, expand_env_path
from smak.utils.yaml import safe_load


_EMBEDDING_SETUP_YAML = Path(__file__).resolve().parent / "embedding_setup.yaml"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding service configuration loaded from ``embedding_setup.yaml``."""

    api_base: str = "http://f15dtpai1:11515/v1"
    model: str = "qwen3_embedding_8B"
    timeout: float = 600.0
    batch_size: int = 32


def load_embedding_config(path: str | Path | None = None) -> EmbeddingConfig:
    """Load :class:`EmbeddingConfig` from a YAML file.

    Falls back to the package-level ``embedding_setup.yaml`` when *path* is
    ``None`` or when the file does not exist.
    """
    target = Path(path) if path else _EMBEDDING_SETUP_YAML
    if not target.exists():
        return EmbeddingConfig()
    raw = target.read_text(encoding="utf-8")
    data: Any = safe_load(raw) or {}
    if not isinstance(data, Mapping):
        return EmbeddingConfig()
    known = {f for f in EmbeddingConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in data.items() if k in known}
    return EmbeddingConfig(**kwargs)


@dataclass(frozen=True)
class IndexConfig:
    """Configuration for an index."""

    name: str
    description: str
    uri: str
    paths: list[str] = field(default_factory=lambda: ["."])


@dataclass(frozen=True)
class SmakConfig:
    """Typed configuration container."""

    indices: list[IndexConfig] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    embedding_dimensions: int | None = None

    def get_index(self, name: str) -> IndexConfig | None:
        """Return the IndexConfig with the given name, or None if not found."""
        return next((entry for entry in self.indices if entry.name == name), None)


def _resolve_absolute_path(raw: str, base: Path, env: dict[str, str]) -> str:
    """Resolve *raw* relative to *base*, or expand it if already absolute.

    Supports ``$VAR`` references which are expanded using *env*
    before path resolution.
    """
    if contains_env_var(raw):
        raw = expand_env_path(raw, env)
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())
    return str((base / expanded).resolve())


def _has_glob_meta(pattern: str) -> bool:
    """Return ``True`` if *pattern* contains shell glob metacharacters."""
    return any(ch in pattern for ch in ("*", "?", "["))


def _expand_glob_paths(raw_paths: list[str], base_path: Path, env: dict[str, str]) -> list[str]:
    """Resolve and expand *raw_paths*, supporting shell glob patterns.

    Literal (non-glob) paths are resolved as before.  Glob patterns are
    expanded via :func:`glob.glob` and both **files and directories** are kept.

    Raises:
        ValueError: If a glob pattern matches zero files or directories.
    """
    expanded: list[str] = []
    for raw in raw_paths:
        resolved = _resolve_absolute_path(raw, base_path, env)
        if _has_glob_meta(resolved):
            matches = sorted(
                p for p in _glob.glob(resolved) if Path(p).is_dir() or Path(p).is_file()
            )
            if not matches:
                raise ValueError(
                    f"Glob pattern '{raw}' (resolved to '{resolved}') "
                    "matched zero files or directories."
                )
            expanded.extend(matches)
        else:
            expanded.append(resolved)
    return expanded


def _resolve_config(cfg: SmakConfig, config_path: str | Path) -> SmakConfig:
    config_file = Path(config_path)
    base_path = config_file.resolve().parent if config_file.exists() else Path.cwd().resolve()
    env = cfg.env
    resolved_indices = []
    for index in cfg.indices:
        resolved_paths = _expand_glob_paths(index.paths, base_path, env)
        resolved_uri = _resolve_absolute_path(index.uri, base_path, env)
        resolved_indices.append(
            IndexConfig(
                name=index.name,
                description=index.description,
                uri=resolved_uri,
                paths=resolved_paths,
            )
        )
    return SmakConfig(indices=resolved_indices, env=env, embedding_dimensions=cfg.embedding_dimensions)


def load_config(path: str | Path) -> SmakConfig:
    """Load configuration from a YAML file."""

    raw = Path(path).read_text(encoding="utf-8")
    data: Any = safe_load(raw) or {}
    cfg = _coerce_config(data)
    return _resolve_config(cfg, path)


def _coerce_config(data: Mapping[str, Any]) -> SmakConfig:
    # Parse env block
    raw_env = data.get("env", {}) if isinstance(data, Mapping) else {}
    env: dict[str, str] = {}
    if isinstance(raw_env, Mapping):
        env = {str(k): str(v) for k, v in raw_env.items()}

    indices_data = data.get("indices", []) if isinstance(data, Mapping) else []
    indices: list[IndexConfig] = []
    if isinstance(indices_data, list):
        for entry in indices_data:
            if isinstance(entry, Mapping):
                raw_paths = entry.get("paths", ["."])
                paths = [str(p) for p in raw_paths] if isinstance(raw_paths, list) else [str(raw_paths)]
                index_name = str(entry.get("name", ""))
                if entry.get("uri") is None:
                    raise ValueError(
                        f"Index '{index_name}' is missing the required 'uri' field. "
                        "Every index must specify a uri for its vector store."
                    )
                raw_uri = str(entry["uri"])
                # Validate uri is absolute or uses $VAR (after env expansion)
                test_uri = raw_uri
                if contains_env_var(test_uri):
                    try:
                        test_uri = expand_env_path(test_uri, env)
                    except ValueError:
                        pass  # undefined var — will fail later during resolve
                if not contains_env_var(raw_uri) and not Path(test_uri).expanduser().is_absolute():
                    raise ValueError(
                        f"Index '{index_name}' has a relative uri '{raw_uri}'. "
                        "The uri must be an absolute path or use a variable "
                        "defined in the env block (e.g. $SMAK_DATA/source_code)."
                    )
                indices.append(
                    IndexConfig(
                        name=index_name,
                        description=str(entry.get("description", "")),
                        uri=raw_uri,
                        paths=paths,
                    )
                )
    return SmakConfig(
        indices=indices,
        env=env,
    )


__all__ = [
    "EmbeddingConfig",
    "IndexConfig",
    "SmakConfig",
    "load_config",
    "load_embedding_config",
]
