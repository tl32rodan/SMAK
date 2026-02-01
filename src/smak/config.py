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


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str = "openai"
    model: str | None = None


@dataclass(frozen=True)
class StorageConfig:
    """Configuration for vector storage."""

    base_path: str = "vault"


@dataclass(frozen=True)
class SmakConfig:
    """Typed configuration container."""

    indices: list[IndexConfig] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    embedding_dimensions: int = 3


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
                    )
                )
    llm_data = data.get("llm", {}) if isinstance(data, Mapping) else {}
    llm = LLMConfig(
        provider=str(llm_data.get("provider", "openai")),
        model=llm_data.get("model"),
    )
    storage_data = data.get("storage", {}) if isinstance(data, Mapping) else {}
    storage = StorageConfig(base_path=str(storage_data.get("base_path", "vault")))
    if isinstance(data, Mapping):
        embedding_dimensions = int(data.get("embedding_dimensions", 3))
    else:
        embedding_dimensions = 3
    return SmakConfig(
        indices=indices,
        llm=llm,
        storage=storage,
        embedding_dimensions=embedding_dimensions,
    )


__all__ = ["IndexConfig", "LLMConfig", "SmakConfig", "StorageConfig", "load_config"]
