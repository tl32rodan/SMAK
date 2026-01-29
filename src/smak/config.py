"""Configuration defaults for SMAK."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmakConfig:
    """Simple configuration container."""

    embedding_dimensions: int = 3
    default_source: str = "content"
