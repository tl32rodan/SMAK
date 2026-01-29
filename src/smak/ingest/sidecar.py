"""Sidecar metadata loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class SidecarLoader:
    """Parse sidecar metadata for ingest operations."""

    def load(self, payload: str | None) -> dict[str, Any]:
        """Load metadata from a JSON payload or return a raw wrapper."""

        if not payload:
            return {}
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {"raw": payload}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
