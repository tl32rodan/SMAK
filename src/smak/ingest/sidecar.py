"""Sidecar metadata loader and validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from smak.core.domain import KnowledgeUnit
from smak.utils.yaml import safe_load


class IntegrityError(ValueError):
    """Raised when sidecar metadata is out of sync with source."""


@dataclass
class SidecarManager:
    """Parse and validate sidecar metadata."""

    def load(self, payload: str | None) -> dict[str, Any]:
        """Load metadata from YAML payload."""

        if not payload:
            return {}
        parsed = safe_load(payload)
        if parsed is None:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    def validate(self, symbols: Sequence[str], metadata: Mapping[str, Any]) -> None:
        """Ensure annotated symbols exist in the source symbol list."""

        annotations = metadata.get("symbols", {})
        if not isinstance(annotations, Mapping):
            raise IntegrityError("Sidecar symbols must be a mapping.")
        missing = [name for name in annotations.keys() if name not in symbols]
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise IntegrityError(f"Missing symbols in source: {missing_list}")

    def apply(
        self, units: Sequence[KnowledgeUnit], metadata: Mapping[str, Any]
    ) -> list[KnowledgeUnit]:
        """Apply sidecar metadata and relations to knowledge units."""

        annotations = metadata.get("symbols", {})
        if not isinstance(annotations, Mapping):
            annotations = {}
        enriched_units: list[KnowledgeUnit] = []
        for unit in units:
            symbol = unit.metadata.get("symbol")
            extra = annotations.get(symbol, {}) if symbol else {}
            if not isinstance(extra, Mapping):
                extra = {"value": extra}
            relations = extra.get("relations", unit.relations)
            if not isinstance(relations, Sequence) or isinstance(relations, str):
                relations = [str(relations)]
            merged_metadata = {**unit.metadata, **extra}
            enriched_units.append(
                KnowledgeUnit(
                    uid=unit.uid,
                    content=unit.content,
                    source_type=unit.source_type,
                    relations=tuple(relations),
                    metadata=merged_metadata,
                )
            )
        return enriched_units
