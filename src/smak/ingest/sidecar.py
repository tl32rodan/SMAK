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

        annotations = metadata.get("symbols", [])
        if annotations is None:
            return
        if (
            not isinstance(annotations, Sequence)
            or isinstance(annotations, (str, bytes, Mapping))
        ):
            raise IntegrityError("Sidecar symbols must be a list of objects.")
        missing: list[str] = []
        for entry in annotations:
            if not isinstance(entry, Mapping):
                raise IntegrityError("Sidecar symbols must be a list of objects.")
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                raise IntegrityError("Sidecar symbol entries must include a name.")
            if name not in symbols:
                missing.append(name)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise IntegrityError(f"Missing symbols in source: {missing_list}")

    def apply(
        self, units: Sequence[KnowledgeUnit], metadata: Mapping[str, Any]
    ) -> list[KnowledgeUnit]:
        """Apply sidecar metadata and relations to knowledge units."""

        annotations = metadata.get("symbols", [])
        annotation_map: dict[str, Mapping[str, Any]] = {}
        if isinstance(annotations, Sequence) and not isinstance(
            annotations, (str, bytes, Mapping)
        ):
            for entry in annotations:
                if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
                    annotation_map[entry["name"]] = entry
        enriched_units: list[KnowledgeUnit] = []
        for unit in units:
            symbol = unit.metadata.get("symbol")
            extra = annotation_map.get(symbol, {}) if symbol else {}
            relations = extra.get("relations", unit.relations)
            if not isinstance(relations, Sequence) or isinstance(relations, str):
                relations = [str(relations)]
            mesh_relations = [str(relation) for relation in relations]
            merged_metadata = {
                **unit.metadata,
                **extra,
                "file_name": unit.metadata.get("source") or unit.metadata.get("file_name"),
                "symbol_name": symbol,
                "intent": extra.get("intent"),
                "mesh_relations": mesh_relations,
            }
            enriched_units.append(
                KnowledgeUnit(
                    uid=unit.uid,
                    content=unit.content,
                    source_type=unit.source_type,
                    relations=tuple(mesh_relations),
                    metadata=merged_metadata,
                )
            )
        return enriched_units
