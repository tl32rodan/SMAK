from __future__ import annotations

from pathlib import Path

from smak.sidecar.paths import iter_sidecar_files, source_path_from_sidecar
from smak.utils.yaml import safe_load


class DoctorService:
    def __init__(self, vector_store: object | None = None) -> None:
        self.vector_store = vector_store

    def validate_sidecars(self, target_path: Path) -> list[str]:
        issues: list[str] = []
        root = target_path if target_path.is_dir() else target_path.parent
        for sidecar_file in iter_sidecar_files(root):
            source = source_path_from_sidecar(sidecar_file)
            if not source.exists():
                issues.append(f"Orphaned sidecar: {sidecar_file}")
        return issues

    def validate_mesh_integrity(self, target_path: Path) -> list[str]:
        if self.vector_store is None:
            return []
        warnings: list[str] = []
        root = target_path if target_path.is_dir() else target_path.parent
        for sidecar_file in iter_sidecar_files(root):
            payload = safe_load(sidecar_file.read_text(encoding="utf-8")) or {}
            symbols = payload.get("symbols", []) if isinstance(payload, dict) else []
            for symbol in symbols if isinstance(symbols, list) else []:
                if not isinstance(symbol, dict):
                    continue
                for relation in (
                    symbol.get("relations", [])
                    if isinstance(symbol.get("relations", []), list)
                    else []
                ):
                    if self.vector_store.get_by_id(str(relation)) is None:
                        warnings.append(
                            (
                                f"Warning: {sidecar_file} references '{relation}', "
                                f"but '{relation}' does not exist in the vector index."
                            )
                        )
        return warnings
