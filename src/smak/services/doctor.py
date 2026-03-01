from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from smak.config import SmakConfig
from smak.sidecar.paths import iter_sidecar_files, source_path_from_sidecar
from smak.utils.yaml import safe_load


class DoctorService:
    def __init__(self, config: SmakConfig, vector_store_loader: Callable[[str], Any]) -> None:
        self.config = config
        self.vector_store_loader = vector_store_loader
        self._stores: dict[str, Any] = {}

    def _get_store(self, index_name: str) -> Any:
        if index_name not in self._stores:
            self._stores[index_name] = self.vector_store_loader(index_name)
        return self._stores[index_name]

    @staticmethod
    def _store_contains_uid(store: Any, target_uid: str) -> bool:
        contains = getattr(store, "contains", None)
        if callable(contains):
            return bool(contains(target_uid))
        get_by_id = getattr(store, "get_by_id", None)
        if callable(get_by_id):
            return get_by_id(target_uid) is not None
        return False

    def _exists_in_any_index(self, target_uid: str) -> bool:
        for index_config in self.config.indices:
            store = self._get_store(index_config.name)
            if self._store_contains_uid(store, target_uid):
                return True
        return False

    def validate_sidecars(self, target_path: Path) -> list[str]:
        issues: list[str] = []
        root = target_path if target_path.is_dir() else target_path.parent
        for sidecar_file in iter_sidecar_files(root):
            source = source_path_from_sidecar(sidecar_file)
            if not source.exists():
                issues.append(f"Orphaned sidecar: {sidecar_file}")
        return issues

    def validate_mesh_integrity(self, target_path: Path) -> list[str]:
        if not self.config.indices:
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
                    if not self._exists_in_any_index(str(relation)):
                        warnings.append(
                            (
                                f"Warning: {sidecar_file} references '{relation}', "
                                f"but '{relation}' does not exist in any configured vector index."
                            )
                        )
        return warnings
