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

    def validate_all(self) -> None:
        issues = []
        dangling = []
        for index_config in self.config.indices:
            for path_str in index_config.paths:
                target_path = Path(path_str)
                if target_path.exists():
                    issues.extend(self.validate_sidecars(target_path))
                    dangling.extend(self.validate_mesh_integrity(target_path))
        problems = [*issues, *dangling]
        if problems:
            raise RuntimeError("\n".join(problems))

    def graph_stats(self) -> dict[str, Any]:
        """Compute knowledge graph coverage statistics across all indices.

        Returns a dict with overall metrics and per-index breakdown:
        total symbols, enriched symbols (with intent), total relations,
        coverage percentage, and asymmetric relation warnings.
        """
        overall_total = 0
        overall_enriched = 0
        overall_relations = 0
        by_index: dict[str, dict[str, Any]] = {}
        asymmetric_warnings: list[str] = []
        all_relations: set[tuple[str, str]] = set()  # (source_uid, target_uid)

        for index_config in self.config.indices:
            idx_total = 0
            idx_enriched = 0
            idx_relations = 0
            idx_files_with_sidecars = 0

            for path_str in index_config.paths:
                target_path = Path(path_str)
                if not target_path.exists():
                    continue

                for sidecar_file in iter_sidecar_files(target_path):
                    idx_files_with_sidecars += 1
                    payload = safe_load(sidecar_file.read_text(encoding="utf-8")) or {}
                    symbols = payload.get("symbols", []) if isinstance(payload, dict) else []

                    source_path = source_path_from_sidecar(sidecar_file)
                    for symbol in symbols if isinstance(symbols, list) else []:
                        if not isinstance(symbol, dict):
                            continue
                        idx_total += 1
                        name = symbol.get("name", "")
                        intent = symbol.get("intent", "")
                        relations = (
                            symbol.get("relations", [])
                            if isinstance(symbol.get("relations", []), list)
                            else []
                        )

                        if intent:
                            idx_enriched += 1
                        idx_relations += len(relations)

                        source_uid = f"{source_path}::{name}"
                        for rel in relations:
                            all_relations.add((source_uid, str(rel)))

            coverage = (idx_enriched / idx_total * 100) if idx_total > 0 else 0.0
            by_index[index_config.name] = {
                "total_symbols": idx_total,
                "enriched_symbols": idx_enriched,
                "total_relations": idx_relations,
                "coverage_pct": round(coverage, 2),
                "files_with_sidecars": idx_files_with_sidecars,
            }
            overall_total += idx_total
            overall_enriched += idx_enriched
            overall_relations += idx_relations

        # Detect asymmetric relations (A→B exists but B→A doesn't)
        for src, tgt in all_relations:
            reverse_exists = any(
                rs == tgt and rt == src for rs, rt in all_relations
            )
            if not reverse_exists:
                asymmetric_warnings.append(
                    f"Asymmetric: {src} -> {tgt} (no reverse relation)"
                )

        overall_coverage = (overall_enriched / overall_total * 100) if overall_total > 0 else 0.0

        return {
            "total_symbols": overall_total,
            "enriched_symbols": overall_enriched,
            "total_relations": overall_relations,
            "coverage_pct": round(overall_coverage, 2),
            "by_index": by_index,
            "asymmetric_warnings": asymmetric_warnings[:50],  # cap output
        }
