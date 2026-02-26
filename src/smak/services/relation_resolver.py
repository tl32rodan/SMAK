from __future__ import annotations

from pathlib import Path
from typing import Any

from smak.services.sidecar_paths import sidecar_path_for_source
from smak.utils.yaml import safe_load


def build_symbol_name_candidates(uid: str, metadata: dict[str, Any]) -> set[str]:
    candidates = {uid}
    symbol_name = metadata.get("symbol")
    if isinstance(symbol_name, str) and symbol_name:
        candidates.add(symbol_name)
    if "::" in uid:
        candidates.add(uid.split("::", 1)[1])
    return candidates


class SidecarRelationResolver:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def resolve(self, uid: str, metadata: dict[str, Any]) -> list[str]:
        source = metadata.get("source")
        if not isinstance(source, str) or not source:
            return []

        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = self.workspace_root / source_path
        sidecar_path = sidecar_path_for_source(source_path)
        if not sidecar_path.exists() or not sidecar_path.is_file():
            return []

        try:
            parsed = safe_load(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(parsed, dict):
            return []

        symbols = parsed.get("symbols")
        if not isinstance(symbols, list):
            return []

        candidate_names = build_symbol_name_candidates(uid, metadata)
        for symbol in symbols:
            if not isinstance(symbol, dict):
                continue
            name = symbol.get("name")
            if not isinstance(name, str) or name not in candidate_names:
                continue
            relations = symbol.get("relations", [])
            if isinstance(relations, str):
                return [relations]
            if isinstance(relations, list):
                return [str(target_uid) for target_uid in relations if str(target_uid)]
            return []
        return []
