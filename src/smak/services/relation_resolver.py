from __future__ import annotations

from pathlib import Path
from typing import Any

from smak.services.sidecar_store import SidecarStore


def build_symbol_name_candidates(uid: str, metadata: dict[str, Any]) -> set[str]:
    candidates = {uid}
    symbol_name = metadata.get("symbol")
    if isinstance(symbol_name, str) and symbol_name:
        candidates.add(symbol_name)
    if "::" in uid:
        candidates.add(uid.split("::", 1)[1])
    return candidates


class SidecarRelationResolver:
    def __init__(self, sidecar_store: SidecarStore | None = None) -> None:
        self.sidecar_store = sidecar_store or SidecarStore()

    def resolve(self, uid: str, metadata: dict[str, Any]) -> list[str]:
        source = metadata.get("source")
        if not isinstance(source, str) or not source:
            return []

        symbols = self.sidecar_store.load_symbols_for_source(Path(source))
        if not isinstance(symbols, list):
            return []

        candidate_names = build_symbol_name_candidates(uid, metadata)
        for symbol in symbols:
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
