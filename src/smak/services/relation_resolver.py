from __future__ import annotations

from pathlib import Path
from typing import Any

from smak.sidecar.protocols import SidecarLoader
from smak.sidecar.store import YAMLSidecarStore
from smak.utils.path_env import contains_env_var, expand_env_path


def build_symbol_name_candidates(uid: str, metadata: dict[str, Any]) -> set[str]:
    candidates = {uid}
    symbol_name = metadata.get("symbol")
    if isinstance(symbol_name, str) and symbol_name:
        candidates.add(symbol_name)
    if "::" in uid:
        candidates.add(uid.split("::", 1)[1])
    return candidates


class SidecarRelationResolver:
    def __init__(
        self,
        sidecar_store: SidecarLoader | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.sidecar_store = sidecar_store or YAMLSidecarStore()
        self.env: dict[str, str] = env or {}

    def resolve(self, uid: str, metadata: dict[str, Any]) -> list[str]:
        source = metadata.get("source")
        if not isinstance(source, str) or not source:
            return []

        source_path = source
        if contains_env_var(source_path):
            try:
                source_path = expand_env_path(source_path, self.env)
            except ValueError:
                pass

        symbols = self.sidecar_store.load_symbols_for_source(Path(source_path))
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
