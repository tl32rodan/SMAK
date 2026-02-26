from __future__ import annotations

from pathlib import Path
from typing import Any

from smak.services.sidecar_paths import sidecar_path_for_source
from smak.utils.yaml import safe_dump, safe_load


class SidecarStore:
    """I/O boundary for sidecar persistence and source->sidecar mapping."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = workspace_root

    def resolve_source_path(self, source_path: Path) -> Path:
        if source_path.is_absolute() or self.workspace_root is None:
            return source_path
        return self.workspace_root / source_path

    def resolve_sidecar_path(self, source_path: Path) -> Path:
        return sidecar_path_for_source(self.resolve_source_path(source_path))

    def load_payload_for_source(self, source_path: Path) -> dict[str, Any]:
        sidecar_path = self.resolve_sidecar_path(source_path)
        if not sidecar_path.exists() or not sidecar_path.is_file():
            return {}
        try:
            parsed = safe_load(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def load_symbols_for_source(self, source_path: Path) -> list[dict[str, Any]]:
        payload = self.load_payload_for_source(source_path)
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            return []
        return [item for item in symbols if isinstance(item, dict)]

    def save_symbols_for_source(self, source_path: Path, symbols: list[dict[str, Any]]) -> Path:
        sidecar_path = self.resolve_sidecar_path(source_path)
        payload = {"symbols": symbols}
        sidecar_path.write_text(safe_dump(payload), encoding="utf-8")
        return sidecar_path

    def merge_symbols_for_source(
        self,
        source_path: Path,
        updates: list[dict[str, Any]],
    ) -> tuple[Path, int]:
        sidecar_path = self.resolve_sidecar_path(source_path)
        payload = self.load_payload_for_source(source_path)
        existing = payload.get("symbols")
        if not isinstance(existing, list):
            existing = []

        table: dict[str, dict[str, Any]] = {}
        for entry in existing:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                table[entry["name"]] = dict(entry)
        for update in updates:
            name = update["name"]
            target = table.get(name, {"name": name})
            target.update(update)
            table[name] = target

        payload["symbols"] = sorted(table.values(), key=lambda item: str(item.get("name", "")))
        sidecar_path.write_text(safe_dump(payload), encoding="utf-8")
        return sidecar_path, len(payload["symbols"])
