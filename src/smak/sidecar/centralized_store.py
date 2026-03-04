from __future__ import annotations

from pathlib import Path
from typing import Any

from smak.utils.yaml import safe_dump, safe_load


class CentralizedYAMLSidecarStore:
    """Persist all sidecar payloads in a single workspace-level YAML file."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.store_path = workspace_root / ".smak" / "sidecars.yaml"

    def _key(self, source_path: Path) -> str:
        try:
            return str(source_path.resolve().relative_to(self.workspace_root.resolve()))
        except ValueError:
            return str(source_path)

    def _load_all(self) -> dict[str, Any]:
        if not self.store_path.exists() or not self.store_path.is_file():
            return {"sources": {}}
        try:
            parsed = safe_load(self.store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"sources": {}}
        if not isinstance(parsed, dict):
            return {"sources": {}}
        sources = parsed.get("sources")
        if not isinstance(sources, dict):
            parsed["sources"] = {}
        return parsed

    def _save_all(self, data: dict[str, Any]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(safe_dump(data), encoding="utf-8")

    def load_payload_for_source(self, source_path: Path) -> dict[str, Any]:
        payload = self._load_all().get("sources", {}).get(self._key(source_path), {})
        if isinstance(payload, dict):
            return payload
        return {}

    def load_symbols_for_source(self, source_path: Path) -> list[dict[str, Any]]:
        payload = self.load_payload_for_source(source_path)
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            return []
        return [item for item in symbols if isinstance(item, dict)]

    def save_symbols_for_source(self, source_path: Path, symbols: list[dict[str, Any]]) -> Path:
        data = self._load_all()
        sources = data.setdefault("sources", {})
        if not isinstance(sources, dict):
            sources = {}
            data["sources"] = sources
        sources[self._key(source_path)] = {"symbols": symbols}
        self._save_all(data)
        return self.store_path

    def merge_symbols_for_source(
        self,
        source_path: Path,
        updates: list[dict[str, Any]],
    ) -> tuple[Path, int]:
        data = self._load_all()
        sources = data.setdefault("sources", {})
        if not isinstance(sources, dict):
            sources = {}
            data["sources"] = sources

        current = sources.get(self._key(source_path), {})
        if not isinstance(current, dict):
            current = {}

        existing = current.get("symbols")
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

        current["symbols"] = sorted(table.values(), key=lambda item: str(item.get("name", "")))
        sources[self._key(source_path)] = current
        self._save_all(data)
        return self.store_path, len(current["symbols"])
