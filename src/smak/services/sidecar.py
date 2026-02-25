from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smak.ingest.parsers import get_parser_for_path
from smak.services.sidecar_paths import is_sidecar_file, sidecar_path_for_source
from smak.utils.yaml import safe_dump, safe_load


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")



def _iter_source_files(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file() and not is_sidecar_file(path):
            yield path


class SidecarService:
    def inspect(self, path: Path, *, workspace_root: Path | None = None) -> list[str]:
        parser = get_parser_for_path(path, root_path=workspace_root)
        content = _read_text_with_fallback(path)
        return [unit.uid for unit in parser.parse(content, source=str(path))]

    def init(self, target_path: Path, *, workspace_root: Path | None = None) -> Path:
        if target_path.is_dir():
            symbols: list[str] = []
            for source_path in sorted(_iter_source_files(target_path)):
                symbols.extend(self.inspect(source_path, workspace_root=workspace_root))
            lines = ["symbols:"]
            for symbol in symbols:
                lines.extend([f"  - name: {symbol}", '    intent: ""', "    relations: []"])
            payload = "\n".join(lines) + "\n" if symbols else "symbols: []\n"
            output = target_path / "sidecar.yaml"
            output.write_text(payload, encoding="utf-8")
            return output

        parser = get_parser_for_path(target_path, root_path=workspace_root)
        units = parser.parse(
            _read_text_with_fallback(target_path), source=str(target_path)
        )
        lines = ["symbols:"]
        for unit in units:
            lines.extend(
                [
                    f"  - name: {unit.metadata.get('symbol', unit.uid)}",
                    '    intent: ""',
                    "    relations: []",
                ]
            )
        payload = "\n".join(lines) + "\n" if units else "symbols: []\n"
        output = sidecar_path_for_source(target_path)
        output.write_text(payload, encoding="utf-8")
        return output

    def update(self, file_path: Path, updates: str) -> dict[str, Any]:
        parsed_updates = json.loads(updates)
        normalized = self._normalize_updates(parsed_updates)
        sidecar_path = sidecar_path_for_source(file_path)
        total_symbols = self._merge_updates(sidecar_path, normalized)
        return {
            "file_path": str(file_path),
            "sidecar_path": str(sidecar_path),
            "applied_updates": len(normalized),
            "total_symbols": total_symbols,
        }

    def _normalize_updates(self, updates: Any) -> list[dict[str, Any]]:
        if not isinstance(updates, list):
            raise ValueError("'updates' must be a list.")
        normalized = []
        for entry in updates:
            if not isinstance(entry, dict):
                raise ValueError("Each update must be an object.")
            symbol = entry.get("symbol")
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("Each update requires a non-empty 'symbol'.")
            record = {"name": symbol}
            if "intent" in entry:
                record["intent"] = str(entry.get("intent") or "")
            if "relations" in entry:
                relations = entry.get("relations")
                if not isinstance(relations, list):
                    raise ValueError("'relations' must be a list when provided.")
                record["relations"] = [str(item) for item in relations]
            normalized.append(record)
        return normalized

    def _merge_updates(self, sidecar_path: Path, updates: list[dict[str, Any]]) -> int:
        payload = (
            safe_load(_read_text_with_fallback(sidecar_path)) if sidecar_path.exists() else {}
        )
        if not isinstance(payload, dict):
            payload = {}
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
        return len(payload["symbols"])
