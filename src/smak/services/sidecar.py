from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smak.services.ingest.parsers import get_parser_for_path
from smak.sidecar.paths import is_sidecar_file
from smak.sidecar.protocols import SidecarStore
from smak.sidecar.store import YAMLSidecarStore


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
    def __init__(self, sidecar_store: SidecarStore | None = None) -> None:
        self.sidecar_store = sidecar_store or YAMLSidecarStore()

    def inspect(self, path: Path) -> list[str]:
        parser = get_parser_for_path(path)
        content = _read_text_with_fallback(path)
        return [unit.uid for unit in parser.parse(content, source=str(path))]

    def init(self, target_path: Path) -> Path:
        if target_path.is_dir():
            symbols: list[str] = []
            for source_path in sorted(_iter_source_files(target_path)):
                symbols.extend(self.inspect(source_path))
            lines = ["symbols:"]
            for symbol in symbols:
                lines.extend([f"  - name: {symbol}", '    intent: ""', "    relations: []"])
            payload = "\n".join(lines) + "\n" if symbols else "symbols: []\n"
            output = target_path / ".sidecar.yaml"
            output.write_text(payload, encoding="utf-8")
            return output

        parser = get_parser_for_path(target_path)
        units = parser.parse(_read_text_with_fallback(target_path), source=str(target_path))
        symbols = [
            {
                "name": str(unit.metadata.get("symbol", unit.uid)),
                "intent": "",
                "relations": [],
            }
            for unit in units
        ]
        return self.sidecar_store.save_symbols_for_source(target_path, symbols)

    def update(self, file_path: Path, updates: str) -> dict[str, Any]:
        parsed_updates = json.loads(updates)
        normalized = self._normalize_updates(parsed_updates)
        sidecar_path, total_symbols = self.sidecar_store.merge_symbols_for_source(
            file_path,
            normalized,
        )
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
