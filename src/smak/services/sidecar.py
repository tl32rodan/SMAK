from __future__ import annotations

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
        return [
            unit.metadata.get("symbol", unit.uid)
            for unit in parser.parse(content, source=str(path))
        ]

    def update(
        self,
        file_path: Path,
        *,
        symbol: str | None = None,
        intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        if symbol is not None:
            return self._update_single_symbol(file_path, symbol, intent, relations)
        return self._update_full_sync(file_path)

    def _update_full_sync(self, file_path: Path) -> dict[str, Any]:
        current_uids = set(self.inspect(file_path))
        existing = self.sidecar_store.load_symbols_for_source(file_path)
        existing_by_name: dict[str, dict[str, Any]] = {
            e["name"]: e for e in existing if isinstance(e.get("name"), str)
        }

        deleted = set(existing_by_name.keys()) - current_uids
        blocked = sorted(
            name for name in deleted if existing_by_name[name].get("relations")
        )
        if blocked:
            cmds = "\n".join(
                f'  smak sidecar clear {file_path} --symbol "{name}"'
                for name in blocked
            )
            raise ValueError(
                f"Cannot remove symbols with existing relations. "
                f"Clear them first:\n{cmds}"
            )

        merged: list[dict[str, Any]] = []
        added = 0
        for uid in sorted(current_uids):
            if uid in existing_by_name:
                merged.append(existing_by_name[uid])
            else:
                merged.append({"name": uid, "intent": "", "relations": []})
                added += 1

        sidecar_path = self.sidecar_store.save_symbols_for_source(file_path, merged)
        return {
            "file_path": str(file_path),
            "sidecar_path": str(sidecar_path),
            "total_symbols": len(merged),
            "added": added,
            "removed": len(deleted),
        }

    def _update_single_symbol(
        self,
        file_path: Path,
        symbol: str,
        intent: str | None,
        relations: list[str] | None,
    ) -> dict[str, Any]:
        if intent is None and relations is None:
            raise ValueError(
                "At least one of --intent or --relations is required with --symbol."
            )
        update_entry: dict[str, Any] = {"name": symbol}
        if intent is not None:
            update_entry["intent"] = intent
        if relations is not None:
            update_entry["relations"] = relations

        sidecar_path, total_symbols = self.sidecar_store.merge_symbols_for_source(
            file_path, [update_entry]
        )
        return {
            "file_path": str(file_path),
            "sidecar_path": str(sidecar_path),
            "total_symbols": total_symbols,
        }

    def clear_symbol(self, file_path: Path, symbol_name: str) -> dict[str, Any]:
        existing = self.sidecar_store.load_symbols_for_source(file_path)
        filtered = [e for e in existing if e.get("name") != symbol_name]
        if len(filtered) == len(existing):
            raise ValueError(
                f"Symbol '{symbol_name}' not found in sidecar for {file_path}"
            )
        sidecar_path = self.sidecar_store.save_symbols_for_source(file_path, filtered)
        return {
            "file_path": str(file_path),
            "sidecar_path": str(sidecar_path),
            "cleared_symbol": symbol_name,
            "remaining_symbols": len(filtered),
        }
