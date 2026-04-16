from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from smak.parsers import get_parser_for_path
from smak.utils.path_env import contains_env_var, expand_env_path, warn_path_mismatch
from smak.utils.yaml import safe_dump

logger = logging.getLogger(__name__)
from smak.sidecar.manager import IntegrityError, SidecarManager
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
    def __init__(
        self,
        sidecar_store: SidecarStore | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.sidecar_store = sidecar_store or YAMLSidecarStore()
        self.env: dict[str, str] = env or {}

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
        if relations:
            self._check_path_mismatch(file_path, relations)
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

    def preview_update(
        self,
        file_path: Path,
        symbol: str,
        intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute enriched sidecar content without writing to disk.

        Combines full-sync and single-symbol-update logic in memory.
        Returns a dict with ``sidecar_yaml``, ``sidecar_path``, and
        ``total_symbols``.
        """
        # 1. Full sync in memory
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

        table: dict[str, dict[str, Any]] = {}
        for uid in sorted(current_uids):
            if uid in existing_by_name:
                table[uid] = dict(existing_by_name[uid])
            else:
                table[uid] = {"name": uid, "intent": "", "relations": []}

        # 2. Apply single-symbol update in memory
        if intent is not None or relations is not None:
            if relations:
                self._check_path_mismatch(file_path, relations)
            target = table.get(symbol, {"name": symbol})
            if intent is not None:
                target["intent"] = intent
            if relations is not None:
                target["relations"] = relations
            table[symbol] = target

        final = sorted(table.values(), key=lambda e: str(e.get("name", "")))
        sidecar_path = self.sidecar_store.resolve_sidecar_path(file_path)
        sidecar_yaml = safe_dump({"symbols": final})

        return {
            "file_path": str(file_path),
            "sidecar_path": str(sidecar_path),
            "sidecar_yaml": sidecar_yaml,
            "total_symbols": len(final),
        }

    def validate(self, file_path: Path) -> dict[str, Any]:
        """Validate that sidecar symbol names match the source file's parsed symbols.

        Raises IntegrityError if the sidecar references symbols not present in the source.
        """
        current_symbols = self.inspect(file_path)
        existing = self.sidecar_store.load_symbols_for_source(file_path)
        if not existing:
            return {"file_path": str(file_path), "valid": True, "issues": []}

        metadata = {"symbols": existing}
        manager = SidecarManager()
        try:
            manager.validate(current_symbols, metadata)
        except IntegrityError as exc:
            return {"file_path": str(file_path), "valid": False, "issues": [str(exc)]}
        return {"file_path": str(file_path), "valid": True, "issues": []}

    def _check_path_mismatch(
        self, file_path: Path, relations: list[str]
    ) -> None:
        """Warn if any relation UID contains env vars whose expanded root
        doesn't match the sidecar file's location."""
        import re

        resolved_file = file_path.resolve()
        for rel in relations:
            if not contains_env_var(rel):
                continue
            rel_path = rel.split("::")[0] if "::" in rel else rel
            # Extract the first env var to find the root directory
            match = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)", rel_path)
            if match is None:
                continue
            env_root = self.env.get(match.group(1))
            if env_root is None:
                continue
            env_root_path = Path(env_root).resolve()
            try:
                resolved_file.relative_to(env_root_path)
            except ValueError:
                logger.warning(
                    "Path mismatch: sidecar at '%s' has relation targeting '%s' "
                    "(env root: '%s'). The sidecar file location does not match "
                    "the relation target's root directory. This is expected when "
                    "editing in an SOS workspace.",
                    file_path,
                    rel,
                    env_root_path,
                )
                return

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
