"""MCP server tools for the SMAK passive knowledge kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smak.ingest.parsers import PythonParser
from smak.ingest.sidecar import SidecarManager
from smak.utils.yaml import safe_dump, safe_load


@dataclass
class SmakMcpServer:
    workspace_root: Path

    def get_file_structure(self, file_path: str) -> list[str]:
        path = self.workspace_root / file_path
        parser = PythonParser(root_path=str(self.workspace_root))
        content = path.read_text(encoding="utf-8", errors="replace")
        units = parser.parse(content, source=str(path))
        return [unit.uid for unit in units]

    def get_symbol_context(self, file_path: str, symbol: str) -> str:
        path = self.workspace_root / file_path
        parser = PythonParser(root_path=str(self.workspace_root))
        content = path.read_text(encoding="utf-8", errors="replace")
        units = parser.parse(content, source=str(path))
        sidecar_data = self._load_sidecar(path)
        enriched = SidecarManager().apply(units, sidecar_data)
        matched = next((unit for unit in enriched if unit.metadata.get("symbol") == symbol), None)
        if matched is None:
            return f"# Symbol not found\n\n- file: `{file_path}`\n- symbol: `{symbol}`"
        intent = matched.metadata.get("intent") or ""
        relations = matched.metadata.get("mesh_relations") or []
        relation_lines = "\n".join(f"- {relation}" for relation in relations) or "- (none)"
        return (
            f"# {matched.uid}\n\n"
            f"## Code Definition\n```python\n{matched.content}\n```\n\n"
            f"## Sidecar Intent\n{intent or '(missing)'}\n\n"
            f"## Inherited Issue Links\n{relation_lines}\n"
        )

    def upsert_sidecar(
        self,
        file_path: str,
        symbol: str,
        intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        path = self.workspace_root / file_path
        sidecar_path = path.with_name(f"{path.name}.sidecar.yaml")
        payload = self._load_sidecar(path)
        entries = payload.get("symbols", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            entries = []
        found = None
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name") == symbol:
                found = entry
                break
        if found is None:
            found = {"name": symbol}
            entries.append(found)
        if intent is not None:
            found["intent"] = intent
        if relations is not None:
            found["relations"] = relations
        payload = {"symbols": entries}
        sidecar_path.write_text(safe_dump(payload), encoding="utf-8")
        return found

    def link_issue(self, symbol_id: str, issue_id: str) -> dict[str, Any]:
        file_path, symbol = symbol_id.split("::", 1)
        path = self.workspace_root / file_path
        current = self._load_sidecar(path)
        entries = current.get("symbols", []) if isinstance(current, dict) else []
        existing: list[str] = []
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name") == symbol:
                rels = entry.get("relations", [])
                existing = [str(x) for x in rels] if isinstance(rels, list) else []
                break
        if issue_id not in existing:
            existing.append(issue_id)
        return self.upsert_sidecar(file_path, symbol, relations=existing)

    def diagnose_mesh(self, path: str | None = None) -> list[str]:
        root = self.workspace_root / path if path else self.workspace_root
        problems: list[str] = []
        for sidecar in root.rglob("*.sidecar.yaml"):
            source = sidecar.with_name(sidecar.name.replace(".sidecar.yaml", ""))
            if not source.exists():
                problems.append(f"Orphaned sidecar: {sidecar.relative_to(self.workspace_root)}")
        return problems

    def _load_sidecar(self, source_path: Path) -> dict[str, Any]:
        sidecar_path = source_path.with_name(f"{source_path.name}.sidecar.yaml")
        if not sidecar_path.exists():
            return {}
        data = safe_load(sidecar_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
