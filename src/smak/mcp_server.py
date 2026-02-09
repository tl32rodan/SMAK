"""MCP server tools for the SMAK passive knowledge kernel."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

import smak.cli as smak_cli


@dataclass
class SmakMcpServer:
    workspace_root: Path

    def get_file_structure(self, file_path: str) -> list[str]:
        path = self.workspace_root / file_path
        if not path.exists() or not path.is_file():
            return []
        return smak_cli._symbols_for_path(path, root_path=self.workspace_root)

    def get_symbol_context(self, file_path: str, symbol: str) -> str:
        path = self.workspace_root / file_path
        if not path.exists() or not path.is_file():
            return f"# File not found\n\n- file: `{file_path}`"
        symbol_table = smak_cli._symbols_for_path(path, root_path=self.workspace_root)
        matched_symbol_id = next(
            (item for item in symbol_table if item.endswith(f"::{symbol}")),
            None,
        )
        if matched_symbol_id is None:
            return f"# Symbol not found\n\n- file: `{file_path}`\n- symbol: `{symbol}`"

        symbol_payload = self._read_sidecar_entry(path, symbol)
        intent = symbol_payload.get("intent") if isinstance(symbol_payload, dict) else None
        relations = symbol_payload.get("relations") if isinstance(symbol_payload, dict) else []
        relation_lines = (
            "\n".join(f"- {relation}" for relation in relations) if relations else "- (none)"
        )
        return (
            f"# {matched_symbol_id}\n\n"
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
        payload = self._load_sidecar(sidecar_path)
        entries = payload.get("symbols")
        if not isinstance(entries, list):
            entries = []

        found: dict[str, Any] | None = None
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
            found["relations"] = list(relations)

        sidecar_path.write_text(self._render_sidecar(entries), encoding="utf-8")
        return found

    def link_issue(self, symbol_id: str, issue_id: str) -> dict[str, Any]:
        file_path, symbol = symbol_id.split("::", 1)
        path = self.workspace_root / file_path
        symbol_payload = self._read_sidecar_entry(path, symbol)
        existing = symbol_payload.get("relations") if isinstance(symbol_payload, dict) else []
        relation_list = [str(item) for item in existing] if isinstance(existing, list) else []
        if issue_id not in relation_list:
            relation_list.append(issue_id)
        return self.upsert_sidecar(file_path, symbol, relations=relation_list)

    def diagnose_mesh(self, path: str | None = None) -> list[str]:
        root = self.workspace_root / path if path else self.workspace_root
        issues: list[str] = []
        for sidecar in root.rglob("*.sidecar.yaml"):
            source = sidecar.with_name(sidecar.name.replace(".sidecar.yaml", ""))
            if not source.exists():
                issues.append(f"Orphaned sidecar: {sidecar.relative_to(self.workspace_root)}")
        for sidecar in root.rglob("*.sidecar.yml"):
            source = sidecar.with_name(sidecar.name.replace(".sidecar.yml", ""))
            if not source.exists():
                issues.append(f"Orphaned sidecar: {sidecar.relative_to(self.workspace_root)}")
        return issues

    def _read_sidecar_entry(self, source_path: Path, symbol: str) -> dict[str, Any]:
        sidecar_path = source_path.with_name(f"{source_path.name}.sidecar.yaml")
        payload = self._load_sidecar(sidecar_path)
        entries = payload.get("symbols") if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            return {}
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name") == symbol:
                return entry
        return {}

    @staticmethod
    def _load_sidecar(sidecar_path: Path) -> dict[str, Any]:
        if not sidecar_path.exists():
            return {}
        loaded = SmakMcpServer._load_yaml(sidecar_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _load_yaml(text: str) -> Any:
        yaml_module = importlib.import_module("smak.utils.yaml")
        return yaml_module.safe_load(text)

    @staticmethod
    def _render_sidecar(entries: list[dict[str, Any]]) -> str:
        if not entries:
            return "symbols: []\n"
        lines = ["symbols:"]
        for entry in entries:
            name = str(entry.get("name", ""))
            intent = str(entry.get("intent", ""))
            relations = entry.get("relations", [])
            relation_list = [str(item) for item in relations] if isinstance(relations, list) else []
            lines.append(f"  - name: {name}")
            lines.append(f'    intent: "{intent}"')
            if relation_list:
                lines.append("    relations:")
                for relation in relation_list:
                    lines.append(f"      - {relation}")
            else:
                lines.append("    relations: []")
        return "\n".join(lines) + "\n"


def build_mcp_server(workspace_root: str | Path = ".") -> FastMCP:
    smak_server = SmakMcpServer(workspace_root=Path(workspace_root).resolve())
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def get_file_structure(file_path: str) -> list[str]:
        """Return canonical symbols extracted from a source file."""

        return smak_server.get_file_structure(file_path)

    @mcp.tool()
    def get_symbol_context(file_path: str, symbol: str) -> str:
        """Return sidecar-enriched context for a symbol."""

        return smak_server.get_symbol_context(file_path, symbol)

    @mcp.tool()
    def upsert_sidecar(
        file_path: str,
        symbol: str,
        intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create or update sidecar metadata for a symbol."""

        return smak_server.upsert_sidecar(file_path, symbol, intent=intent, relations=relations)

    @mcp.tool()
    def link_issue(symbol_id: str, issue_id: str) -> dict[str, Any]:
        """Attach an issue relation to a symbol."""

        return smak_server.link_issue(symbol_id, issue_id)

    @mcp.tool()
    def diagnose_mesh(path: str | None = None) -> list[str]:
        """Run sidecar/source consistency diagnostics."""

        return smak_server.diagnose_mesh(path=path)

    return mcp


__all__ = ["SmakMcpServer", "build_mcp_server"]
