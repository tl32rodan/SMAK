"""MCP server tools for the SMAK passive knowledge kernel."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


@dataclass
class SmakMcpServer:
    workspace_root: Path
    smak_binary: str = "smak"
    config_name: str = "workspace_config.yaml"

    def _run_cli(self, args: list[str]) -> str:
        completed = subprocess.run(
            [self.smak_binary, *args],
            cwd=self.workspace_root,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "Unknown CLI failure"
            raise RuntimeError(message)
        return completed.stdout.strip()

    def refresh_knowledge(self, folder: str = ".", index: str = "source_code") -> str:
        return self._run_cli(
            ["ingest", "--folder", folder, "--index", index, "--config", self.config_name]
        )

    def semantic_search(
        self, query: str, index: str = "source_code", top_k: int = 5
    ) -> dict[str, Any]:
        output = self._run_cli(
            ["query", query, "--index", index, "--top-k", str(top_k), "--config", self.config_name]
        )
        parsed = json.loads(output) if output else {}
        return parsed if isinstance(parsed, dict) else {}

    def manage_sidecar(
        self,
        action: str,
        file_path: str,
        updates: list[dict[str, Any]] | None = None,
        reingest: bool = False,
        index: str = "source_code",
    ) -> dict[str, Any] | list[str] | str:
        if action == "inspect":
            output = self._run_cli(
                ["sidecar", "inspect", file_path, "--config", self.config_name, "--json-output"]
            )
            parsed = json.loads(output) if output else []
            return [str(symbol) for symbol in parsed]
        if action == "init":
            return self._run_cli(["sidecar", "init", file_path, "--config", self.config_name])
        if action == "update":
            command = [
                "sidecar",
                "update",
                file_path,
                "--updates",
                json.dumps(updates or [], ensure_ascii=False),
                "--config",
                self.config_name,
                "--index",
                index,
            ]
            if reingest:
                command.append("--reingest")
            output = self._run_cli(command)
            parsed = json.loads(output) if output else {}
            return parsed if isinstance(parsed, dict) else {}
        raise ValueError("action must be one of: init, update, inspect")

    def validate_mesh(self, path: str = ".") -> str:
        return self._run_cli(["doctor", "--path", path, "--config", self.config_name])


def build_mcp_server(workspace_root: str | Path = ".") -> FastMCP:
    smak_server = SmakMcpServer(workspace_root=Path(workspace_root).resolve())
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def refresh_knowledge(folder: str = ".", index: str = "source_code") -> str:
        return smak_server.refresh_knowledge(folder=folder, index=index)

    @mcp.tool()
    def semantic_search(query: str, index: str = "source_code", top_k: int = 5) -> dict[str, Any]:
        return smak_server.semantic_search(query=query, index=index, top_k=top_k)

    @mcp.tool()
    def manage_sidecar(
        action: str,
        file_path: str,
        updates: list[dict[str, Any]] | None = None,
        reingest: bool = False,
        index: str = "source_code",
    ) -> dict[str, Any] | list[str] | str:
        return smak_server.manage_sidecar(
            action=action,
            file_path=file_path,
            updates=updates,
            reingest=reingest,
            index=index,
        )

    @mcp.tool()
    def validate_mesh(path: str = ".") -> str:
        return smak_server.validate_mesh(path=path)

    return mcp


def main() -> None:
    server = build_mcp_server(Path.cwd())
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
