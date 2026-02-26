"""MCP bridge for exposing SMAK CLI capabilities as tool-callable operations.

This module intentionally keeps business logic in CLI commands (``smak ...``) and
provides a thin wrapper for MCP/agent integrations. Agents can call one of four
high-level tools:

- ``refresh_knowledge`` -> run ingestion
- ``semantic_search`` -> run semantic+relation query
- ``manage_sidecar`` -> init/update/inspect sidecar metadata
- ``validate_mesh`` -> run mesh/sidecar consistency checks

All tool operations execute in a configured workspace root and use
``workspace_config.yaml`` by default.
"""

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
    """CLI-backed adapter used by MCP tool handlers.

    Attributes:
        workspace_root: Project directory where ``smak`` commands are executed.
        smak_binary: CLI executable name/path (defaults to ``smak``).
        config_name: Workspace config file passed to SMAK commands.
    """

    workspace_root: Path
    smak_binary: str = "smak"
    config_name: str = "workspace_config.yaml"

    def _run_cli(self, args: list[str]) -> str:
        """Execute a SMAK CLI command and return stdout.

        Raises:
            RuntimeError: If the underlying command exits with non-zero status.
        """

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

    def refresh_knowledge(
        self,
        folder: str = ".",
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Ingest workspace content into a target index."""

        command = ["ingest", "--folder", folder, "--index", index, "--config", self.config_name]
        if not follow_symlinks:
            command.append("--no-follow-symlinks")
        return self._run_cli(command)

    def semantic_search(
        self, query: str, index: str = "source_code", top_k: int = 5
    ) -> dict[str, Any]:
        """Run ``smak query`` and parse the JSON response payload."""

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
        """Manage sidecar metadata through one unified entrypoint.

        Supported actions:
            - ``inspect``: Return parsed symbol ids for a source file.
            - ``init``: Create initial sidecar content.
            - ``update``: Apply updates and optionally trigger re-ingest.
        """

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
        """Run mesh/sidecar integrity checks via ``smak doctor``."""

        return self._run_cli(["doctor", "--path", path, "--config", self.config_name])


def build_mcp_server(workspace_root: str | Path = ".") -> FastMCP:
    """Build the FastMCP instance and register SMAK tools.

    The returned server exposes tool signatures that are intentionally
    straightforward for AI agents to plan against.
    """

    smak_server = SmakMcpServer(workspace_root=Path(workspace_root).resolve())
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def refresh_knowledge(
        folder: str = ".",
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Refresh vector knowledge by ingesting files from ``folder``."""

        return smak_server.refresh_knowledge(
            folder=folder,
            index=index,
            follow_symlinks=follow_symlinks,
        )

    @mcp.tool()
    def semantic_search(query: str, index: str = "source_code", top_k: int = 5) -> dict[str, Any]:
        """Search for relevant knowledge and one-hop related context."""

        return smak_server.semantic_search(query=query, index=index, top_k=top_k)

    @mcp.tool()
    def manage_sidecar(
        action: str,
        file_path: str,
        updates: list[dict[str, Any]] | None = None,
        reingest: bool = False,
        index: str = "source_code",
    ) -> dict[str, Any] | list[str] | str:
        """Inspect/init/update sidecar annotations for a source file."""

        return smak_server.manage_sidecar(
            action=action,
            file_path=file_path,
            updates=updates,
            reingest=reingest,
            index=index,
        )

    @mcp.tool()
    def validate_mesh(path: str = ".") -> str:
        """Validate sidecar and mesh consistency for the given path."""

        return smak_server.validate_mesh(path=path)

    return mcp


def main() -> None:
    """Run the SMAK MCP server over stdio transport."""

    server = build_mcp_server(Path.cwd())
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
