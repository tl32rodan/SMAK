"""MCP bridge for exposing SMAK CLI capabilities as tool-callable operations.

This module intentionally keeps business logic in CLI commands (``smak ...``) and
provides a thin wrapper for MCP/agent integrations.

Strict multi-workspace mode is enforced via a registry file that defines all
available workspaces. All operational tools require an explicit ``workspace``
parameter and commands are routed by changing the process cwd.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from smak.utils.yaml import safe_load


@dataclass
class SmakMcpServer:
    """CLI-backed adapter used by MCP tool handlers.

    Attributes:
        registry_path: Path to the mandatory multi-workspace registry YAML file.
        smak_binary: CLI executable name/path (defaults to ``smak``).
        config_name: Workspace config file passed to SMAK commands.
        workspaces: Parsed workspace registry entries keyed by workspace name.
    """

    registry_path: Path
    smak_binary: str = "smak"
    config_name: str = "workspace_config.yaml"
    workspaces: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Registry file not found at {self.registry_path}. "
                "A registry file (e.g., all_workspaces.yaml) is mandatory."
            )

        content = self.registry_path.read_text(encoding="utf-8")
        data = safe_load(content) or {}
        self.workspaces = data.get("workspaces", {})

        if not self.workspaces:
            raise ValueError(
                "Registry file must contain at least one workspace under the 'workspaces' key."
            )

    def _get_workspace_cwd(self, workspace_name: str) -> Path:
        """Resolve the target directory for a given workspace name."""

        if workspace_name not in self.workspaces:
            raise ValueError(f"Workspace '{workspace_name}' not found in registry.")

        base_dir = self.registry_path.parent
        target_path = Path(self.workspaces[workspace_name]["path"])
        if not target_path.is_absolute():
            target_path = (base_dir / target_path).resolve()

        return target_path

    def _run_cli(self, args: list[str], workspace: str) -> str:
        """Execute a SMAK CLI command and return stdout.

        Raises:
            RuntimeError: If the underlying command exits with non-zero status.
        """

        completed = subprocess.run(
            [self.smak_binary, *args],
            cwd=self._get_workspace_cwd(workspace),
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
        workspace: str,
        folder: str = ".",
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Ingest workspace content into a target index."""

        command = ["ingest", "--folder", folder, "--index", index, "--config", self.config_name]
        if not follow_symlinks:
            command.append("--no-follow-symlinks")
        return self._run_cli(command, workspace=workspace)

    def semantic_search(
        self,
        workspace: str,
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run ``smak query`` and parse the JSON response payload."""

        output = self._run_cli(
            ["query", query, "--index", index, "--top-k", str(top_k), "--config", self.config_name],
            workspace=workspace,
        )
        parsed = json.loads(output) if output else {}
        return parsed if isinstance(parsed, dict) else {}

    def manage_sidecar(
        self,
        workspace: str,
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
                ["sidecar", "inspect", file_path, "--config", self.config_name, "--json-output"],
                workspace=workspace,
            )
            parsed = json.loads(output) if output else []
            return [str(symbol) for symbol in parsed]
        if action == "init":
            return self._run_cli(
                ["sidecar", "init", file_path, "--config", self.config_name], workspace=workspace
            )
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
            output = self._run_cli(command, workspace=workspace)
            parsed = json.loads(output) if output else {}
            return parsed if isinstance(parsed, dict) else {}
        raise ValueError("action must be one of: init, update, inspect")

    def validate_mesh(self, workspace: str, path: str = ".") -> str:
        """Run mesh/sidecar integrity checks via ``smak doctor``."""

        return self._run_cli(
            ["doctor", "--path", path, "--config", self.config_name],
            workspace=workspace,
        )


def build_mcp_server(registry_path: str | Path) -> FastMCP:
    """Build the FastMCP instance and register SMAK tools."""

    smak_server = SmakMcpServer(registry_path=Path(registry_path).resolve())
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def list_available_workspaces() -> dict[str, Any]:
        """List all available workspaces and descriptions.

        Call this first to choose a workspace for other tools.
        """

        return smak_server.workspaces

    @mcp.tool()
    def refresh_knowledge(
        workspace: str,
        folder: str = ".",
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Refresh vector knowledge by ingesting files from ``folder``."""

        return smak_server.refresh_knowledge(
            workspace=workspace,
            folder=folder,
            index=index,
            follow_symlinks=follow_symlinks,
        )

    @mcp.tool()
    def semantic_search(
        workspace: str,
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search for relevant knowledge and one-hop related context."""

        return smak_server.semantic_search(
            workspace=workspace,
            query=query,
            index=index,
            top_k=top_k,
        )

    @mcp.tool()
    def manage_sidecar(
        workspace: str,
        action: str,
        file_path: str,
        updates: list[dict[str, Any]] | None = None,
        reingest: bool = False,
        index: str = "source_code",
    ) -> dict[str, Any] | list[str] | str:
        """Inspect/init/update sidecar annotations for a source file."""

        return smak_server.manage_sidecar(
            workspace=workspace,
            action=action,
            file_path=file_path,
            updates=updates,
            reingest=reingest,
            index=index,
        )

    @mcp.tool()
    def validate_mesh(workspace: str, path: str = ".") -> str:
        """Validate sidecar and mesh consistency for the given path."""

        return smak_server.validate_mesh(workspace=workspace, path=path)

    return mcp


def main() -> None:
    """Run the SMAK MCP server over stdio transport."""

    import argparse

    parser = argparse.ArgumentParser(description="SMAK MCP Server")
    parser.add_argument(
        "--registry",
        type=str,
        default="all_workspaces.yaml",
        help="Path to all_workspaces.yaml registry file (Mandatory)",
    )
    args = parser.parse_args()

    server = build_mcp_server(registry_path=Path(args.registry).resolve())
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
