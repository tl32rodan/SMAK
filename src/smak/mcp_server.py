"""MCP server tools for the SMAK passive knowledge kernel."""

from __future__ import annotations

import json
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
        cmd = [self.smak_binary, *args]
        completed = subprocess.run(
            cmd,
            cwd=self.workspace_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "Unknown CLI failure"
            raise RuntimeError(message)
        return completed.stdout.strip()

    def refresh_knowledge(self, folder: str = ".", index: str = "source_code") -> str:
        return self._run_cli(
            [
                "ingest",
                "--folder",
                folder,
                "--index",
                index,
                "--config",
                self.config_name,
            ]
        )

    def inspect_file_symbols(self, file_path: str) -> list[str]:
        output = self._run_cli(["search", file_path, "--config", self.config_name, "--json-output"])
        parsed = json.loads(output) if output else []
        return [str(symbol) for symbol in parsed]

    def semantic_search(
        self,
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        output = self._run_cli(
            [
                "query",
                query,
                "--index",
                index,
                "--top-k",
                str(top_k),
                "--config",
                self.config_name,
            ]
        )
        parsed = json.loads(output) if output else []
        return parsed if isinstance(parsed, list) else []

    def update_sidecar_metadata(
        self,
        file_path: str,
        updates: list[dict[str, Any]],
        reingest: bool = False,
        index: str = "source_code",
    ) -> dict[str, Any]:
        command = [
            "sidecar",
            "update",
            file_path,
            "--updates",
            json.dumps(updates, ensure_ascii=False),
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

    def validate_mesh(self, path: str = ".") -> str:
        return self._run_cli(["doctor", "--path", path])


def build_mcp_server(workspace_root: str | Path = ".") -> FastMCP:
    smak_server = SmakMcpServer(workspace_root=Path(workspace_root).resolve())
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def refresh_knowledge(folder: str = ".", index: str = "source_code") -> str:
        """Trigger incremental ingest to refresh SMAK knowledge."""

        return smak_server.refresh_knowledge(folder=folder, index=index)

    @mcp.tool()
    def inspect_file_symbols(file_path: str) -> list[str]:
        """Inspect canonical symbol IDs from a source file."""

        return smak_server.inspect_file_symbols(file_path)

    @mcp.tool()
    def semantic_search(
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Run semantic retrieval over the vector store."""

        return smak_server.semantic_search(query=query, index=index, top_k=top_k)

    @mcp.tool()
    def update_sidecar_metadata(
        file_path: str,
        updates: list[dict[str, Any]],
        reingest: bool = False,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Update structured sidecar metadata for symbols."""

        return smak_server.update_sidecar_metadata(
            file_path=file_path,
            updates=updates,
            reingest=reingest,
            index=index,
        )

    @mcp.tool()
    def validate_mesh(path: str = ".") -> str:
        """Validate sidecar/source mesh consistency."""

        return smak_server.validate_mesh(path=path)

    return mcp


def main() -> None:
    server = build_mcp_server(Path.cwd())
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
