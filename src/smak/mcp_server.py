"""MCP bridge for exposing SMAK services as tool-callable operations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from smak.cli import _load_vector_store
from smak.config import SmakConfig, load_config
from smak.services import DoctorService, IngestService, QueryService, SidecarService
from smak.services.relation_resolver import SidecarRelationResolver
from smak.sidecar.store import SidecarStore
from smak.utils.embedding import (
    InternalNomicEmbedding,
    initialize_embedding_dimensions,
    validate_vector_store_dimension,
)
from smak.utils.yaml import safe_load


@dataclass
class SmakMcpServer:
    """In-process adapter used by MCP tool handlers."""

    registry_path: Path
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

    def _get_workspace_path(self, workspace_name: str) -> Path:
        """Resolve the absolute directory path for a workspace."""

        if workspace_name not in self.workspaces:
            raise ValueError(f"Workspace '{workspace_name}' not found in registry.")

        base_dir = self.registry_path.parent
        target_path = Path(self.workspaces[workspace_name]["path"])
        if not target_path.is_absolute():
            target_path = (base_dir / target_path).resolve()

        return target_path

    def _get_workspace_context(self, workspace_name: str) -> tuple[Path, SmakConfig]:
        """Returns the base_path and loaded SmakConfig for a workspace."""

        base_path = self._get_workspace_path(workspace_name)
        config_path = base_path / self.config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        config = load_config(config_path)
        config = initialize_embedding_dimensions(config, InternalNomicEmbedding())
        return base_path, config

    def _load_workspace_vector_store(
        self,
        config: SmakConfig,
        index: str,
    ) -> object:
        index_config = next((entry for entry in config.indices if entry.name == index), None)
        if index_config is None:
            raise ValueError(f"Index '{index}' not found in configuration.")
        vector_store = _load_vector_store(index_config, config)
        validate_vector_store_dimension(vector_store, config.embedding_dimensions)
        return vector_store

    def refresh_knowledge(
        self,
        workspace: str,
        folder: str = ".",
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Ingest workspace content into a target index."""

        base_path, config = self._get_workspace_context(workspace)
        from smak.cli import _resolve_config
        config = _resolve_config(config, str(base_path / self.config_name))
        vector_store = self._load_workspace_vector_store(config, index)
        folder_path = Path(folder)
        target_folder = (
            folder_path if folder_path.is_absolute() else (base_path / folder_path).resolve()
        )
        service = IngestService(vector_store=vector_store)
        stats = service.ingest_folder(
            target_folder,
            follow_symlinks=follow_symlinks,
        )
        return (
            "Ingestion Complete! "
            f"Processed Files: {stats.files}, "
            f"Skipped Files: {stats.skipped}, Vectors Added: {stats.vectors}"
        )

    def semantic_search(
        self,
        workspace: str,
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run in-process semantic query and return serializable payload."""

        base_path, config = self._get_workspace_context(workspace)
        from smak.cli import _resolve_config
        config = _resolve_config(config, str(base_path / self.config_name))
        index_config = next((entry for entry in config.indices if entry.name == index), None)
        if index_config is None:
            raise ValueError(f"Index '{index}' not found in configuration.")

        vector_store = self._load_workspace_vector_store(config, index)
        sidecar_store = SidecarStore()
        service = QueryService(
            vector_store=vector_store,
            config=config,
            index_config=index_config,
            vector_store_loader=lambda idx, cfg: _load_vector_store(
                idx,
                cfg,
            ),
            relation_resolver=SidecarRelationResolver(sidecar_store),
        )
        result = service.search(query, top_k=top_k)
        return result if isinstance(result, dict) else {}

    def manage_sidecar(
        self,
        workspace: str,
        action: str,
        file_path: str,
        updates: list[dict[str, Any]] | None = None,
        reingest: bool = False,
        index: str = "source_code",
    ) -> dict[str, Any] | list[str] | str:
        """Manage sidecar metadata through one unified entrypoint."""

        base_path, config = self._get_workspace_context(workspace)
        from smak.cli import _resolve_config
        config = _resolve_config(config, str(base_path / self.config_name))
        raw_source_path = Path(file_path)
        source_path = (
            raw_source_path
            if raw_source_path.is_absolute()
            else (base_path / raw_source_path).resolve()
        )
        sidecar_store = SidecarStore()
        service = SidecarService(sidecar_store=sidecar_store)

        if action == "inspect":
            return service.inspect(source_path)
        if action == "init":
            output = service.init(source_path)
            return str(output)
        if action == "update":
            update_result = service.update(
                source_path,
                json.dumps(updates or [], ensure_ascii=False),
            )
            if reingest:
                vector_store = self._load_workspace_vector_store(config, index)
                ingest_service = IngestService(vector_store=vector_store)
                ingest_stats = ingest_service.ingest_folder(
                    source_path.parent,
                )
                update_result["reingest"] = {
                    "files": ingest_stats.files,
                    "vectors": ingest_stats.vectors,
                    "skipped": ingest_stats.skipped,
                }
            return update_result
        raise ValueError("action must be one of: init, update, inspect")

    def validate_mesh(self, workspace: str, path: str = ".") -> str:
        """Run mesh/sidecar integrity checks in-process."""

        base_path, config = self._get_workspace_context(workspace)
        from smak.cli import _resolve_config
        config = _resolve_config(config, str(base_path / self.config_name))
        path_obj = Path(path)
        target_path = path_obj if path_obj.is_absolute() else (base_path / path_obj).resolve()

        def _load_store(index_name: str) -> object:
            return self._load_workspace_vector_store(config, index_name)

        service = DoctorService(config=config, vector_store_loader=_load_store)
        issues = service.validate_sidecars(target_path)
        dangling = service.validate_mesh_integrity(target_path)
        problems = [*issues, *dangling]
        if problems:
            raise RuntimeError("\n".join(problems))
        return "Mesh diagnostics passed."


def build_mcp_server(registry_path: str | Path) -> FastMCP:
    """Build the FastMCP instance and register SMAK tools."""

    smak_server = SmakMcpServer(registry_path=Path(registry_path).resolve())
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def list_available_workspaces() -> dict[str, Any]:
        return smak_server.workspaces

    @mcp.tool()
    def refresh_knowledge(
        workspace: str,
        folder: str = ".",
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
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
