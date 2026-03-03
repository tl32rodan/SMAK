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
    configs: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Registry file not found at {self.registry_path}. "
                "A registry file (e.g., registry.yaml) is mandatory."
            )

        content = self.registry_path.read_text(encoding="utf-8")
        data = safe_load(content) or {}
        self.configs = data.get("configs", {})

        if not self.configs:
            raise ValueError(
                "Registry file must contain at least one entry under the 'configs' key."
            )

    def _resolve_config_path(self, config_name: str) -> Path:
        """Resolve the absolute path to a config file from a registry entry."""

        if config_name not in self.configs:
            raise ValueError(f"Config '{config_name}' not found in registry.")

        base_dir = self.registry_path.parent
        target_path = Path(self.configs[config_name]["config_path"])
        if not target_path.is_absolute():
            target_path = (base_dir / target_path).resolve()

        return target_path

    def _load_config(self, config_name: str) -> SmakConfig:
        """Load and return the resolved SmakConfig for a registry entry."""

        config_path = self._resolve_config_path(config_name)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        config = load_config(config_path)
        config = initialize_embedding_dimensions(config, InternalNomicEmbedding())
        return config

    @staticmethod
    def _get_index_config(config: SmakConfig, index: str) -> object:
        index_config = config.get_index(index)
        if index_config is None:
            raise ValueError(f"Index '{index}' not found in configuration.")
        return index_config

    def _load_index_vector_store(
        self,
        config: SmakConfig,
        index: str,
    ) -> tuple[object, object]:
        index_config = self._get_index_config(config, index)
        vector_store = _load_vector_store(index_config, config)
        validate_vector_store_dimension(vector_store, config.embedding_dimensions)
        return vector_store, index_config

    def refresh_knowledge(
        self,
        config: str,
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Ingest content into a target index (folder is defined by the index config)."""

        cfg = self._load_config(config)
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        target_folder = Path(index_config.path)
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
        config: str,
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run in-process semantic query and return serializable payload."""

        cfg = self._load_config(config)
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        sidecar_store = SidecarStore()
        service = QueryService(
            vector_store=vector_store,
            config=cfg,
            index_config=index_config,
            vector_store_loader=_load_vector_store,
            relation_resolver=SidecarRelationResolver(sidecar_store),
        )
        result = service.search(query, top_k=top_k)
        return result if isinstance(result, dict) else {}

    def manage_sidecar(
        self,
        config: str,
        action: str,
        file_path: str,
        updates: list[dict[str, Any]] | None = None,
        index: str = "source_code",
    ) -> dict[str, Any] | list[str] | str:
        """Manage sidecar metadata through one unified entrypoint."""

        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        raw_source_path = Path(file_path)
        source_path = (
            raw_source_path
            if raw_source_path.is_absolute()
            else (Path(index_config.path) / raw_source_path).resolve()
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
            return update_result
        raise ValueError("action must be one of: init, update, inspect")

    def validate_mesh(self, config: str) -> str:
        """Run mesh/sidecar integrity checks in-process."""

        cfg = self._load_config(config)

        def _load_store(index_name: str) -> object:
            store, _ = self._load_index_vector_store(cfg, index_name)
            return store

        service = DoctorService(config=cfg, vector_store_loader=_load_store)
        service.validate_all()
        return "Mesh diagnostics passed."


def build_mcp_server(registry_path: str | Path) -> FastMCP:
    """Build the FastMCP instance and register SMAK tools."""

    smak_server = SmakMcpServer(registry_path=Path(registry_path).resolve())
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def list_available_configs() -> dict[str, Any]:
        return smak_server.configs

    @mcp.tool()
    def refresh_knowledge(
        config: str,
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        return smak_server.refresh_knowledge(
            config=config,
            index=index,
            follow_symlinks=follow_symlinks,
        )

    @mcp.tool()
    def semantic_search(
        config: str,
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> dict[str, Any]:
        return smak_server.semantic_search(
            config=config,
            query=query,
            index=index,
            top_k=top_k,
        )

    @mcp.tool()
    def manage_sidecar(
        config: str,
        action: str,
        file_path: str,
        updates: list[dict[str, Any]] | None = None,
        index: str = "source_code",
    ) -> dict[str, Any] | list[str] | str:
        return smak_server.manage_sidecar(
            config=config,
            action=action,
            file_path=file_path,
            updates=updates,
            index=index,
        )

    @mcp.tool()
    def validate_mesh(config: str) -> str:
        return smak_server.validate_mesh(config=config)

    return mcp


def main() -> None:
    """Run the SMAK MCP server over stdio transport."""

    import argparse

    parser = argparse.ArgumentParser(description="SMAK MCP Server")
    parser.add_argument(
        "--registry",
        type=str,
        default="registry.yaml",
        help="Path to registry.yaml file (Mandatory)",
    )
    args = parser.parse_args()

    server = build_mcp_server(registry_path=Path(args.registry).resolve())
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
