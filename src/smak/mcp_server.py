"""MCP bridge for exposing SMAK services as tool-callable operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from smak.config import EmbeddingConfig, SmakConfig, load_config, load_embedding_config
from smak.factory import (
    create_doctor_service,
    create_query_service,
    create_sidecar_service,
    init_config,
    load_and_validate_vector_store,
    on_ghost_source,
    sidecar_skip_file,
)
from smak.services import IngestService
from smak.utils.embedding import InternalNomicEmbedding

_DEFAULT_EMBEDDING_SETUP = str(Path(__file__).resolve().parent / "embedding_setup.yaml")


@dataclass
class SmakMcpServer:
    """In-process adapter used by MCP tool handlers.

    Bound to a single workspace configuration file.
    """

    config_path: Path
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    _config: SmakConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {self.config_path}."
            )
        self._config = init_config(
            load_config(self.config_path),
            embedding_config=self.embedding_config,
        )

    def _get_config(self) -> SmakConfig:
        return self._config

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
        vector_store = load_and_validate_vector_store(index_config, config)
        return vector_store, index_config

    @staticmethod
    def _format_candidates(candidates: list[str]) -> str:
        if len(candidates) == 1:
            return f"'{candidates[0]}'"
        if len(candidates) == 2:
            return f"'{candidates[0]}' or '{candidates[1]}'"
        quoted = [f"'{candidate}'" for candidate in candidates]
        return ", ".join(quoted[:-1]) + f", or {quoted[-1]}"

    def _resolve_source_path(
        self,
        index: str,
        index_config: object,
        file_path: str,
    ) -> Path:
        """Resolve *file_path* with defensive fallback across all index roots."""

        index_roots = [Path(p).resolve() for p in index_config.paths if Path(p).is_dir()]
        index_files = [Path(p).resolve() for p in index_config.paths if Path(p).is_file()]
        raw_source_path = Path(file_path)

        if raw_source_path.is_absolute():
            if raw_source_path.exists():
                return raw_source_path
        else:
            for index_root in index_roots:
                primary_path = (index_root / raw_source_path).resolve()
                if primary_path.exists():
                    return primary_path

        file_name = raw_source_path.name
        all_candidates: list[Path] = []
        for idx_file in index_files:
            if idx_file.name == file_name:
                all_candidates.append(idx_file)
        for index_root in index_roots:
            all_candidates.extend(index_root.rglob(f"*{file_name}") if file_name else [])
        candidates = sorted(all_candidates)

        if len(candidates) == 1:
            return candidates[0].resolve()

        if len(candidates) > 1:
            relative_candidates: list[str] = []
            for candidate in candidates:
                for index_root in index_roots:
                    try:
                        relative_candidates.append(
                            str(candidate.resolve().relative_to(index_root))
                        )
                        break
                    except ValueError:
                        continue
                else:
                    relative_candidates.append(str(candidate.resolve()))
            hints = self._format_candidates(relative_candidates)
            raise ValueError(f"Ambiguous file path '{file_path}'. Did you mean {hints}?")

        roots_display = ", ".join(f"'{r}'" for r in index_roots)
        raise FileNotFoundError(
            f"File path resolution failed under index roots [{roots_display}] "
            f"(index='{index}', file_path='{file_path}'). "
            "Try using semantic_search first and pass one of the returned file paths."
        )

    # ------------------------------------------------------------------
    # Public service methods (no config parameter — single config)
    # ------------------------------------------------------------------

    def refresh_knowledge(
        self,
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Ingest content into a target index."""

        cfg = self._get_config()
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        target_paths = [Path(p) for p in index_config.paths]
        service = IngestService(vector_store=vector_store)
        emb_cfg = self.embedding_config
        stats = service.ingest_paths(
            target_paths,
            follow_symlinks=follow_symlinks,
            skip_file=sidecar_skip_file,
            on_ghost_source=on_ghost_source,
            embedder_loader=lambda: InternalNomicEmbedding(embedding_config=emb_cfg),
        )
        return (
            "Ingestion Complete! "
            f"Processed Files: {stats.files}, "
            f"Skipped Files: {stats.skipped}, Vectors Added: {stats.vectors}"
        )

    def semantic_search(
        self,
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run semantic query and return serializable payload."""

        cfg = self._get_config()
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        service = create_query_service(
            vector_store, cfg, index_config, embedding_config=self.embedding_config,
        )
        result = service.search(query, top_k=top_k)
        return result if isinstance(result, dict) else {}

    def list_available_indices(self) -> list[dict[str, str]]:
        """Return all index names/descriptions."""

        cfg = self._get_config()
        return [
            {"name": index.name, "description": index.description}
            for index in cfg.indices
        ]

    def inspect_sidecar(
        self,
        file_path: str,
        index: str = "source_code",
    ) -> list[str]:
        """Return the list of symbol UIDs parsed from a source file."""

        cfg = self._get_config()
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(index, index_config, file_path)
        return create_sidecar_service().inspect(source_path)

    def update_sidecar(
        self,
        file_path: str,
        index: str = "source_code",
        symbol: str | None = None,
        intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sync or update sidecar metadata for a source file."""

        cfg = self._get_config()
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(index, index_config, file_path)
        return create_sidecar_service().update(
            source_path, symbol=symbol, intent=intent, relations=relations
        )

    def clear_sidecar_symbol(
        self,
        file_path: str,
        symbol: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Remove a single symbol entry from a sidecar file."""

        cfg = self._get_config()
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(index, index_config, file_path)
        return create_sidecar_service().clear_symbol(source_path, symbol)

    def lookup_symbol(
        self,
        uid: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Check whether a UID exists in the vector store."""

        cfg = self._get_config()
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        service = create_query_service(
            vector_store, cfg, index_config, embedding_config=self.embedding_config,
        )
        return service.lookup(uid)

    def validate_mesh(self) -> str:
        """Run mesh/sidecar integrity checks."""

        cfg = self._get_config()
        service = create_doctor_service(cfg)
        service.validate_all()
        return "Mesh diagnostics passed."

    # ------------------------------------------------------------------
    # Composite tools
    # ------------------------------------------------------------------

    def multi_index_search(
        self,
        query: str,
        indices: list[str] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Search across multiple (or all) indices and return aggregated results."""

        cfg = self._get_config()
        target_indices = indices or [idx.name for idx in cfg.indices]
        results: dict[str, Any] = {}
        for index_name in target_indices:
            index_config = cfg.get_index(index_name)
            if index_config is None:
                results[index_name] = {"error": f"Index '{index_name}' not found"}
                continue
            try:
                vector_store = load_and_validate_vector_store(index_config, cfg)
                service = create_query_service(
                    vector_store, cfg, index_config,
                    embedding_config=self.embedding_config,
                )
                results[index_name] = service.search(query, top_k=top_k)
            except Exception as exc:
                results[index_name] = {"error": str(exc)}
        return results

    def workspace_status(self) -> dict[str, Any]:
        """Return health dashboard with per-index stats."""

        cfg = self._get_config()
        index_stats: list[dict[str, Any]] = []
        for index_config in cfg.indices:
            stat: dict[str, Any] = {
                "name": index_config.name,
                "description": index_config.description,
                "paths": index_config.paths,
            }
            try:
                vector_store = load_and_validate_vector_store(index_config, cfg)
                stat["vector_count"] = vector_store.count()
                stat["last_update"] = vector_store.last_update()
            except Exception as exc:
                stat["error"] = str(exc)
            index_stats.append(stat)
        return {"indices": index_stats}

    def batch_update_sidecars(
        self,
        file_paths: list[str],
        index: str = "source_code",
    ) -> list[dict[str, Any]]:
        """Sync sidecars for multiple files at once."""

        cfg = self._get_config()
        index_config = self._get_index_config(cfg, index)
        sidecar_service = create_sidecar_service()
        results: list[dict[str, Any]] = []
        for fp in file_paths:
            try:
                source_path = self._resolve_source_path(index, index_config, fp)
                result = sidecar_service.update(source_path)
                results.append(result)
            except Exception as exc:
                results.append({"file_path": fp, "error": str(exc)})
        return results


def build_mcp_server(
    config_path: str | Path,
    embedding_setup: str | Path | None = None,
) -> FastMCP:
    """Build the FastMCP instance and register SMAK tools."""

    emb_cfg = load_embedding_config(embedding_setup)
    smak_server = SmakMcpServer(
        config_path=Path(config_path).resolve(),
        embedding_config=emb_cfg,
    )
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def refresh_knowledge(
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Ingest (or re-ingest) a source folder into a SMAK vector-store index.

        Args:
            index: Name of the index to refresh.  Defaults to ``"source_code"``.
            follow_symlinks: Whether to follow symbolic links.  Defaults to ``True``.

        Returns:
            A human-readable summary of ingestion results.
        """
        return smak_server.refresh_knowledge(index=index, follow_symlinks=follow_symlinks)

    @mcp.tool()
    def semantic_search(
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search a SMAK index with a natural-language query.

        Args:
            query: Natural-language description of the intent or behavior.
            index: Name of the index to query.
            top_k: Maximum number of results.  Defaults to ``5``.

        Returns:
            Dict with ``hits`` and ``related_context``.
        """
        return smak_server.semantic_search(query=query, index=index, top_k=top_k)

    @mcp.tool()
    def list_available_indices() -> list[dict[str, str]]:
        """List searchable indices for this workspace.

        Returns:
            Ordered list of objects containing ``name`` and ``description``.
        """
        return smak_server.list_available_indices()

    @mcp.tool()
    def inspect_sidecar(
        file_path: str,
        index: str = "source_code",
    ) -> list[str]:
        """List the symbol UIDs parsed from a source file.

        Args:
            file_path: Path to the source file.
            index: Index whose root resolves relative paths.

        Returns:
            Ordered list of symbol UID strings.
        """
        return smak_server.inspect_sidecar(file_path=file_path, index=index)

    @mcp.tool()
    def update_sidecar(
        file_path: str,
        index: str = "source_code",
        symbol: str | None = None,
        intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sync or update sidecar metadata for a source file.

        Args:
            file_path: Path to the source file.
            index: Index whose root resolves relative paths.
            symbol: UID of a single symbol to update.
            intent: New intent description (only with ``symbol``).
            relations: New relation list (only with ``symbol``).

        Returns:
            A dict describing the result.
        """
        return smak_server.update_sidecar(
            file_path=file_path, index=index,
            symbol=symbol, intent=intent, relations=relations,
        )

    @mcp.tool()
    def clear_sidecar_symbol(
        file_path: str,
        symbol: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Remove a symbol entry from a sidecar file.

        Args:
            file_path: Path to the source file.
            symbol: Exact UID of the symbol to remove.
            index: Index whose root resolves relative paths.

        Returns:
            A dict with cleared symbol info.
        """
        return smak_server.clear_sidecar_symbol(
            file_path=file_path, symbol=symbol, index=index,
        )

    @mcp.tool()
    def lookup_symbol(
        uid: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Check whether a specific UID exists in the SMAK vector store.

        Args:
            uid: Full UID to look up.
            index: Index to query.

        Returns:
            A dict with ``found`` (bool) and ``uid``.
        """
        return smak_server.lookup_symbol(uid=uid, index=index)

    @mcp.tool()
    def validate_mesh() -> str:
        """Run integrity diagnostics across the SMAK mesh.

        Returns:
            ``"Mesh diagnostics passed."`` if no issues.
        """
        return smak_server.validate_mesh()

    @mcp.tool()
    def multi_index_search(
        query: str,
        indices: list[str] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Search across multiple (or all) indices in parallel.

        Returns aggregated results grouped by index name.

        Args:
            query: Natural-language query.
            indices: List of index names to search. ``None`` = all indices.
            top_k: Max results per index.  Defaults to ``3``.

        Returns:
            Dict mapping index name to search results.
        """
        return smak_server.multi_index_search(query=query, indices=indices, top_k=top_k)

    @mcp.tool()
    def workspace_status() -> dict[str, Any]:
        """Return health dashboard with per-index stats.

        Returns:
            Dict with ``indices`` list containing per-index stats.
        """
        return smak_server.workspace_status()

    @mcp.tool()
    def batch_update_sidecars(
        file_paths: list[str],
        index: str = "source_code",
    ) -> list[dict[str, Any]]:
        """Sync sidecars for multiple files at once.

        Args:
            file_paths: List of file paths to sync.
            index: Index whose root resolves relative paths.

        Returns:
            List of per-file results.
        """
        return smak_server.batch_update_sidecars(file_paths=file_paths, index=index)

    return mcp


def main() -> None:
    """Run the SMAK MCP server over stdio transport."""

    import argparse

    parser = argparse.ArgumentParser(description="SMAK MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        default="workspace_config.yaml",
        help="Path to workspace_config.yaml (Mandatory)",
    )
    parser.add_argument(
        "--embedding-setup",
        type=str,
        default=_DEFAULT_EMBEDDING_SETUP,
        help="Path to embedding_setup.yaml",
    )
    args = parser.parse_args()

    server = build_mcp_server(
        config_path=Path(args.config).resolve(),
        embedding_setup=args.embedding_setup,
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
