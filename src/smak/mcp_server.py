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
    """Stateless MCP adapter.  Config is loaded per-call and cached."""

    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    _config_cache: dict[str, SmakConfig] = field(
        init=False, default_factory=dict, repr=False,
    )

    # ------------------------------------------------------------------
    # Config loading with cache
    # ------------------------------------------------------------------

    def _load_config(self, config: str) -> SmakConfig:
        """Load and cache a workspace config by path."""
        resolved = str(Path(config).resolve())
        if resolved in self._config_cache:
            return self._config_cache[resolved]
        path = Path(resolved)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {resolved}.")
        cfg = init_config(
            load_config(path), embedding_config=self.embedding_config,
        )
        self._config_cache[resolved] = cfg
        return cfg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_index_config(config: SmakConfig, index: str) -> object:
        index_config = config.get_index(index)
        if index_config is None:
            raise ValueError(f"Index '{index}' not found in configuration.")
        return index_config

    def _load_index_vector_store(
        self, config: SmakConfig, index: str,
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
        quoted = [f"'{c}'" for c in candidates]
        return ", ".join(quoted[:-1]) + f", or {quoted[-1]}"

    def _resolve_source_path(
        self, index: str, index_config: object, file_path: str,
    ) -> Path:
        """Resolve *file_path* with defensive fallback across all index roots."""
        index_roots = [Path(p).resolve() for p in index_config.paths if Path(p).is_dir()]
        index_files = [Path(p).resolve() for p in index_config.paths if Path(p).is_file()]
        raw = Path(file_path)

        if raw.is_absolute():
            if raw.exists():
                return raw
        else:
            for root in index_roots:
                candidate = (root / raw).resolve()
                if candidate.exists():
                    return candidate

        name = raw.name
        all_candidates: list[Path] = []
        for f in index_files:
            if f.name == name:
                all_candidates.append(f)
        for root in index_roots:
            all_candidates.extend(root.rglob(f"*{name}") if name else [])
        candidates = sorted(all_candidates)

        if len(candidates) == 1:
            return candidates[0].resolve()

        if len(candidates) > 1:
            rel: list[str] = []
            for c in candidates:
                for root in index_roots:
                    try:
                        rel.append(str(c.resolve().relative_to(root)))
                        break
                    except ValueError:
                        continue
                else:
                    rel.append(str(c.resolve()))
            raise ValueError(
                f"Ambiguous file path '{file_path}'. Did you mean {self._format_candidates(rel)}?"
            )

        roots_display = ", ".join(f"'{r}'" for r in index_roots)
        raise FileNotFoundError(
            f"File path resolution failed under index roots [{roots_display}] "
            f"(index='{index}', file_path='{file_path}'). "
            "Try using semantic_search first and pass one of the returned file paths."
        )

    # ------------------------------------------------------------------
    # Public service methods — every method takes config path
    # ------------------------------------------------------------------

    def refresh_knowledge(
        self, config: str, index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        cfg = self._load_config(config)
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
        self, config: str, query: str,
        index: str = "source_code", top_k: int = 5,
    ) -> dict[str, Any]:
        cfg = self._load_config(config)
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        service = create_query_service(
            vector_store, cfg, index_config, embedding_config=self.embedding_config,
        )
        result = service.search(query, top_k=top_k)
        return result if isinstance(result, dict) else {}

    def list_available_indices(self, config: str) -> list[dict[str, str]]:
        cfg = self._load_config(config)
        return [
            {"name": idx.name, "description": idx.description}
            for idx in cfg.indices
        ]

    def inspect_sidecar(
        self, config: str, file_path: str, index: str = "source_code",
    ) -> list[str]:
        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(index, index_config, file_path)
        return create_sidecar_service().inspect(source_path)

    def update_sidecar(
        self, config: str, file_path: str, index: str = "source_code",
        symbol: str | None = None, intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(index, index_config, file_path)
        return create_sidecar_service().update(
            source_path, symbol=symbol, intent=intent, relations=relations,
        )

    def clear_sidecar_symbol(
        self, config: str, file_path: str, symbol: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(index, index_config, file_path)
        return create_sidecar_service().clear_symbol(source_path, symbol)

    def lookup_symbol(
        self, config: str, uid: str, index: str = "source_code",
    ) -> dict[str, Any]:
        cfg = self._load_config(config)
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        service = create_query_service(
            vector_store, cfg, index_config, embedding_config=self.embedding_config,
        )
        return service.lookup(uid)

    def validate_mesh(self, config: str) -> str:
        cfg = self._load_config(config)
        service = create_doctor_service(cfg)
        service.validate_all()
        return "Mesh diagnostics passed."

    # ------------------------------------------------------------------
    # Composite tools
    # ------------------------------------------------------------------

    def multi_index_search(
        self, config: str, query: str,
        indices: list[str] | None = None, top_k: int = 3,
    ) -> dict[str, Any]:
        cfg = self._load_config(config)
        target_indices = indices or [idx.name for idx in cfg.indices]
        results: dict[str, Any] = {}
        for index_name in target_indices:
            index_config = cfg.get_index(index_name)
            if index_config is None:
                results[index_name] = {"error": f"Index '{index_name}' not found"}
                continue
            try:
                vs = load_and_validate_vector_store(index_config, cfg)
                svc = create_query_service(
                    vs, cfg, index_config, embedding_config=self.embedding_config,
                )
                results[index_name] = svc.search(query, top_k=top_k)
            except Exception as exc:
                results[index_name] = {"error": str(exc)}
        return results

    def workspace_status(self, config: str) -> dict[str, Any]:
        cfg = self._load_config(config)
        stats: list[dict[str, Any]] = []
        for index_config in cfg.indices:
            stat: dict[str, Any] = {
                "name": index_config.name,
                "description": index_config.description,
                "paths": index_config.paths,
            }
            try:
                vs = load_and_validate_vector_store(index_config, cfg)
                stat["vector_count"] = vs.count()
                stat["last_update"] = vs.last_update()
            except Exception as exc:
                stat["error"] = str(exc)
            stats.append(stat)
        return {"indices": stats}

    def batch_update_sidecars(
        self, config: str, file_paths: list[str],
        index: str = "source_code",
    ) -> list[dict[str, Any]]:
        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        sidecar_service = create_sidecar_service()
        results: list[dict[str, Any]] = []
        for fp in file_paths:
            try:
                source_path = self._resolve_source_path(index, index_config, fp)
                results.append(sidecar_service.update(source_path))
            except Exception as exc:
                results.append({"file_path": fp, "error": str(exc)})
        return results


# ------------------------------------------------------------------
# FastMCP server builder
# ------------------------------------------------------------------

def build_mcp_server(
    embedding_setup: str | Path | None = None,
) -> FastMCP:
    """Build the FastMCP instance and register SMAK tools."""

    emb_cfg = load_embedding_config(embedding_setup)
    smak = SmakMcpServer(embedding_config=emb_cfg)
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def refresh_knowledge(
        config: str,
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Ingest (or re-ingest) a source folder into a SMAK vector-store index.

        Args:
            config: Path to ``workspace_config.yaml``.
            index: Name of the index to refresh.
            follow_symlinks: Follow symbolic links.  Defaults to ``True``.
        """
        return smak.refresh_knowledge(config=config, index=index, follow_symlinks=follow_symlinks)

    @mcp.tool()
    def semantic_search(
        config: str, query: str,
        index: str = "source_code", top_k: int = 5,
    ) -> dict[str, Any]:
        """Search a SMAK index with a natural-language query.

        Args:
            config: Path to ``workspace_config.yaml``.
            query: Natural-language description of the intent or behavior.
            index: Name of the index to query.
            top_k: Maximum results.  Defaults to ``5``.
        """
        return smak.semantic_search(config=config, query=query, index=index, top_k=top_k)

    @mcp.tool()
    def list_available_indices(config: str) -> list[dict[str, str]]:
        """List searchable indices for a workspace.

        Args:
            config: Path to ``workspace_config.yaml``.
        """
        return smak.list_available_indices(config=config)

    @mcp.tool()
    def inspect_sidecar(
        config: str, file_path: str, index: str = "source_code",
    ) -> list[str]:
        """List the symbol UIDs parsed from a source file.

        Args:
            config: Path to ``workspace_config.yaml``.
            file_path: Path to the source file.
            index: Index whose root resolves relative paths.
        """
        return smak.inspect_sidecar(config=config, file_path=file_path, index=index)

    @mcp.tool()
    def update_sidecar(
        config: str, file_path: str, index: str = "source_code",
        symbol: str | None = None, intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sync or update sidecar metadata for a source file.

        Args:
            config: Path to ``workspace_config.yaml``.
            file_path: Path to the source file.
            index: Index whose root resolves relative paths.
            symbol: UID of a single symbol to update.
            intent: New intent description (only with ``symbol``).
            relations: New relation list (only with ``symbol``).
        """
        return smak.update_sidecar(
            config=config, file_path=file_path, index=index,
            symbol=symbol, intent=intent, relations=relations,
        )

    @mcp.tool()
    def clear_sidecar_symbol(
        config: str, file_path: str, symbol: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Remove a symbol entry from a sidecar file.

        Args:
            config: Path to ``workspace_config.yaml``.
            file_path: Path to the source file.
            symbol: Exact UID of the symbol to remove.
            index: Index whose root resolves relative paths.
        """
        return smak.clear_sidecar_symbol(
            config=config, file_path=file_path, symbol=symbol, index=index,
        )

    @mcp.tool()
    def lookup_symbol(
        config: str, uid: str, index: str = "source_code",
    ) -> dict[str, Any]:
        """Check whether a specific UID exists in the SMAK vector store.

        Args:
            config: Path to ``workspace_config.yaml``.
            uid: Full UID to look up.
            index: Index to query.
        """
        return smak.lookup_symbol(config=config, uid=uid, index=index)

    @mcp.tool()
    def validate_mesh(config: str) -> str:
        """Run integrity diagnostics across the SMAK mesh.

        Args:
            config: Path to ``workspace_config.yaml``.
        """
        return smak.validate_mesh(config=config)

    @mcp.tool()
    def multi_index_search(
        config: str, query: str,
        indices: list[str] | None = None, top_k: int = 3,
    ) -> dict[str, Any]:
        """Search across multiple (or all) indices at once.

        Args:
            config: Path to ``workspace_config.yaml``.
            query: Natural-language query.
            indices: Index names to search.  ``None`` = all.
            top_k: Max results per index.
        """
        return smak.multi_index_search(config=config, query=query, indices=indices, top_k=top_k)

    @mcp.tool()
    def workspace_status(config: str) -> dict[str, Any]:
        """Return health dashboard with per-index stats.

        Args:
            config: Path to ``workspace_config.yaml``.
        """
        return smak.workspace_status(config=config)

    @mcp.tool()
    def batch_update_sidecars(
        config: str, file_paths: list[str],
        index: str = "source_code",
    ) -> list[dict[str, Any]]:
        """Sync sidecars for multiple files at once.

        Args:
            config: Path to ``workspace_config.yaml``.
            file_paths: List of file paths to sync.
            index: Index whose root resolves relative paths.
        """
        return smak.batch_update_sidecars(config=config, file_paths=file_paths, index=index)

    return mcp


def main() -> None:
    """Run the SMAK MCP server over stdio transport."""

    import argparse

    parser = argparse.ArgumentParser(description="SMAK MCP Server")
    parser.add_argument(
        "--embedding-setup",
        type=str,
        default=_DEFAULT_EMBEDDING_SETUP,
        help="Path to embedding_setup.yaml",
    )
    args = parser.parse_args()

    server = build_mcp_server(embedding_setup=args.embedding_setup)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
