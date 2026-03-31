"""MCP bridge — intent-based tools for agent-driven SMAK operations."""

from __future__ import annotations

import re
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

# Regex to extract symbol names from the "Cannot remove symbols" error.
_STALE_SYMBOL_RE = re.compile(r'--symbol "([^"]+)"')


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
        index_roots = [Path(p).resolve() for p in index_config.paths if Path(p).is_dir()]
        index_files = [Path(p).resolve() for p in index_config.paths if Path(p).is_file()]
        raw = Path(file_path)

        if raw.is_absolute() and raw.exists():
            return raw
        if not raw.is_absolute():
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
                f"Ambiguous file path '{file_path}'. "
                f"Did you mean {self._format_candidates(rel)}?"
            )

        roots_display = ", ".join(f"'{r}'" for r in index_roots)
        raise FileNotFoundError(
            f"File path resolution failed under index roots [{roots_display}] "
            f"(index='{index}', file_path='{file_path}'). "
            "Try using semantic_search first and pass one of the returned file paths."
        )

    # ==================================================================
    # Intent-based tools
    # ==================================================================

    # ---- describe_workspace -----------------------------------------

    def describe_workspace(self, config: str) -> dict[str, Any]:
        """Return workspace metadata so the agent knows what's available."""
        cfg = self._load_config(config)
        return {
            "config_path": config,
            "indices": [
                {
                    "name": idx.name,
                    "description": idx.description,
                    "paths": idx.paths,
                    "path_env": idx.path_env,
                }
                for idx in cfg.indices
            ],
        }

    # ---- search -----------------------------------------------------

    def search(
        self, config: str, query: str,
        index: str = "source_code", top_k: int = 5,
    ) -> dict[str, Any]:
        """Semantic search within a single index."""
        cfg = self._load_config(config)
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        service = create_query_service(
            vector_store, cfg, index_config,
            embedding_config=self.embedding_config,
        )
        result = service.search(query, top_k=top_k)
        return result if isinstance(result, dict) else {}

    # ---- search_all -------------------------------------------------

    def search_all(
        self, config: str, query: str,
        indices: list[str] | None = None, top_k: int = 3,
    ) -> dict[str, Any]:
        """Search across multiple (or all) indices at once."""
        cfg = self._load_config(config)
        target = indices or [idx.name for idx in cfg.indices]
        results: dict[str, Any] = {}
        for name in target:
            ic = cfg.get_index(name)
            if ic is None:
                results[name] = {"error": f"Index '{name}' not found"}
                continue
            try:
                vs = load_and_validate_vector_store(ic, cfg)
                svc = create_query_service(
                    vs, cfg, ic, embedding_config=self.embedding_config,
                )
                results[name] = svc.search(query, top_k=top_k)
            except Exception as exc:
                results[name] = {"error": str(exc)}
        return results

    # ---- lookup -----------------------------------------------------

    def lookup(
        self, config: str, uid: str, index: str = "source_code",
    ) -> dict[str, Any]:
        """Check whether a UID exists in the vector store."""
        cfg = self._load_config(config)
        vs, ic = self._load_index_vector_store(cfg, index)
        svc = create_query_service(
            vs, cfg, ic, embedding_config=self.embedding_config,
        )
        return svc.lookup(uid)

    # ---- ingest -----------------------------------------------------

    def ingest(
        self, config: str, index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Re-ingest files into a vector store index."""
        cfg = self._load_config(config)
        vs, ic = self._load_index_vector_store(cfg, index)
        target_paths = [Path(p) for p in ic.paths]
        service = IngestService(vector_store=vs)
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

    # ---- enrich_symbol ----------------------------------------------

    def enrich_symbol(
        self, config: str, file_path: str, symbol: str,
        intent: str | None = None, relations: list[str] | None = None,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Annotate a single symbol with intent and/or relations.

        Automatically:
        - Validates the symbol exists in the source file
        - Ensures the sidecar is synced (auto-creates if missing)
        - Clears stale symbols if they block a full sync
        """
        cfg = self._load_config(config)
        ic = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(index, ic, file_path)
        svc = create_sidecar_service()

        # 1. Validate symbol exists
        valid_symbols = svc.inspect(source_path)
        if symbol not in valid_symbols:
            return {
                "status": "error",
                "message": (
                    f"Symbol '{symbol}' not found in {file_path}. "
                    f"Valid symbols: {valid_symbols}"
                ),
                "valid_symbols": valid_symbols,
            }

        # 2. Ensure sidecar is synced (full sync first)
        try:
            svc.update(source_path)
        except ValueError as exc:
            # Stale symbols blocking removal — auto-clear them
            stale = _STALE_SYMBOL_RE.findall(str(exc))
            if not stale:
                return {"status": "error", "message": str(exc)}
            for s in stale:
                try:
                    svc.clear_symbol(source_path, s)
                except ValueError:
                    pass
            svc.update(source_path)

        # 3. Apply the enrichment
        if intent is None and relations is None:
            return {
                "status": "ok",
                "message": f"Sidecar synced for {file_path}. "
                f"No intent/relations provided for '{symbol}'.",
            }
        svc.update(
            source_path, symbol=symbol, intent=intent, relations=relations,
        )
        return {
            "status": "ok",
            "file_path": str(source_path),
            "symbol": symbol,
            "intent": intent,
            "relations": relations,
        }

    # ---- enrich_file ------------------------------------------------

    def enrich_file(
        self, config: str, file_path: str, index: str = "source_code",
    ) -> dict[str, Any]:
        """Initialize or sync a file's sidecar (full sync)."""
        cfg = self._load_config(config)
        ic = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(index, ic, file_path)
        svc = create_sidecar_service()
        return svc.update(source_path)

    # ---- enrich_batch -----------------------------------------------

    def enrich_batch(
        self, config: str, file_paths: list[str],
        index: str = "source_code",
    ) -> list[dict[str, Any]]:
        """Sync sidecars for multiple files at once."""
        cfg = self._load_config(config)
        ic = self._get_index_config(cfg, index)
        svc = create_sidecar_service()
        results: list[dict[str, Any]] = []
        for fp in file_paths:
            try:
                source = self._resolve_source_path(index, ic, fp)
                results.append(svc.update(source))
            except Exception as exc:
                results.append({"file_path": fp, "error": str(exc)})
        return results

    # ---- check_health -----------------------------------------------

    def check_health(self, config: str) -> dict[str, Any]:
        """Run diagnostics and return structured health report."""
        cfg = self._load_config(config)
        service = create_doctor_service(cfg)
        try:
            service.validate_all()
        except RuntimeError as exc:
            return {
                "status": "unhealthy",
                "issues": str(exc).split("\n"),
            }
        return {"status": "healthy", "issues": []}


# ------------------------------------------------------------------
# FastMCP server builder
# ------------------------------------------------------------------

def build_mcp_server(
    embedding_setup: str | Path | None = None,
) -> FastMCP:
    """Build the FastMCP instance with intent-based SMAK tools."""

    emb_cfg = load_embedding_config(embedding_setup)
    smak = SmakMcpServer(embedding_config=emb_cfg)
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def describe_workspace(config: str) -> dict[str, Any]:
        """Describe a SMAK workspace: list all indices with their names,
        descriptions, and paths.  Call this first to understand what's
        available before searching or enriching.

        Args:
            config: Path to ``workspace_config.yaml``.
        """
        return smak.describe_workspace(config=config)

    @mcp.tool()
    def search(
        config: str, query: str,
        index: str = "source_code", top_k: int = 5,
    ) -> dict[str, Any]:
        """Semantic search within a single index.  Returns hits with
        1-hop related context from sidecar relations.

        Write queries as natural-language descriptions of **intent or
        behavior** — not symbol names or file paths.

        Args:
            config: Path to ``workspace_config.yaml``.
            query: Natural-language query.
            index: Target index name.
            top_k: Max results.
        """
        return smak.search(config=config, query=query, index=index, top_k=top_k)

    @mcp.tool()
    def search_all(
        config: str, query: str,
        indices: list[str] | None = None, top_k: int = 3,
    ) -> dict[str, Any]:
        """Search across multiple (or all) indices at once.  Use when
        you don't know which index contains the answer.

        Args:
            config: Path to ``workspace_config.yaml``.
            query: Natural-language query.
            indices: Index names to search (``None`` = all).
            top_k: Max results per index.
        """
        return smak.search_all(config=config, query=query, indices=indices, top_k=top_k)

    @mcp.tool()
    def lookup(
        config: str, uid: str, index: str = "source_code",
    ) -> dict[str, Any]:
        """Check whether a UID exists in the vector store.  Use this to
        verify a relation target before adding it.

        Args:
            config: Path to ``workspace_config.yaml``.
            uid: Full UID (``path::symbol``).
            index: Index to query.
        """
        return smak.lookup(config=config, uid=uid, index=index)

    @mcp.tool()
    def ingest(
        config: str, index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Re-ingest source files into a vector store index.
        **Resource-intensive** — only call when source files have changed.

        Args:
            config: Path to ``workspace_config.yaml``.
            index: Index to refresh.
            follow_symlinks: Follow symlinks during walk.
        """
        return smak.ingest(config=config, index=index, follow_symlinks=follow_symlinks)

    @mcp.tool()
    def enrich_symbol(
        config: str, file_path: str, symbol: str,
        intent: str | None = None, relations: list[str] | None = None,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Annotate a single code symbol with intent and/or relations.

        This is the primary sidecar editing tool.  It automatically:
        - Validates the symbol exists in the file
        - Creates/syncs the sidecar if needed
        - Clears stale symbols that block sync
        - Writes the intent and relations

        Args:
            config: Path to ``workspace_config.yaml``.
            file_path: Source file path (relative OK — resolved automatically).
            symbol: Short symbol name (e.g. ``CsvEditor.update_cell``).
            intent: Human description of the symbol's purpose.
            relations: List of full UIDs to link as relations.
            index: Index whose paths resolve the file.
        """
        return smak.enrich_symbol(
            config=config, file_path=file_path, symbol=symbol,
            intent=intent, relations=relations, index=index,
        )

    @mcp.tool()
    def enrich_file(
        config: str, file_path: str, index: str = "source_code",
    ) -> dict[str, Any]:
        """Initialize or sync a file's sidecar.  Creates stub entries for
        every symbol found in the source.

        Args:
            config: Path to ``workspace_config.yaml``.
            file_path: Source file path.
            index: Index whose paths resolve the file.
        """
        return smak.enrich_file(config=config, file_path=file_path, index=index)

    @mcp.tool()
    def enrich_batch(
        config: str, file_paths: list[str], index: str = "source_code",
    ) -> list[dict[str, Any]]:
        """Sync sidecars for multiple files at once.  Continues on error.

        Args:
            config: Path to ``workspace_config.yaml``.
            file_paths: List of source file paths.
            index: Index whose paths resolve the files.
        """
        return smak.enrich_batch(config=config, file_paths=file_paths, index=index)

    @mcp.tool()
    def check_health(config: str) -> dict[str, Any]:
        """Run mesh integrity diagnostics.  Returns structured report
        with status (``"healthy"`` / ``"unhealthy"``) and issues list.

        Args:
            config: Path to ``workspace_config.yaml``.
        """
        return smak.check_health(config=config)

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
