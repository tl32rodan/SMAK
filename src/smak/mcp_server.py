"""MCP bridge — intent-based tools for agent-driven SMAK operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from smak.config import EmbeddingConfig, SmakConfig, load_config, load_embedding_config
from smak.core_ops import (
    do_describe,
    do_enrich_batch,
    do_enrich_file,
    do_enrich_symbol,
    do_graph_stats,
    do_health,
    do_ingest,
    do_lookup,
    do_search,
    do_search_all,
    resolve_source_path,
)
from smak.factory import init_config

_DEFAULT_EMBEDDING_SETUP = str(Path(__file__).resolve().parent / "embedding_setup.yaml")


@dataclass
class SmakMcpServer:
    """Stateless MCP adapter.  Config is loaded per-call and cached."""

    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    _config_cache: dict[str, SmakConfig] = field(
        init=False, default_factory=dict, repr=False,
    )

    # ------------------------------------------------------------------
    # Internal helpers (caching layer only — logic lives in core_ops)
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

    # Kept for backward-compat with tests that call it directly
    @staticmethod
    def _resolve_source_path(
        index: str, index_config: object, file_path: str,
    ) -> Path:
        return resolve_source_path(index, index_config, file_path)

    # ==================================================================
    # Intent-based tools — all delegate to core_ops
    # ==================================================================

    def describe_workspace(self, config: str) -> dict[str, Any]:
        """Describe a SMAK workspace: list all indices with their names,
        descriptions, and paths.  Call this first to understand what's
        available before searching or enriching.

        Args:
            config: Path to ``workspace_config.yaml``.
        """
        cfg = self._load_config(config)
        result = do_describe(cfg)
        result["config_path"] = config
        return result

    def search(
        self, config: str, query: str,
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
        cfg = self._load_config(config)
        return do_search(cfg, query, index=index, top_k=top_k,
                         embedding_config=self.embedding_config)

    def search_all(
        self, config: str, query: str,
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
        cfg = self._load_config(config)
        return do_search_all(cfg, query, indices=indices, top_k=top_k,
                             embedding_config=self.embedding_config)

    def lookup(
        self, config: str, uid: str, index: str = "source_code",
    ) -> dict[str, Any]:
        """Check whether a UID exists in the vector store.  Use this to
        verify a relation target before adding it.

        Args:
            config: Path to ``workspace_config.yaml``.
            uid: Full UID (``path::symbol``).
            index: Index to query.
        """
        cfg = self._load_config(config)
        return do_lookup(cfg, uid, index=index,
                         embedding_config=self.embedding_config)

    def ingest(
        self, config: str, index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Re-ingest source files into a vector store index.
        **Resource-intensive** — only call when source files have changed.

        Args:
            config: Path to ``workspace_config.yaml``.
            index: Index to refresh.
            follow_symlinks: Follow symlinks during walk.
        """
        cfg = self._load_config(config)
        result = do_ingest(cfg, index=index, follow_symlinks=follow_symlinks,
                           embedding_config=self.embedding_config)
        return (
            "Ingestion Complete! "
            f"Processed Files: {result['files']}, "
            f"Skipped Files: {result['skipped']}, Vectors Added: {result['vectors']}"
        )

    def enrich_symbol(
        self, config: str, file_path: str, symbol: str,
        intent: str | None = None, relations: list[str] | None = None,
        index: str = "source_code",
        bidirectional: bool = False,
    ) -> dict[str, Any]:
        """Annotate a single code symbol with intent and/or relations.

        This is the primary sidecar editing tool.  It automatically:
        - Validates the symbol exists in the file
        - Creates/syncs the sidecar if needed
        - Clears stale symbols that block sync
        - Writes the intent and relations

        When ``bidirectional=True`` and relations are provided, also adds
        a reverse relation from each target back to this symbol.  The
        reverse relation target symbol defaults to ``*`` (file-level).

        Args:
            config: Path to ``workspace_config.yaml``.
            file_path: Source file path (relative OK — resolved automatically).
            symbol: Short symbol name (e.g. ``CsvEditor.update_cell``).
            intent: Human description of the symbol's purpose.
            relations: List of full UIDs to link as relations.
            index: Index whose paths resolve the file.
            bidirectional: If True, add reverse relations from targets back to this symbol.
        """
        cfg = self._load_config(config)
        return do_enrich_symbol(
            cfg, file_path, symbol,
            intent=intent, relations=relations,
            index=index, bidirectional=bidirectional,
        )

    def enrich_file(
        self, config: str, file_path: str, index: str = "source_code",
    ) -> dict[str, Any]:
        """Initialize or sync a file's sidecar.  Creates stub entries for
        every symbol found in the source.

        Args:
            config: Path to ``workspace_config.yaml``.
            file_path: Source file path.
            index: Index whose paths resolve the file.
        """
        cfg = self._load_config(config)
        return do_enrich_file(cfg, file_path, index=index)

    def enrich_batch(
        self, config: str, file_paths: list[str],
        index: str = "source_code",
    ) -> list[dict[str, Any]]:
        """Sync sidecars for multiple files at once.  Continues on error.

        Args:
            config: Path to ``workspace_config.yaml``.
            file_paths: List of source file paths.
            index: Index whose paths resolve the files.
        """
        cfg = self._load_config(config)
        return do_enrich_batch(cfg, file_paths, index=index)

    def check_health(self, config: str) -> dict[str, Any]:
        """Run mesh integrity diagnostics.  Returns structured report
        with status (``"healthy"`` / ``"unhealthy"``) and issues list.

        Args:
            config: Path to ``workspace_config.yaml``.
        """
        cfg = self._load_config(config)
        return do_health(cfg)

    def graph_stats(self, config: str) -> dict[str, Any]:
        """Return knowledge graph coverage statistics.

        Computes per-index and overall metrics: total symbols,
        enriched symbols (with intent), total relations, coverage
        percentage, and asymmetric relation warnings.

        Use this to understand the maturity of the knowledge graph
        and identify indices that need more enrichment.

        Args:
            config: Path to ``workspace_config.yaml``.
        """
        cfg = self._load_config(config)
        return do_graph_stats(cfg)


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

    for method in (
        smak.describe_workspace,
        smak.search,
        smak.search_all,
        smak.lookup,
        smak.ingest,
        smak.enrich_symbol,
        smak.enrich_file,
        smak.enrich_batch,
        smak.check_health,
        smak.graph_stats,
    ):
        mcp.tool()(method)

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
