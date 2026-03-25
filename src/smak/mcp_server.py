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
from smak.utils.yaml import safe_load

_DEFAULT_EMBEDDING_SETUP = str(Path(__file__).resolve().parent / "embedding_setup.yaml")


@dataclass
class SmakMcpServer:
    """In-process adapter used by MCP tool handlers."""

    registry_path: Path
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
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
        """Resolve the absolute path to a config file from a registry entry.

        Args:
            config_name: Key name as declared in the registry YAML.

        Returns:
            Absolute ``Path`` to the config file.

        Raises:
            ValueError: If *config_name* is not present in the registry.
        """

        if config_name not in self.configs:
            raise ValueError(f"Config '{config_name}' not found in registry.")

        base_dir = self.registry_path.parent
        target_path = Path(self.configs[config_name]["config_path"])
        if not target_path.is_absolute():
            target_path = (base_dir / target_path).resolve()

        return target_path

    def _load_config(self, config_name: str) -> SmakConfig:
        """Load and return the resolved SmakConfig for a registry entry.

        The embedding dimensions are initialised as a side-effect so the
        returned config is ready for immediate use.

        Args:
            config_name: Key name as declared in the registry YAML.

        Returns:
            Fully-initialised :class:`~smak.config.SmakConfig`.

        Raises:
            FileNotFoundError: If the resolved config file does not exist.
        """

        config_path = self._resolve_config_path(config_name)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        config = load_config(config_path)
        return init_config(config, embedding_config=self.embedding_config)

    @staticmethod
    def _get_index_config(config: SmakConfig, index: str) -> object:
        """Return the index-level configuration block for *index*.

        Args:
            config: Top-level SMAK configuration.
            index: Name of the index (e.g. ``"source_code"``).

        Returns:
            The index configuration object stored inside *config*.

        Raises:
            ValueError: If *index* is not found in *config*.
        """

        index_config = config.get_index(index)
        if index_config is None:
            raise ValueError(f"Index '{index}' not found in configuration.")
        return index_config

    def _load_index_vector_store(
        self,
        config: SmakConfig,
        index: str,
    ) -> tuple[object, object]:
        """Load and validate the vector store for a given index.

        Args:
            config: Top-level SMAK configuration.
            index: Name of the index whose vector store should be loaded.

        Returns:
            A ``(vector_store, index_config)`` tuple ready for use by services.

        Raises:
            ValueError: If *index* is not found in *config* or the vector
                store dimension does not match the configured embedding size.
        """

        index_config = self._get_index_config(config, index)
        vector_store = load_and_validate_vector_store(index_config, config)
        return vector_store, index_config

    @staticmethod
    def _format_candidates(candidates: list[str]) -> str:
        """Format candidate path hints for human-readable error messages."""

        if len(candidates) == 1:
            return f"'{candidates[0]}'"
        if len(candidates) == 2:
            return f"'{candidates[0]}' or '{candidates[1]}'"
        quoted = [f"'{candidate}'" for candidate in candidates]
        return ", ".join(quoted[:-1]) + f", or {quoted[-1]}"

    def _resolve_source_path(
        self,
        config_name: str,
        index: str,
        index_config: object,
        file_path: str,
    ) -> Path:
        """Resolve *file_path* with defensive fallback across all index roots.

        Resolution order:

        1. Try the provided path directly (absolute) or relative to each index root.
        2. If missing, scan all index roots with ``Path.rglob(f"*{name}")``.
        3. Use unique hit silently, reject ambiguous hits, and fail actionable
           when no candidates exist.

        Args:
            index_config: Index configuration object that exposes a ``paths``
                attribute listing the monitored folders.
            file_path: A file path string. Absolute paths are strongly
                recommended to reduce agent ambiguity; relative paths are
                still supported and resolved against each index root.

        Returns:
            Resolved absolute :class:`~pathlib.Path`.

        Raises:
            ValueError: If fallback scanning finds multiple candidates.
            FileNotFoundError: If no match can be found.
        """

        index_roots = [Path(p).resolve() for p in index_config.paths if Path(p).is_dir()]
        index_files = [Path(p).resolve() for p in index_config.paths if Path(p).is_file()]
        raw_source_path = Path(file_path)

        # Try absolute path directly, or relative to each index root
        if raw_source_path.is_absolute():
            if raw_source_path.exists():
                return raw_source_path
        else:
            for index_root in index_roots:
                primary_path = (index_root / raw_source_path).resolve()
                if primary_path.exists():
                    return primary_path

        # Fallback: scan all roots for filename match
        file_name = raw_source_path.name
        all_candidates: list[Path] = []
        # Include individually-listed files whose name matches
        for idx_file in index_files:
            if idx_file.name == file_name:
                all_candidates.append(idx_file)
        for index_root in index_roots:
            all_candidates.extend(index_root.rglob(f"*{file_name}") if file_name else [])
        candidates = sorted(all_candidates)

        if len(candidates) == 1:
            return candidates[0].resolve()

        if len(candidates) > 1:
            # Build relative paths from whichever root contains each candidate
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
            f"(config='{config_name}', index='{index}', file_path='{file_path}'). "
            "Try using semantic_search first and pass one of the returned file paths."
        )

    # ------------------------------------------------------------------
    # Public service methods
    # ------------------------------------------------------------------

    def refresh_knowledge(
        self,
        config: str,
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Ingest content into a target index (folders are defined by the index config).

        Walks all folders associated with *index* and upserts all discovered
        files into the vector store.  Existing vectors for unchanged files are
        skipped automatically.

        Args:
            config: Registry key that identifies the project configuration.
            index: Name of the index to refresh.  Defaults to
                ``"source_code"``.
            follow_symlinks: Whether to follow symbolic links while walking
                the target folders.  Defaults to ``True``.

        Returns:
            A human-readable summary string with processed / skipped / added
            file and vector counts.
        """

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
        self,
        config: str,
        query: str,
        index: str = "source_code",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run in-process semantic query and return serializable payload.

        Embeds *query* using the configured embedding model, retrieves the
        *top_k* nearest vectors from the index, and returns enriched results
        that include sidecar metadata when available.

        Args:
            config: Registry key that identifies the project configuration.
            query: Free-text search query.
            index: Name of the index to query. Always pass this explicitly so
                search runs against the intended index for the selected
                config. The default ``"source_code"`` is preserved only for
                backwards compatibility.
            top_k: Maximum number of results to return.  Defaults to ``5``.

        Returns:
            A serialisable :class:`dict` containing the search results, or an
            empty dict if no results were found.
        """

        cfg = self._load_config(config)
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        service = create_query_service(
            vector_store, cfg, index_config, embedding_config=self.embedding_config,
        )
        result = service.search(query, top_k=top_k)
        return result if isinstance(result, dict) else {}

    def list_available_indices(self, config: str) -> list[dict[str, str]]:
        """Return all index names/descriptions for a config.

        Args:
            config: Registry key that identifies the project configuration.

        Returns:
            Ordered list of ``{"name": str, "description": str}`` objects.
        """

        cfg = self._load_config(config)
        return [
            {
                "name": index.name,
                "description": index.description,
            }
            for index in cfg.indices
        ]

    def inspect_sidecar(
        self,
        config: str,
        file_path: str,
        index: str = "source_code",
    ) -> list[str]:
        """Return the list of symbol UIDs parsed from a source file.

        Parses *file_path* with the appropriate language parser and returns
        the UID of every code unit found.  No sidecar file is read or written;
        this is a read-only introspection operation.

        Args:
            config: Registry key that identifies the project configuration.
            file_path: Path to the source file. Prefer an absolute path to
                minimize agent cognitive overhead; relative paths are still
                accepted and resolved from the index root.
            index: Name of the index whose root is used to resolve relative
                paths.  Defaults to ``"source_code"``.

        Returns:
            Ordered list of symbol UID strings as produced by the parser.
        """

        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(config, index, index_config, file_path)
        return create_sidecar_service().inspect(source_path)

    def update_sidecar(
        self,
        config: str,
        file_path: str,
        index: str = "source_code",
        symbol: str | None = None,
        intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sync or update sidecar metadata for a source file.

        **Full sync** (no ``symbol``): parses the source file and creates or
        updates the sidecar so that it contains exactly the symbols currently
        in the source.  Existing ``intent`` and ``relations`` are preserved.
        If a symbol would be removed but still has relations, the operation
        fails and lists the ``clear_sidecar_symbol`` calls needed first.

        **Single-symbol update** (``symbol`` provided): updates only the
        specified symbol's ``intent`` and/or ``relations``.

        Args:
            config: Registry key that identifies the project configuration.
            file_path: Path to the source file whose sidecar should be
                updated.  Absolute paths are strongly recommended.
            index: Index whose root resolves relative paths.  Defaults to
                ``"source_code"``.
            symbol: UID of a single symbol to update.  When omitted the full
                sync mode is used instead.
            intent: New intent description (only with ``symbol``).
            relations: New relation list (only with ``symbol``).

        Returns:
            A dict describing the result of the operation.
        """

        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(config, index, index_config, file_path)
        return create_sidecar_service().update(
            source_path, symbol=symbol, intent=intent, relations=relations
        )

    def clear_sidecar_symbol(
        self,
        config: str,
        file_path: str,
        symbol: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Remove a single symbol entry from a sidecar file.

        Use this to clear a symbol that has relations before running a full
        ``update_sidecar`` sync, which would otherwise refuse to delete it.

        Args:
            config: Registry key that identifies the project configuration.
            file_path: Path to the source file whose sidecar should be
                modified.  Absolute paths are strongly recommended.
            symbol: UID of the symbol to remove.
            index: Index whose root resolves relative paths.  Defaults to
                ``"source_code"``.

        Returns:
            A dict with ``file_path``, ``sidecar_path``, ``cleared_symbol``,
            and ``remaining_symbols``.
        """

        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(config, index, index_config, file_path)
        return create_sidecar_service().clear_symbol(source_path, symbol)

    def lookup_symbol(
        self,
        config: str,
        uid: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Check whether a UID exists in the vector store.

        Args:
            config: Registry key that identifies the project configuration.
            uid: The full UID to look up (format: ``{absolute_path}::{symbol}``).
            index: Name of the index to query.  Defaults to ``"source_code"``.

        Returns:
            A dict with ``found`` (bool), ``uid``, and — when found —
            ``content`` and ``metadata``.
        """

        cfg = self._load_config(config)
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        service = create_query_service(
            vector_store, cfg, index_config, embedding_config=self.embedding_config,
        )
        return service.lookup(uid)

    def validate_mesh(self, config: str) -> str:
        """Run mesh/sidecar integrity checks in-process.

        Loads every configured index and validates consistency between the
        vector store, the source files, and the sidecar metadata.  Raises on
        the first detected inconsistency.

        Args:
            config: Registry key that identifies the project configuration.

        Returns:
            A confirmation string if all diagnostics pass.

        Raises:
            Exception: Any exception raised by
                :class:`~smak.services.DoctorService` when a problem is found.
        """

        cfg = self._load_config(config)
        service = create_doctor_service(cfg)
        service.validate_all()
        return "Mesh diagnostics passed."


def build_mcp_server(
    registry_path: str | Path,
    embedding_setup: str | Path | None = None,
) -> FastMCP:
    """Build the FastMCP instance and register SMAK tools."""

    emb_cfg = load_embedding_config(embedding_setup)
    smak_server = SmakMcpServer(
        registry_path=Path(registry_path).resolve(),
        embedding_config=emb_cfg,
    )
    mcp = FastMCP("SMAK")

    @mcp.tool()
    def list_available_configs() -> dict[str, Any]:
        """Return all project configurations registered in the registry file.

        Use this tool to discover which *config* values are valid before
        calling any other SMAK tool.

        Returns:
            A dict mapping each config name to its registry metadata (e.g.
            ``config_path``).
        """

        return smak_server.configs

    @mcp.tool()
    def refresh_knowledge(
        config: str,
        index: str = "source_code",
        follow_symlinks: bool = True,
    ) -> str:
        """Ingest (or re-ingest) a source folder into a SMAK vector-store index.

        Walks the folder bound to *index* in the project config and upserts
        every file into the vector store.  Call this tool after adding,
        modifying, or deleting source files so that ``semantic_search``
        returns up-to-date results.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).
            index: Name of the index to refresh.  Defaults to
                ``"source_code"``.
            follow_symlinks: Whether to descend into symbolic links during
                folder traversal.  Defaults to ``True``.

        Returns:
            A human-readable summary such as
            ``"Ingestion Complete! Processed Files: 42, Skipped Files: 3,
            Vectors Added: 210"``.
        """

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
        """Search a SMAK index with a natural-language query.

        Embeds *query* and retrieves the *top_k* most similar code units from
        the vector store.  Results are enriched with sidecar metadata
        (``intent``, ``relations``) when available.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).
            query: Natural-language description of the **intent or behavior**
                you are looking for — not symbol names or file paths.

                Good queries (intent-based):
                  - ``"error handling for out-of-range index in cell update"``
                  - ``"how log entries are parsed from structured log files"``
                  - ``"retry logic for network failures"``

                Bad queries (use grep/ripgrep instead):
                  - ``"append_row"``  ← exact function name
                  - ``"csv_editor.py"``  ← exact filename


            index: **Required in practice**. Always pass the explicit target
                index name (discover with ``list_available_indices``).

                Index selection guide:
                  - ``"source_code"`` — code logic, implementation intent
                  - ``"issues"`` — historical bugs, known problems, tickets
                  - ``"tests"`` — test coverage, test cases
                  - ``"documentation"`` — architecture docs, API docs

                The default ``"source_code"`` is kept only for backward
                compatibility and can point to the wrong index for multi-index
                configs.
            top_k: Maximum number of results to return.  Defaults to ``5``.

        Returns:
            A serialisable dict ``{"hits": [...], "related_context": [...]}``
            or ``{}`` when the index is empty.

            Each hit contains: ``uid``, ``exact_relative_path``, ``score``
            (backend-dependent scale/direction — do not threshold on this
            value), ``match_type`` (``"semantic"``), and ``content``.

            Each related_context entry contains: ``uid``, ``match_type``
            (``"relation"``), ``source_hit`` (uid of the hit that links here),
            and ``content``.

            **Important:** copy ``hits[].exact_relative_path`` verbatim as the
            ``file_path`` argument to sidecar tools — never rewrite it.
        """

        return smak_server.semantic_search(
            config=config,
            query=query,
            index=index,
            top_k=top_k,
        )

    @mcp.tool()
    def list_available_indices(config: str) -> list[dict[str, str]]:
        """List searchable indices for a specific config.

        Use this tool before ``semantic_search`` whenever you are unsure
        which indices are defined for the current project config.

        Index purpose guide (typical multi-index project layout):
          - ``"source_code"`` — code logic, function/class implementations
          - ``"issues"`` — bug reports, GitHub issues, Jira tickets
          - ``"tests"`` — unit/integration tests and test cases
          - ``"documentation"`` — architecture docs, API references

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).

        Returns:
            Ordered list of objects containing ``name`` and ``description``.
        """

        return smak_server.list_available_indices(config=config)

    @mcp.tool()
    def inspect_sidecar(
        config: str,
        file_path: str,
        index: str = "source_code",
    ) -> list[str]:
        """List the symbol UIDs that SMAK can parse from a source file.

        Runs the language-specific parser over *file_path* and returns the UID
        of every discovered code unit.  No files are written.

        **Always call this before ``update_sidecar``** to confirm which symbol
        UIDs are valid for the target file.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).
            file_path: Path to the source file. Use the ``exact_relative_path``
                value returned by ``semantic_search`` to avoid path errors.
                Absolute paths are strongly recommended; relative paths are
                resolved against the index root folder.
            index: Index whose root resolves relative paths.  Defaults to
                ``"source_code"``.

        Returns:
            Ordered list of symbol UID strings.

            Example: ``["CsvEditor", "CsvEditor.append_row",
            "CsvEditor.update_cell", "CsvEditor.read_rows"]``

            Use these exact strings as the ``symbol`` argument in
            ``update_sidecar`` calls.
        """

        return smak_server.inspect_sidecar(
            config=config,
            file_path=file_path,
            index=index,
        )

    @mcp.tool()
    def update_sidecar(
        config: str,
        file_path: str,
        index: str = "source_code",
        symbol: str | None = None,
        intent: str | None = None,
        relations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Sync or update sidecar metadata for a source file.

        **Full sync mode** (omit ``symbol``): parses the source file and
        creates or updates the sidecar so it contains exactly the symbols
        currently present in the source.  Existing ``intent`` and
        ``relations`` values are preserved for symbols that still exist.
        New symbols get empty stubs.  If a symbol would be removed but still
        has relations, the call fails with an error listing the
        ``clear_sidecar_symbol`` calls needed first.

        **Single-symbol mode** (provide ``symbol``): updates only the named
        symbol's ``intent`` and/or ``relations``.  At least one of
        ``intent`` or ``relations`` must be supplied.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).
            file_path: Path to the source file whose sidecar should be
                updated.  Absolute paths are strongly recommended; relative
                paths are resolved against the index root folder.
            index: Index whose root resolves relative paths.  Defaults to
                ``"source_code"``.
            symbol: UID of a single symbol to update.  When omitted the full
                sync mode is used.
            intent: New intent description (only used with ``symbol``).
            relations: New relation list (only used with ``symbol``).

        Returns:
            A dict describing the result of the operation.
        """

        return smak_server.update_sidecar(
            config=config,
            file_path=file_path,
            index=index,
            symbol=symbol,
            intent=intent,
            relations=relations,
        )

    @mcp.tool()
    def clear_sidecar_symbol(
        config: str,
        file_path: str,
        symbol: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Remove a symbol entry from a sidecar file.

        Deletes the named symbol from the sidecar regardless of whether it
        has relations.  Use this before ``update_sidecar`` (full sync) when
        the sync would otherwise refuse to remove a symbol that still
        carries relation metadata.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).
            file_path: Path to the source file whose sidecar should be
                modified.  Absolute paths are strongly recommended.
            symbol: Exact UID of the symbol to remove from the sidecar.
            index: Index whose root resolves relative paths.  Defaults to
                ``"source_code"``.

        Returns:
            A dict with ``file_path``, ``sidecar_path``,
            ``cleared_symbol``, and ``remaining_symbols``.
        """

        return smak_server.clear_sidecar_symbol(
            config=config,
            file_path=file_path,
            symbol=symbol,
            index=index,
        )

    @mcp.tool()
    def lookup_symbol(
        config: str,
        uid: str,
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Check whether a specific UID exists in the SMAK vector store.

        Use this to verify that a symbol has been ingested before relying on
        it in ``semantic_search`` or sidecar ``relations``.

        The UID format is ``{absolute_path}::{symbol_name}`` — copy the
        ``uid`` value from a ``semantic_search`` hit or construct it from
        ``inspect_sidecar`` output.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).
            uid: Full UID to look up
                (e.g. ``"/home/user/project/src/foo.py::ClassName"``).
            index: Index to query.  Defaults to ``"source_code"``.

        Returns:
            A dict with ``found`` (bool) and ``uid``.  When found, also
            includes ``content`` and ``metadata``.
        """

        return smak_server.lookup_symbol(
            config=config,
            uid=uid,
            index=index,
        )

    @mcp.tool()
    def validate_mesh(config: str) -> str:
        """Run integrity diagnostics across the SMAK mesh for a project.

        Checks that every source file tracked in the vector store still exists
        on disk, that sidecar symbol references resolve correctly, and that
        index dimensions are consistent.  Raises on the first problem found.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).

        Returns:
            ``"Mesh diagnostics passed."`` if no issues are detected.
        """

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
    parser.add_argument(
        "--embedding-setup",
        type=str,
        default=_DEFAULT_EMBEDDING_SETUP,
        help="Path to embedding_setup.yaml",
    )
    args = parser.parse_args()

    server = build_mcp_server(
        registry_path=Path(args.registry).resolve(),
        embedding_setup=args.embedding_setup,
    )
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
