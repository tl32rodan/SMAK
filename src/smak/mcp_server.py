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
from smak.sidecar.store import YAMLSidecarStore
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
        config = initialize_embedding_dimensions(config, InternalNomicEmbedding())
        return config

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
        vector_store = _load_vector_store(index_config, config)
        validate_vector_store_dimension(vector_store, config.embedding_dimensions)
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
        """Resolve *file_path* with defensive fallback inside the index root.

        Resolution order:

        1. Try the provided path directly (absolute) or relative to index root.
        2. If missing, scan index root with ``Path.rglob(f"*{name}")``.
        3. Use unique hit silently, reject ambiguous hits, and fail actionable
           when no candidates exist.

        Args:
            index_config: Index configuration object that exposes a ``path``
                attribute pointing to the monitored folder.
            file_path: A file path string. Absolute paths are strongly
                recommended to reduce agent ambiguity; relative paths are
                still supported and resolved against the index root.

        Returns:
            Resolved absolute :class:`~pathlib.Path`.

        Raises:
            ValueError: If fallback scanning finds multiple candidates.
            FileNotFoundError: If no match can be found.
        """

        index_root = Path(index_config.path).resolve()
        raw_source_path = Path(file_path)
        primary_path = (
            raw_source_path
            if raw_source_path.is_absolute()
            else (index_root / raw_source_path).resolve()
        )
        if primary_path.exists():
            return primary_path

        file_name = raw_source_path.name
        candidates = sorted(index_root.rglob(f"*{file_name}")) if file_name else []
        if len(candidates) == 1:
            return candidates[0].resolve()

        relative_candidates = [str(path.resolve().relative_to(index_root)) for path in candidates]
        if len(relative_candidates) > 1:
            hints = self._format_candidates(relative_candidates)
            raise ValueError(f"Ambiguous file path '{file_path}'. Did you mean {hints}?")

        raise FileNotFoundError(
            "File path resolution failed under index root "
            f"'{index_root}' (config='{config_name}', index='{index}', file_path='{file_path}'). "
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
        """Ingest content into a target index (folder is defined by the index config).

        Walks the folder associated with *index* and upserts all discovered
        files into the vector store.  Existing vectors for unchanged files are
        skipped automatically.

        Args:
            config: Registry key that identifies the project configuration.
            index: Name of the index to refresh.  Defaults to
                ``"source_code"``.
            follow_symlinks: Whether to follow symbolic links while walking
                the target folder.  Defaults to ``True``.

        Returns:
            A human-readable summary string with processed / skipped / added
            file and vector counts.
        """

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
        """Run in-process semantic query and return serializable payload.

        Embeds *query* using the configured embedding model, retrieves the
        *top_k* nearest vectors from the index, and returns enriched results
        that include sidecar metadata when available.

        Args:
            config: Registry key that identifies the project configuration.
            query: Free-text search query.
            index: Name of the index to query.  Defaults to ``"source_code"``.
            top_k: Maximum number of results to return.  Defaults to ``5``.

        Returns:
            A serialisable :class:`dict` containing the search results, or an
            empty dict if no results were found.
        """

        cfg = self._load_config(config)
        vector_store, index_config = self._load_index_vector_store(cfg, index)
        sidecar_store = YAMLSidecarStore()
        service = QueryService(
            vector_store=vector_store,
            config=cfg,
            index_config=index_config,
            vector_store_loader=_load_vector_store,
            relation_resolver=SidecarRelationResolver(sidecar_store),
        )
        result = service.search(query, top_k=top_k)
        return result if isinstance(result, dict) else {}

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
        service = SidecarService(sidecar_store=YAMLSidecarStore())
        return service.inspect(source_path)

    def init_sidecar(
        self,
        config: str,
        file_path: str,
        index: str = "source_code",
    ) -> str:
        """Scaffold a sidecar YAML for a source file or directory.

        For a **file**, creates a ``.<file>.sidecar.yaml`` next to the source
        containing one stub entry per parsed symbol.

        For a **directory**, creates a single ``.sidecar.yaml`` inside the
        directory covering every non-sidecar source file found recursively.

        Existing sidecar files are overwritten.

        Args:
            config: Registry key that identifies the project configuration.
            file_path: Path to the source file or directory. Prefer an
                absolute path to minimize agent cognitive overhead; relative
                paths are still accepted and resolved from the index root.
            index: Name of the index whose root is used to resolve relative
                paths.  Defaults to ``"source_code"``.

        Returns:
            Absolute path to the created sidecar file as a string.
        """

        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(config, index, index_config, file_path)
        service = SidecarService(sidecar_store=YAMLSidecarStore())
        output = service.init(source_path)
        return str(output)

    def update_sidecar(
        self,
        config: str,
        file_path: str,
        updates: list[dict[str, Any]],
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Merge metadata updates into the sidecar file for a source file.

        Each entry in *updates* must contain a ``"symbol"`` key whose value
        matches a UID in the sidecar.  Optional keys ``"intent"`` (str) and
        ``"relations"`` (list[str]) are merged into the existing record.
        Missing sidecar fields are left unchanged.

        Args:
            config: Registry key that identifies the project configuration.
            file_path: Path to the source file whose sidecar should be
                updated. Prefer an absolute path to minimize agent cognitive
                overhead; relative paths are still accepted and resolved from
                the index root.
            updates: List of update objects.  Each object must have:

                * ``symbol`` *(str, required)* — UID of the target symbol.
                * ``intent`` *(str, optional)* — New intent description.
                * ``relations`` *(list[str], optional)* — New relation list.

            index: Name of the index whose root is used to resolve relative
                paths.  Defaults to ``"source_code"``.

        Returns:
            A dict with keys ``file_path``, ``sidecar_path``,
            ``applied_updates``, and ``total_symbols`` describing the result.
        """

        cfg = self._load_config(config)
        index_config = self._get_index_config(cfg, index)
        source_path = self._resolve_source_path(config, index, index_config, file_path)
        service = SidecarService(sidecar_store=YAMLSidecarStore())
        return service.update(source_path, json.dumps(updates, ensure_ascii=False))

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
            query: Natural-language description of what you are looking for.
            index: Name of the index to query.  Defaults to ``"source_code"``.
            top_k: Maximum number of results to return.  Defaults to ``5``.

        Returns:
            A serialisable dict containing the ranked results, or ``{}`` when
            the index is empty.
        """

        return smak_server.semantic_search(
            config=config,
            query=query,
            index=index,
            top_k=top_k,
        )

    @mcp.tool()
    def inspect_sidecar(
        config: str,
        file_path: str,
        index: str = "source_code",
    ) -> list[str]:
        """List the symbol UIDs that SMAK can parse from a source file.

        Runs the language-specific parser over *file_path* and returns the UID
        of every discovered code unit.  No files are written.  Use the result
        to know which symbol names are valid for ``update_sidecar``.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).
            file_path: Path to the source file. Absolute paths are strongly
                recommended to reduce agent ambiguity; relative paths are
                still supported and resolved against the index root folder.
            index: Index whose root resolves relative paths.  Defaults to
                ``"source_code"``.

        Returns:
            Ordered list of symbol UID strings (e.g.
            ``["module::ClassName", "module::ClassName::method"]``).
        """

        return smak_server.inspect_sidecar(
            config=config,
            file_path=file_path,
            index=index,
        )

    @mcp.tool()
    def init_sidecar(
        config: str,
        file_path: str,
        index: str = "source_code",
    ) -> str:
        """Create or overwrite a sidecar YAML stub for a file or directory.

        Parses *file_path* (or every source file inside a directory) and
        writes stub sidecar entries with empty ``intent`` and ``relations``
        fields.  Populate those fields afterwards with ``update_sidecar``.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).
            file_path: Path to a source file or directory. Absolute paths
                are strongly recommended to reduce agent ambiguity; relative
                paths are still supported and resolved against the index root
                folder.
            index: Index whose root resolves relative paths.  Defaults to
                ``"source_code"``.

        Returns:
            Absolute path to the created sidecar file as a string.
        """

        return smak_server.init_sidecar(
            config=config,
            file_path=file_path,
            index=index,
        )

    @mcp.tool()
    def update_sidecar(
        config: str,
        file_path: str,
        updates: list[dict[str, Any]],
        index: str = "source_code",
    ) -> dict[str, Any]:
        """Merge intent and relation metadata into a sidecar file.

        Each item in *updates* targets one symbol by UID and supplies new
        values for ``intent`` and/or ``relations``.  Fields not mentioned in
        an update object are left unchanged.

        Args:
            config: Registry key identifying the project (see
                ``list_available_configs``).
            file_path: Path to the source file whose sidecar should be
                updated. Absolute paths are strongly recommended to reduce
                agent ambiguity; relative paths are still supported and
                resolved against the index root folder.
            updates: List of update objects.  Each object must include:

                * ``symbol`` *(str, required)* — UID of the target symbol
                  (as returned by ``inspect_sidecar``).
                * ``intent`` *(str, optional)* — Human-readable description
                  of what the symbol does.
                * ``relations`` *(list[str], optional)* — UIDs of related
                  symbols.

            index: Index whose root resolves relative paths.  Defaults to
                ``"source_code"``.

        Returns:
            A dict with the following keys:

            * ``file_path`` — absolute path to the source file.
            * ``sidecar_path`` — absolute path to the updated sidecar file.
            * ``applied_updates`` — number of update entries processed.
            * ``total_symbols`` — total symbol count in the sidecar after the
              update.
        """

        return smak_server.update_sidecar(
            config=config,
            file_path=file_path,
            updates=updates,
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
    args = parser.parse_args()

    server = build_mcp_server(registry_path=Path(args.registry).resolve())
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
