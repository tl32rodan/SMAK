"""Command line interface for SMAK."""

from __future__ import annotations

import json
from pathlib import Path

import click

from smak.config import IndexConfig, SmakConfig, load_config
from smak.services import DoctorService, IngestService, QueryService, SidecarService
from smak.services.ingest.pipeline import IntegrityError
from smak.services.relation_resolver import SidecarRelationResolver
from smak.sidecar.store import YAMLSidecarStore
from smak.utils.embedding import (
    InternalNomicEmbedding,
    initialize_embedding_dimensions,
    validate_vector_store_dimension,
)

DEFAULT_MAX_WORKERS = 4


def _load_vector_store(index_config: IndexConfig, config: SmakConfig):
    from smak.storage.faiss_adapter import load_faiss_store

    return load_faiss_store(
        uri=index_config.uri,  # URI is already absolute here
        collection_name=index_config.name,
        dim=config.embedding_dimensions,
    )


def _default_config_template() -> str:
    return "\n".join(
        [
            "# SMAK Workspace Configuration",
            "",
            "indices:",
            "  - name: source_code",
            "    description: Contains the project's source code (Python, Perl), "
            "function definitions, and logic.",
            "    paths:",
            "      - ./src",
            "    # Customize uri if you want this index stored elsewhere.",
            "    uri: ./smak_data/source_code",
            "  - name: issues",
            "    description: Contains historical bug reports, GitHub issues, "
            "and Jira tickets describing known problems.",
            "    paths:",
            "      - ./issues",
            "  - name: tests",
            "    description: Contains unit tests, integration tests, and test cases.",
            "    paths:",
            "      - ./tests",
            "  - name: documentation",
            "    description: Contains architecture diagrams, API docs, and "
            "general knowledge base.",
            "    paths:",
            "      - ./documentation",
            "",
        ]
    )


def _load_vector_store_for_cli(
    index: str,
    config_path: str,
) -> tuple[SmakConfig, IndexConfig, object]:
    cfg = load_config(config_path)
    index_config = cfg.get_index(index)
    if index_config is None:
        raise click.ClickException(f"Index '{index}' not found in configuration.")
    embedder = InternalNomicEmbedding()
    cfg = initialize_embedding_dimensions(cfg, embedder)
    vector_store = _load_vector_store(index_config, cfg)
    validate_vector_store_dimension(vector_store, cfg.embedding_dimensions)
    return cfg, index_config, vector_store


@click.group()
def main() -> None:
    """SMAK: Semantic Mesh Agentic Kernel CLI."""


@main.command()
@click.option("--index", required=True, help="Target index name (e.g., source_code)")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
@click.option("--workers", default=DEFAULT_MAX_WORKERS, help="Max parallel workers")
@click.option("--incremental/--full", default=True, help="Enable mtime-based incremental ingest")
@click.option(
    "--follow-symlinks/--no-follow-symlinks",
    default=True,
    help="Follow symlinked directories during ingest",
)
@click.option(
    "--sync",
    is_flag=True,
    help="Prune deleted files and their sidecars from the index",
)
def ingest(
    index: str,
    config: str,
    workers: int,
    incremental: bool,
    follow_symlinks: bool,
    sync: bool,
) -> None:
    _, index_config, vector_store = _load_vector_store_for_cli(index, config)
    folders = [Path(p) for p in index_config.paths]
    for folder in folders:
        if not folder.exists() or not folder.is_dir():
            raise click.ClickException(f"Folder not found: {folder}")
    service = IngestService(vector_store=vector_store)
    paths_display = ", ".join(f"'{f}'" for f in folders)
    click.echo(f"Starting ingestion for {paths_display} -> Index: '{index}'...")
    try:
        stats = service.ingest_paths(
            folders,
            max_workers=workers,
            incremental=incremental,
            follow_symlinks=follow_symlinks,
            sync=sync,
        )
    except IntegrityError as exc:
        raise click.ClickException(f"Sidecar integrity error: {exc}") from exc
    click.echo("Ingestion Complete!")
    click.echo(f"   - Processed Files: {stats.files}")
    click.echo(f"   - Skipped Files: {stats.skipped}")
    click.echo(f"   - Vectors Added: {stats.vectors}")
    click.echo(f"   - Ghost Files Pruned: {stats.deleted}")


@main.command()
@click.option("--path", "config_path", default="workspace_config.yaml", help="Config path")
@click.option("--force", is_flag=True, help="Overwrite existing config file")
def init(config_path: str, force: bool) -> None:
    target = Path(config_path)
    if target.exists() and not force:
        raise click.ClickException(f"Config already exists: {target}")
    target.write_text(_default_config_template(), encoding="utf-8")
    click.echo(f"Wrote workspace config to {target}")


@main.command("query")
@click.argument("text", type=str)
@click.option("--index", required=True, help="Target index name")
@click.option("--top-k", default=1, show_default=True, type=int, help="Result count")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
def query_command(text: str, index: str, top_k: int, config: str) -> None:
    cfg, index_config, vector_store = _load_vector_store_for_cli(index, config)
    sidecar_store = YAMLSidecarStore()
    service = QueryService(
        vector_store=vector_store,
        config=cfg,
        vector_store_loader=_load_vector_store,
        index_config=index_config,
        relation_resolver=SidecarRelationResolver(sidecar_store),
    )
    output_str = json.dumps(service.search(text, top_k=top_k), ensure_ascii=False, indent=4)
    click.echo(output_str.encode("utf-8"))


@main.group()
def sidecar() -> None:
    """Manage sidecar files."""


@sidecar.command("init")
@click.argument("target_path", type=click.Path(path_type=Path))
def sidecar_init(target_path: Path) -> None:
    if not target_path.exists():
        raise click.ClickException(f"Path not found: {target_path}")
    sidecar_store = YAMLSidecarStore()
    output = SidecarService(sidecar_store).init(target_path)
    click.echo(f"Wrote sidecar template to {output}")


@sidecar.command("inspect")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON")
def sidecar_inspect(file_path: Path, json_output: bool) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Path must be a file: {file_path}")
    sidecar_store = YAMLSidecarStore()
    symbols = SidecarService(sidecar_store).inspect(file_path)
    if json_output:
        output_str = json.dumps(symbols, ensure_ascii=False, indent=4)
        click.echo(output_str.encode("utf-8"))
        return
    for symbol in symbols:
        click.echo(symbol)


@sidecar.command("update")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option("--updates", required=True, help="JSON list of symbol updates")
def sidecar_update(file_path: Path, updates: str) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Source file not found: {file_path}")
    try:
        result = SidecarService().update(file_path, updates)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid updates JSON: {exc}") from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    output_str = json.dumps(result, ensure_ascii=False, indent=4)
    click.echo(output_str.encode("utf-8"))


@main.command("doctor")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
def doctor(config: str) -> None:
    cfg = load_config(config)
    embedder = InternalNomicEmbedding()
    cfg = initialize_embedding_dimensions(cfg, embedder)

    def _load_store(index_name: str) -> object:
        index_config = cfg.get_index(index_name)
        if index_config is None:
            raise click.ClickException(f"Index '{index_name}' not found in configuration.")
        vector_store = _load_vector_store(index_config, cfg)
        validate_vector_store_dimension(vector_store, cfg.embedding_dimensions)
        return vector_store

    service = DoctorService(config=cfg, vector_store_loader=_load_store)
    try:
        service.validate_all()
    except RuntimeError as exc:
        for issue in str(exc).split("\n"):
            click.echo(issue)
        raise click.ClickException("Mesh diagnostics found problems.") from exc
    click.echo("Mesh diagnostics passed.")


__all__ = [
    "ingest",
    "init",
    "main",
    "query_command",
    "sidecar",
    "sidecar_init",
    "sidecar_inspect",
    "sidecar_update",
    "doctor",
]
