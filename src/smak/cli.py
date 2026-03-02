"""Command line interface for SMAK."""

from __future__ import annotations

import json
from pathlib import Path

import click

from smak.config import IndexConfig, SmakConfig, load_config
from smak.services import DoctorService, IngestService, QueryService, SidecarService
from smak.services.ingest.pipeline import IntegrityError
from smak.services.relation_resolver import SidecarRelationResolver
from smak.sidecar.store import SidecarStore
from smak.utils.embedding import (
    InternalNomicEmbedding,
    initialize_embedding_dimensions,
    validate_vector_store_dimension,
)

DEFAULT_MAX_WORKERS = 4
DEFAULT_INDEX_DATA_DIR = "./smak_data"


def _resolve_config(cfg: SmakConfig, config_path: str) -> SmakConfig:
    config_file = Path(config_path)
    base_path = config_file.resolve().parent if config_file.exists() else Path.cwd().resolve()
    resolved_indices = []
    for index in cfg.indices:
        # Resolve path
        resolved_path = str((base_path / Path(index.path).expanduser()).resolve()) if not Path(index.path).expanduser().is_absolute() else str(Path(index.path).expanduser().resolve())
        # Resolve uri
        if index.uri:
            uri_path = Path(index.uri).expanduser()
            resolved_uri = str((base_path / uri_path).resolve()) if not uri_path.is_absolute() else str(uri_path.resolve())
        else:
            resolved_uri = str((base_path / DEFAULT_INDEX_DATA_DIR / index.name).resolve())
        resolved_indices.append(
            IndexConfig(
                name=index.name,
                description=index.description,
                path=resolved_path,
                uri=resolved_uri,
            )
        )
    return SmakConfig(indices=resolved_indices, embedding_dimensions=cfg.embedding_dimensions)


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
            "    path: ./src",
            "    # Customize uri if you want this index stored elsewhere.",
            "    uri: ./smak_data/source_code",
            "  - name: issues",
            "    description: Contains historical bug reports, GitHub issues, "
            "and Jira tickets describing known problems.",
            "    path: ./issues",
            "  - name: tests",
            "    description: Contains unit tests, integration tests, and test cases.",
            "  - name: documentation",
            "    description: Contains architecture diagrams, API docs, and "
            "general knowledge base.",
            "",
        ]
    )


def _load_vector_store_for_cli(index: str, config_path: str) -> tuple[SmakConfig, IndexConfig, object]:
    cfg = load_config(config_path)
    cfg = _resolve_config(cfg, config_path)
    index_config = next((entry for entry in cfg.indices if entry.name == index), None)
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
def ingest(
    index: str,
    config: str,
    workers: int,
    incremental: bool,
    follow_symlinks: bool,
) -> None:
    _, index_config, vector_store = _load_vector_store_for_cli(index, config)
    folder = Path(index_config.path)
    if not folder.exists() or not folder.is_dir():
        raise click.ClickException(f"Folder not found: {folder}")
    service = IngestService(vector_store=vector_store)
    click.echo(f"Starting ingestion for '{folder}' -> Index: '{index}'...")
    try:
        stats = service.ingest_folder(
            folder,
            max_workers=workers,
            incremental=incremental,
            follow_symlinks=follow_symlinks,
        )
    except IntegrityError as exc:
        raise click.ClickException(f"Sidecar integrity error: {exc}") from exc
    click.echo("Ingestion Complete!")
    click.echo(f"   - Processed Files: {stats.files}")
    click.echo(f"   - Skipped Files: {stats.skipped}")
    click.echo(f"   - Vectors Added: {stats.vectors}")


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
    sidecar_store = SidecarStore()
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
    sidecar_store = SidecarStore()
    output = SidecarService(sidecar_store).init(target_path)
    click.echo(f"Wrote sidecar template to {output}")


@sidecar.command("inspect")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON")
def sidecar_inspect(file_path: Path, json_output: bool) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Path must be a file: {file_path}")
    sidecar_store = SidecarStore()
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
    cfg = _resolve_config(cfg, config)
    embedder = InternalNomicEmbedding()
    cfg = initialize_embedding_dimensions(cfg, embedder)

    def _load_store(index_name: str) -> object:
        index_config = next((entry for entry in cfg.indices if entry.name == index_name), None)
        if index_config is None:
            raise click.ClickException(f"Index '{index_name}' not found in configuration.")
        vector_store = _load_vector_store(index_config, cfg)
        validate_vector_store_dimension(vector_store, cfg.embedding_dimensions)
        return vector_store

    service = DoctorService(config=cfg, vector_store_loader=_load_store)

    issues = []
    dangling = []
    for index_config in cfg.indices:
        target_path = Path(index_config.path)
        if target_path.exists():
            issues.extend(service.validate_sidecars(target_path))
            dangling.extend(service.validate_mesh_integrity(target_path))
    problems = [*issues, *dangling]
    if problems:
        for issue in problems:
            click.echo(issue)
        raise click.ClickException("Mesh diagnostics found problems.")
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
