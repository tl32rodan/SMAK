"""Command line interface for SMAK."""

from __future__ import annotations

import json
from pathlib import Path

import click

from smak.config import (
    EmbeddingConfig,
    IndexConfig,
    SmakConfig,
    load_config,
    load_embedding_config,
)
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

DEFAULT_MAX_WORKERS = 4
_DEFAULT_EMBEDDING_SETUP = str(Path(__file__).resolve().parent / "embedding_setup.yaml")


def _load_vector_store_for_cli(
    index: str,
    config_path: str,
    embedding_config: EmbeddingConfig | None = None,
) -> tuple[SmakConfig, IndexConfig, object]:
    cfg = init_config(load_config(config_path), embedding_config=embedding_config)
    index_config = cfg.get_index(index)
    if index_config is None:
        raise click.ClickException(f"Index '{index}' not found in configuration.")
    vector_store = load_and_validate_vector_store(index_config, cfg)
    return cfg, index_config, vector_store


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
@click.option(
    "--embedding-setup",
    default=_DEFAULT_EMBEDDING_SETUP,
    help="Path to embedding_setup.yaml",
)
def ingest(
    index: str,
    config: str,
    workers: int,
    incremental: bool,
    follow_symlinks: bool,
    sync: bool,
    embedding_setup: str,
) -> None:
    emb_cfg = load_embedding_config(embedding_setup)
    _, index_config, vector_store = _load_vector_store_for_cli(index, config, emb_cfg)
    targets = [Path(p) for p in index_config.paths]
    for p in targets:
        if not p.exists():
            raise click.ClickException(f"Path not found: {p}")
        if not p.is_dir() and not p.is_file():
            raise click.ClickException(f"Path is neither a file nor a directory: {p}")
    service = IngestService(vector_store=vector_store)
    paths_display = ", ".join(f"'{f}'" for f in targets)
    click.echo(f"Starting ingestion for {paths_display} -> Index: '{index}'...")
    stats = service.ingest_paths(
        targets,
        max_workers=workers,
        incremental=incremental,
        follow_symlinks=follow_symlinks,
        sync=sync,
        skip_file=sidecar_skip_file,
        on_ghost_source=on_ghost_source,
        embedder_loader=lambda: InternalNomicEmbedding(embedding_config=emb_cfg),
    )
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
@click.option(
    "--embedding-setup",
    default=_DEFAULT_EMBEDDING_SETUP,
    help="Path to embedding_setup.yaml",
)
def query_command(text: str, index: str, top_k: int, config: str, embedding_setup: str) -> None:
    emb_cfg = load_embedding_config(embedding_setup)
    cfg, index_config, vector_store = _load_vector_store_for_cli(index, config, emb_cfg)
    service = create_query_service(vector_store, cfg, index_config, embedding_config=emb_cfg)
    output_str = json.dumps(service.search(text, top_k=top_k), ensure_ascii=False, indent=4)
    click.echo(output_str.encode("utf-8"))


@main.group()
def sidecar() -> None:
    """Manage sidecar files."""


@sidecar.command("inspect")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON")
def sidecar_inspect(file_path: Path, json_output: bool) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Path must be a file: {file_path}")
    symbols = create_sidecar_service().inspect(file_path)
    if json_output:
        output_str = json.dumps(symbols, ensure_ascii=False, indent=4)
        click.echo(output_str.encode("utf-8"))
        return
    for symbol in symbols:
        click.echo(symbol)


@sidecar.command("update")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option("--symbol", default=None, help="Target a single symbol by name")
@click.option("--intent", default=None, help="Intent description for the symbol")
@click.option("--relations", default=None, help="Comma-separated list of relation UIDs")
def sidecar_update(
    file_path: Path,
    symbol: str | None,
    intent: str | None,
    relations: str | None,
) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Source file not found: {file_path}")
    parsed_relations = None
    if relations is not None:
        parsed_relations = [r.strip() for r in relations.split(",") if r.strip()]
    try:
        result = create_sidecar_service().update(
            file_path, symbol=symbol, intent=intent, relations=parsed_relations
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    output_str = json.dumps(result, ensure_ascii=False, indent=4)
    click.echo(output_str.encode("utf-8"))


@sidecar.command("clear")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option("--symbol", required=True, help="Symbol name to remove from sidecar")
def sidecar_clear(file_path: Path, symbol: str) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Source file not found: {file_path}")
    try:
        result = create_sidecar_service().clear_symbol(file_path, symbol)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    output_str = json.dumps(result, ensure_ascii=False, indent=4)
    click.echo(output_str.encode("utf-8"))


@main.command("doctor")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
@click.option(
    "--embedding-setup",
    default=_DEFAULT_EMBEDDING_SETUP,
    help="Path to embedding_setup.yaml",
)
def doctor(config: str, embedding_setup: str) -> None:
    emb_cfg = load_embedding_config(embedding_setup)
    cfg = init_config(load_config(config), embedding_config=emb_cfg)
    service = create_doctor_service(cfg)
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
    "sidecar_clear",
    "sidecar_inspect",
    "sidecar_update",
    "doctor",
]
