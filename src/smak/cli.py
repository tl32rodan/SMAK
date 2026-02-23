"""Command line interface for SMAK."""

from __future__ import annotations

import json
from pathlib import Path

import click

from smak.config import SmakConfig, load_config
from smak.embedding import (
    InternalNomicEmbedding,
    initialize_embedding_dimensions,
    validate_vector_store_dimension,
)
from smak.ingest.pipeline import IntegrityError
from smak.services import DoctorService, IngestService, QueryService, SidecarService

DEFAULT_MAX_WORKERS = 4


def _load_vector_store(index_name: str, config: SmakConfig):
    from smak.storage.faiss_adapter import load_faiss_store

    provider = (config.storage.provider or "faiss").lower()
    if provider != "faiss":
        raise click.ClickException(f"Unsupported vector store provider: {provider}")
    return load_faiss_store(
        uri=config.storage.uri, collection_name=index_name, dim=config.embedding_dimensions
    )


def _default_config_template() -> str:
    return "\n".join(
        [
            "# SMAK Workspace Configuration",
            "",
            "storage:",
            "  provider: faiss",
            "  uri: ./smak_data",
            "",
            "llm:",
            "  provider: qwen",
            "  model: qwen3_235B_A22B",
            "  temperature: 0.0",
            "  # api_base: http://localhost:11434/v1",
            "",
            "indices:",
            "  - name: source_code",
            "    description: Contains the project's source code (Python, Perl), "
            "function definitions, and logic.",
            "  - name: issues",
            "    description: Contains historical bug reports, GitHub issues, "
            "and Jira tickets describing known problems.",
            "  - name: tests",
            "    description: Contains unit tests, integration tests, and test cases.",
            "  - name: documentation",
            "    description: Contains architecture diagrams, API docs, and "
            "general knowledge base.",
            "",
        ]
    )


def _load_workspace_root(config_path: str) -> Path | None:
    config_file = Path(config_path)
    return config_file.resolve().parent if config_file.exists() else None


def _load_vector_store_for_cli(index: str, config_path: str) -> tuple[SmakConfig, object]:
    cfg = load_config(config_path)
    embedder = InternalNomicEmbedding()
    cfg = initialize_embedding_dimensions(cfg, embedder)
    vector_store = _load_vector_store(index, cfg)
    validate_vector_store_dimension(vector_store, cfg.embedding_dimensions)
    return cfg, vector_store


@click.group()
def main() -> None:
    """SMAK: Semantic Mesh Agentic Kernel CLI."""


@main.command()
@click.option("--folder", required=True, type=click.Path(path_type=Path), help="Path to ingest")
@click.option("--index", required=True, help="Target index name (e.g., source_code)")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
@click.option("--workers", default=DEFAULT_MAX_WORKERS, help="Max parallel workers")
@click.option("--incremental/--full", default=True, help="Enable mtime-based incremental ingest")
def ingest(folder: Path, index: str, config: str, workers: int, incremental: bool) -> None:
    if not folder.exists() or not folder.is_dir():
        raise click.ClickException(f"Folder not found: {folder}")
    cfg = load_config(config)
    service = IngestService(config=cfg)
    click.echo(f"Starting ingestion for '{folder}' -> Index: '{index}'...")
    try:
        stats = service.ingest_folder(
            folder,
            index,
            max_workers=workers,
            workspace_root=_load_workspace_root(config),
            incremental=incremental,
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
@click.option("--top-k", default=5, show_default=True, type=int, help="Result count")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
def query_command(text: str, index: str, top_k: int, config: str) -> None:
    _, vector_store = _load_vector_store_for_cli(index, config)
    service = QueryService(vector_store=vector_store)
    click.echo(json.dumps(service.search(text, top_k=top_k), ensure_ascii=False))


@main.group()
def sidecar() -> None:
    """Manage sidecar files."""


@sidecar.command("init")
@click.argument("target_path", type=click.Path(path_type=Path))
@click.option(
    "--config", "config_path", default="workspace_config.yaml", help="Path to workspace config"
)
def sidecar_init(target_path: Path, config_path: str) -> None:
    if not target_path.exists():
        raise click.ClickException(f"Path not found: {target_path}")
    output = SidecarService().init(target_path, workspace_root=_load_workspace_root(config_path))
    click.echo(f"Wrote sidecar template to {output}")


@sidecar.command("inspect")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option(
    "--config", "config_path", default="workspace_config.yaml", help="Path to workspace config"
)
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON")
def sidecar_inspect(file_path: Path, config_path: str, json_output: bool) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Path must be a file: {file_path}")
    symbols = SidecarService().inspect(file_path, workspace_root=_load_workspace_root(config_path))
    if json_output:
        click.echo(json.dumps(symbols, ensure_ascii=False))
        return
    for symbol in symbols:
        click.echo(symbol)


@sidecar.command("update")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option("--updates", required=True, help="JSON list of symbol updates")
@click.option("--reingest", is_flag=True, help="Trigger ingest after sidecar update")
@click.option("--index", default="source_code", help="Index name for optional re-ingest")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
def sidecar_update(file_path: Path, updates: str, reingest: bool, index: str, config: str) -> None:
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Source file not found: {file_path}")
    try:
        result = SidecarService().update(file_path, updates)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid updates JSON: {exc}") from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, ensure_ascii=False))
    if reingest:
        cfg = load_config(config)
        IngestService(cfg).ingest_folder(
            file_path.parent,
            index,
            max_workers=1,
            workspace_root=_load_workspace_root(config),
            incremental=True,
        )


@main.command("doctor")
@click.option("--path", "target_path", default=".", type=click.Path(path_type=Path))
@click.option("--index", default="source_code", help="Index name")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
def doctor(target_path: Path, index: str, config: str) -> None:
    _, vector_store = _load_vector_store_for_cli(index, config)
    service = DoctorService(vector_store=vector_store)
    issues = service.validate_sidecars(target_path)
    dangling = service.validate_mesh_integrity(target_path)
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
