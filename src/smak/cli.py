"""Command line interface for SMAK."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import click
from tqdm import tqdm

from smak.config import SmakConfig, load_config
from smak.db.adapter import VectorDocument
from smak.embedding import (
    EmbeddingProbe,
    InternalNomicEmbedding,
    initialize_embedding_dimensions,
    validate_vector_store_dimension,
)
from smak.ingest.parsers import IssueParser, Parser, PerlParser, PythonParser, SimpleLineParser
from smak.ingest.pipeline import IngestPipeline, IntegrityError
from smak.ingest.sidecar import SidecarManager
from smak.utils.yaml import safe_dump, safe_load

SIDECAR_SUFFIXES = (".sidecar.yaml", ".sidecar.yml")
DEFAULT_MAX_WORKERS = 4


@dataclass(frozen=True)
class IngestStats:
    """Statistics returned after ingesting a folder."""

    files: int
    vectors: int
    skipped: int


def _load_vector_store(index_name: str, config: SmakConfig):
    from smak.storage.faiss_adapter import load_faiss_store

    provider = (config.storage.provider or "faiss").lower()
    if provider != "faiss":
        raise click.ClickException(f"Unsupported vector store provider: {provider}")
    return load_faiss_store(
        uri=config.storage.uri,
        collection_name=index_name,
        dim=config.embedding_dimensions,
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
            (
                "    description: Contains the project's source code (Python, Perl), "
                "function definitions, and logic."
            ),
            "  - name: issues",
            (
                "    description: Contains historical bug reports, GitHub issues, and Jira "
                "tickets describing known problems."
            ),
            "  - name: tests",
            "    description: Contains unit tests, integration tests, and test cases.",
            "  - name: documentation",
            (
                "    description: Contains architecture diagrams, API docs, and general "
                "knowledge base."
            ),
            "",
        ]
    )


def _parser_for_path(path: Path, *, root_path: Path | None = None) -> Parser:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return PythonParser(root_path=str(root_path) if root_path else None)
    if suffix in {".pl", ".pm"}:
        return PerlParser(root_path=str(root_path) if root_path else None)
    if suffix in {".md", ".markdown"}:
        return IssueParser()
    return SimpleLineParser()


def _sidecar_payload(path: Path) -> str | None:
    for suffix in SIDECAR_SUFFIXES:
        candidate = path.with_name(f"{path.name}{suffix}")
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return None


def _iter_source_files(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(SIDECAR_SUFFIXES):
            continue
        yield path


def _symbols_for_path(path: Path, *, root_path: Path | None = None) -> list[str]:
    parser = _parser_for_path(path, root_path=root_path)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - file read failures are OS-specific
        raise click.ClickException(f"Unable to read file: {exc}") from exc
    try:
        units = parser.parse(content, source=str(path))
    except Exception as exc:  # pragma: no cover - parser errors
        raise click.ClickException(f"Failed to parse {path}: {exc}") from exc
    return [unit.uid for unit in units]


def _source_key(path: Path, workspace_root: Path | None = None) -> str:
    if workspace_root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path)


def _source_mtime(path: Path) -> float:
    return path.stat().st_mtime


def _unit_up_to_date(vector_store: object, unit_id: str, file_mtime: float) -> bool:
    get_by_id = getattr(vector_store, "get_by_id", None)
    if not callable(get_by_id):
        return False
    payload = get_by_id(unit_id)
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    try:
        return float(metadata.get("source_mtime")) == float(file_mtime)
    except (TypeError, ValueError):
        return False


def _ingest_folder(
    folder: Path,
    index: str,
    config: SmakConfig,
    vector_store_loader: Callable[[str, SmakConfig], object] | None = None,
    node_class_loader: Callable[[], type] | None = None,
    embedder_loader: Callable[[], EmbeddingProbe] | None = None,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    show_progress: bool = False,
    workspace_root: Path | None = None,
    incremental: bool = True,
) -> IngestStats:
    _ = node_class_loader
    embedder_factory = embedder_loader or InternalNomicEmbedding
    embedder = embedder_factory()
    config = initialize_embedding_dimensions(config, embedder)
    vector_store_factory = vector_store_loader or _load_vector_store
    vector_store = vector_store_factory(index, config)
    validate_vector_store_dimension(vector_store, config.embedding_dimensions)
    sidecar_manager = SidecarManager()

    paths = list(_iter_source_files(folder))
    file_count = 0
    vector_count = 0
    skipped_count = 0

    def process_file(file_path: Path) -> tuple[list[VectorDocument], str | None, bool]:
        parser = _parser_for_path(file_path, root_path=workspace_root)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        parsed_units = parser.parse(content, source=str(file_path))
        source_mtime = _source_mtime(file_path)
        if incremental and parsed_units:
            if _unit_up_to_date(vector_store, parsed_units[0].uid, source_mtime):
                return [], None, True

        pipeline = IngestPipeline(parser=parser, embedder=embedder, sidecar_manager=sidecar_manager)
        sidecar = _sidecar_payload(file_path)
        result = pipeline.run(
            content,
            source=str(file_path),
            sidecar_payload=sidecar,
            compute_embeddings=True,
        )
        source_key = _source_key(file_path, workspace_root)
        docs = pipeline.to_vector_documents(
            result,
            source=source_key,
            source_mtime=source_mtime,
        )
        return docs, source_key, False

    max_workers = max(1, min(max_workers, os.cpu_count() or max_workers))
    progress = tqdm(
        total=len(paths),
        disable=not show_progress or not sys.stderr.isatty(),
        desc="Ingesting",
    )
    pending_docs: list[VectorDocument] = []
    refreshed_sources: set[str] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, path) for path in paths]
        for future in as_completed(futures):
            docs, source_key, skipped = future.result()
            if skipped:
                skipped_count += 1
            else:
                file_count += 1
                if source_key:
                    refreshed_sources.add(source_key)
                pending_docs.extend(docs)
                vector_count += len(docs)
            progress.update(1)
    progress.close()

    if hasattr(vector_store, "delete_by_metadata"):
        for source_key in sorted(refreshed_sources):
            vector_store.delete_by_metadata("source", source_key)
    if pending_docs:
        vector_store.add(pending_docs)

    return IngestStats(files=file_count, vectors=vector_count, skipped=skipped_count)


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


def _normalize_sidecar_updates(updates: Any) -> list[dict[str, Any]]:
    if not isinstance(updates, list):
        raise click.ClickException("'updates' must be a list.")
    normalized: list[dict[str, Any]] = []
    for entry in updates:
        if not isinstance(entry, dict):
            raise click.ClickException("Each update must be an object.")
        symbol = entry.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise click.ClickException("Each update requires a non-empty 'symbol'.")
        record: dict[str, Any] = {"name": symbol}
        if "intent" in entry:
            record["intent"] = str(entry.get("intent") or "")
        if "relations" in entry:
            relations = entry.get("relations")
            if not isinstance(relations, list):
                raise click.ClickException("'relations' must be a list when provided.")
            record["relations"] = [str(item) for item in relations]
        normalized.append(record)
    return normalized


def _merge_sidecar_updates(sidecar_path: Path, updates: list[dict[str, Any]]) -> int:
    payload = safe_load(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    if not isinstance(payload, dict):
        payload = {}
    existing = payload.get("symbols")
    if not isinstance(existing, list):
        existing = []
    table: dict[str, dict[str, Any]] = {}
    for entry in existing:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            table[entry["name"]] = dict(entry)
    for update in updates:
        name = update["name"]
        target = table.get(name, {"name": name})
        target.update(update)
        table[name] = target
    payload["symbols"] = sorted(table.values(), key=lambda item: str(item.get("name", "")))
    sidecar_path.write_text(safe_dump(payload), encoding="utf-8")
    return len(payload["symbols"])


def _vector_store_stats(vector_store: object) -> dict[str, Any]:
    stats: dict[str, Any] = {"total_vectors": None, "last_update": None}
    count = getattr(vector_store, "count", None)
    if callable(count):
        stats["total_vectors"] = count()
    last_update = getattr(vector_store, "last_update", None)
    if callable(last_update):
        stats["last_update"] = last_update()
    return stats


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
    """Ingest a folder into the specified vector index."""
    if not folder.exists() or not folder.is_dir():
        raise click.ClickException(f"Folder not found: {folder}")

    try:
        cfg = load_config(config)
    except Exception as exc:
        raise click.ClickException(f"Error loading config: {exc}") from exc

    click.echo(f"Starting ingestion for '{folder}' -> Index: '{index}'...")
    try:
        stats = _ingest_folder(
            folder,
            index,
            cfg,
            max_workers=workers,
            show_progress=True,
            workspace_root=_load_workspace_root(config),
            incremental=incremental,
        )
    except IntegrityError as exc:
        raise click.ClickException(f"Sidecar integrity error: {exc}") from exc
    except Exception as exc:
        raise click.ClickException(f"Ingestion failed: {exc}") from exc

    click.echo("Ingestion Complete!")
    click.echo(f"   - Processed Files: {stats.files}")
    click.echo(f"   - Skipped Files: {stats.skipped}")
    click.echo(f"   - Vectors Added: {stats.vectors}")


@main.command()
@click.option("--path", "config_path", default="workspace_config.yaml", help="Config path")
@click.option("--force", is_flag=True, help="Overwrite existing config file")
def init(config_path: str, force: bool) -> None:
    """Generate a blank workspace config file."""
    target = Path(config_path)
    if target.exists() and not force:
        raise click.ClickException(f"Config already exists: {target}")
    target.write_text(_default_config_template(), encoding="utf-8")
    click.echo(f"Wrote workspace config to {target}")


@main.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option(
    "--config",
    "config_path",
    default="workspace_config.yaml",
    help="Path to workspace config",
)
@click.option("--json-output", is_flag=True, help="Emit machine-readable JSON")
def search(path: Path, config_path: str, json_output: bool) -> None:
    """Print symbols for a file to populate sidecar metadata."""
    if not path.exists():
        raise click.ClickException(f"Path not found: {path}")
    if not path.is_file():
        raise click.ClickException(f"Path must be a file: {path}")
    workspace_root = _load_workspace_root(config_path)
    symbols = _symbols_for_path(path, root_path=workspace_root)
    if json_output:
        click.echo(json.dumps(symbols, ensure_ascii=False))
        return
    if not symbols:
        click.echo("No symbols found.")
        return
    for symbol in symbols:
        click.echo(symbol)


@main.command("query")
@click.argument("text", type=str)
@click.option("--index", required=True, help="Target index name")
@click.option("--top-k", default=5, show_default=True, type=int, help="Result count")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
def query_command(text: str, index: str, top_k: int, config: str) -> None:
    """Run semantic vector retrieval and print JSON results."""
    _, vector_store = _load_vector_store_for_cli(index, config)
    embedder = InternalNomicEmbedding()
    query_vector = embedder.get_text_embedding(text)
    results = vector_store.search(query_vector, top_k=max(top_k, 1))
    click.echo(json.dumps(results, ensure_ascii=False))


@main.group()
def sidecar() -> None:
    """Manage sidecar files."""


@sidecar.command("init")
@click.argument("target_path", type=click.Path(path_type=Path))
@click.option(
    "--config",
    "config_path",
    default="workspace_config.yaml",
    help="Path to workspace config",
)
def sidecar_init(target_path: Path, config_path: str) -> None:
    """Generate a sidecar skeleton for a source file or directory."""
    if not target_path.exists():
        raise click.ClickException(f"Path not found: {target_path}")
    workspace_root = _load_workspace_root(config_path)
    if target_path.is_dir():
        symbols: list[str] = []
        for source_path in sorted(_iter_source_files(target_path)):
            symbols.extend(_symbols_for_path(source_path, root_path=workspace_root))
        lines = ["symbols:"]
        for symbol in symbols:
            lines.extend([f"  - name: {symbol}", '    intent: ""', "    relations: []"])
        payload = "\n".join(lines) + "\n" if symbols else "symbols: []\n"
        output = target_path / "sidecar.yaml"
        output.write_text(payload, encoding="utf-8")
        click.echo(f"Wrote sidecar template to {output}")
        return
    if not target_path.is_file():
        raise click.ClickException(f"Path must be a file or directory: {target_path}")
    parser = _parser_for_path(target_path, root_path=workspace_root)
    units = parser.parse(
        target_path.read_text(encoding="utf-8", errors="replace"),
        source=str(target_path),
    )
    lines = ["symbols:"]
    for unit in units:
        lines.extend(
            [
                f"  - name: {unit.metadata.get('symbol')}",
                '    intent: ""',
                "    relations: []",
            ]
        )
    payload = "\n".join(lines) + "\n" if units else "symbols: []\n"
    output = target_path.with_name(f"{target_path.name}.sidecar.yaml")
    output.write_text(payload, encoding="utf-8")
    click.echo(f"Wrote sidecar template to {output}")


@sidecar.command("update")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option("--updates", required=True, help="JSON list of symbol updates")
@click.option("--reingest", is_flag=True, help="Trigger ingest after sidecar update")
@click.option("--index", default="source_code", help="Index name for optional re-ingest")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
def sidecar_update(file_path: Path, updates: str, reingest: bool, index: str, config: str) -> None:
    """Merge structured symbol updates into a file sidecar YAML."""
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Source file not found: {file_path}")
    try:
        parsed_updates = json.loads(updates)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid updates JSON: {exc}") from exc
    normalized = _normalize_sidecar_updates(parsed_updates)
    sidecar_path = file_path.with_name(f"{file_path.name}.sidecar.yaml")
    total_symbols = _merge_sidecar_updates(sidecar_path, normalized)
    click.echo(
        json.dumps(
            {
                "file_path": str(file_path),
                "sidecar_path": str(sidecar_path),
                "applied_updates": len(normalized),
                "total_symbols": total_symbols,
            },
            ensure_ascii=False,
        )
    )
    if reingest:
        cfg = load_config(config)
        _ingest_folder(
            file_path.parent,
            index,
            cfg,
            max_workers=1,
            show_progress=False,
            workspace_root=_load_workspace_root(config),
            incremental=True,
        )


@main.command("stats")
@click.option("--index", required=True, help="Target index name")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
def stats_command(index: str, config: str) -> None:
    """Print vector store statistics as JSON."""
    _, vector_store = _load_vector_store_for_cli(index, config)
    click.echo(json.dumps(_vector_store_stats(vector_store), ensure_ascii=False))


@main.command("doctor")
@click.option("--path", "target_path", default=".", type=click.Path(path_type=Path))
def doctor(target_path: Path) -> None:
    """Check sidecar integrity across a path."""
    issues: list[str] = []
    root = target_path if target_path.is_dir() else target_path.parent
    for sidecar_file in root.rglob("*.sidecar.yaml"):
        source = sidecar_file.with_name(sidecar_file.name.replace(".sidecar.yaml", ""))
        if not source.exists():
            issues.append(f"Orphaned sidecar: {sidecar_file}")
    for sidecar_file in root.rglob("*.sidecar.yml"):
        source = sidecar_file.with_name(sidecar_file.name.replace(".sidecar.yml", ""))
        if not source.exists():
            issues.append(f"Orphaned sidecar: {sidecar_file}")
    if issues:
        for issue in issues:
            click.echo(issue)
        raise click.ClickException("Mesh diagnostics found problems.")
    click.echo("Mesh diagnostics passed.")


__all__ = [
    "IngestStats",
    "ingest",
    "init",
    "main",
    "search",
    "query_command",
    "sidecar",
    "sidecar_init",
    "sidecar_update",
    "stats_command",
    "doctor",
]
