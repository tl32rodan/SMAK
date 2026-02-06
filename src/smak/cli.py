"""Command line interface for SMAK."""

from __future__ import annotations

import importlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import click

from smak.config import SmakConfig, load_config
from smak.embedding import initialize_embedding_dimensions, validate_vector_store_dimension
from smak.ingest.parsers import IssueParser, Parser, PerlParser, PythonParser, SimpleLineParser
from smak.ingest.pipeline import Embedder, IngestPipeline, IntegrityError
from smak.ingest.sidecar import SidecarManager
from smak.models import InternalNomicEmbedding

SIDECAR_SUFFIXES = (".sidecar.yaml", ".sidecar.yml")
DEFAULT_MAX_WORKERS = 4


class _NoopProgress:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.disable = True

    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def _get_tqdm():
    spec = importlib.util.find_spec("tqdm")
    if spec is None:
        return _NoopProgress
    module = importlib.import_module("tqdm")
    return getattr(module, "tqdm")


@dataclass(frozen=True)
class IngestStats:
    """Statistics returned after ingesting a folder."""

    files: int
    vectors: int


def _load_text_node_class():
    spec = importlib.util.find_spec("llama_index.core.schema")
    if spec is None:  # pragma: no cover - guard for missing dependency
        raise click.ClickException(
            "Critical dependency 'llama-index-core' not found. "
            "Did you run pip install llama-index-core?"
        )
    module = importlib.import_module("llama_index.core.schema")
    return getattr(module, "TextNode")


def _load_vector_store(index_name: str, config: SmakConfig):
    from smak.storage.faiss_adapter import load_faiss_store

    try:
        provider = (config.storage.provider or "faiss").lower()
        if provider != "faiss":
            raise click.ClickException(f"Unsupported vector store provider: {provider}")
        return load_faiss_store(
            uri=config.storage.uri,
            collection_name=index_name,
            dim=config.embedding_dimensions,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - guard for missing dependency
        raise click.ClickException(str(exc)) from exc


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


def _ingest_folder(
    folder: Path,
    index: str,
    config: SmakConfig,
    vector_store_loader: Callable[[str, SmakConfig], object] | None = None,
    node_class_loader: Callable[[], type] | None = None,
    embedder_loader: Callable[[], Embedder] | None = None,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    show_progress: bool = False,
    workspace_root: Path | None = None,
) -> IngestStats:
    embedder_factory = embedder_loader or InternalNomicEmbedding
    embedder = embedder_factory()
    config = initialize_embedding_dimensions(config, embedder)
    vector_store_factory = vector_store_loader or _load_vector_store
    node_factory = node_class_loader or _load_text_node_class
    vector_store = vector_store_factory(index, config)
    node_class = node_factory()
    validate_vector_store_dimension(vector_store, config.embedding_dimensions)
    sidecar_manager = SidecarManager()

    paths = list(_iter_source_files(folder))
    file_count = 0
    vector_count = 0

    lock = threading.Lock()

    def process_file(file_path: Path) -> tuple[Path, int]:
        parser = _parser_for_path(file_path, root_path=workspace_root)
        pipeline = IngestPipeline(
            parser=parser,
            embedder=embedder,
            sidecar_manager=sidecar_manager,
        )
        content = file_path.read_text(encoding="utf-8", errors="replace")
        sidecar = _sidecar_payload(file_path)
        result = pipeline.run(
            content,
            source=str(file_path),
            sidecar_payload=sidecar,
            compute_embeddings=True,
        )
        nodes = []
        for unit, vector in zip(result.units, result.embeddings):
            node = node_class(
                text=unit.content,
                id_=unit.uid,
                metadata={"relations": list(unit.relations), "meta": unit.metadata},
            )
            node.embedding = vector
            nodes.append(node)
        with lock:
            if hasattr(vector_store, "delete_by_metadata"):
                source_key = (
                    str(file_path.relative_to(workspace_root))
                    if workspace_root
                    else str(file_path)
                )
                vector_store.delete_by_metadata("source", source_key)
            if nodes:
                vector_store.add(nodes)
        return file_path, len(result.units)

    max_workers = max(1, min(max_workers, os.cpu_count() or max_workers))
    tqdm_factory = _get_tqdm()
    progress = tqdm_factory(
        total=len(paths),
        disable=not show_progress or not sys.stderr.isatty(),
        desc="Ingesting",
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, path) for path in paths]
        for future in as_completed(futures):
            _, units_count = future.result()
            file_count += 1
            vector_count += units_count
            progress.update(1)
    progress.close()
    return IngestStats(files=file_count, vectors=vector_count)


@click.group()
def main() -> None:
    """SMAK: Semantic Mesh Agentic Kernel CLI."""


@main.command()
@click.option("--folder", required=True, type=click.Path(path_type=Path), help="Path to ingest")
@click.option("--index", required=True, help="Target index name (e.g., source_code)")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
@click.option("--workers", default=DEFAULT_MAX_WORKERS, help="Max parallel workers")
def ingest(folder: Path, index: str, config: str, workers: int) -> None:
    """Ingest a folder into the specified vector index."""
    if not folder.exists() or not folder.is_dir():
        raise click.ClickException(f"Folder not found: {folder}")

    try:
        cfg = load_config(config)
    except Exception as exc:  # pragma: no cover - click prints the exception
        raise click.ClickException(f"Error loading config: {exc}") from exc

    click.echo(f"Starting ingestion for '{folder}' -> Index: '{index}'...")
    try:
        stats = _ingest_folder(
            folder,
            index,
            cfg,
            max_workers=workers,
            show_progress=True,
            workspace_root=Path(config).resolve().parent,
        )
    except IntegrityError as exc:
        raise click.ClickException(f"Sidecar integrity error: {exc}") from exc
    except Exception as exc:
        raise click.ClickException(f"Ingestion failed: {exc}") from exc

    click.echo("Ingestion Complete!")
    click.echo(f"   - Processed Files: {stats.files}")
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
def search(path: Path, config_path: str) -> None:
    """Print symbols for a file to populate sidecar metadata."""
    if not path.exists():
        raise click.ClickException(f"Path not found: {path}")
    if not path.is_file():
        raise click.ClickException(f"Path must be a file: {path}")
    workspace_root = Path(config_path).resolve().parent if Path(config_path).exists() else None
    symbols = _symbols_for_path(path, root_path=workspace_root)
    if not symbols:
        click.echo("No symbols found.")
        return
    for symbol in symbols:
        click.echo(symbol)


@main.group()
def sidecar() -> None:
    """Manage sidecar files."""


@sidecar.command("init")
@click.argument("file_path", type=click.Path(path_type=Path))
@click.option(
    "--config",
    "config_path",
    default="workspace_config.yaml",
    help="Path to workspace config",
)
def sidecar_init(file_path: Path, config_path: str) -> None:
    """Generate a sidecar skeleton for a source file."""
    if not file_path.exists() or not file_path.is_file():
        raise click.ClickException(f"Path must be a file: {file_path}")
    workspace_root = Path(config_path).resolve().parent if Path(config_path).exists() else None
    parser = _parser_for_path(file_path, root_path=workspace_root)
    units = parser.parse(
        file_path.read_text(encoding="utf-8", errors="replace"),
        source=str(file_path),
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
    output = file_path.with_name(f"{file_path.name}.sidecar.yaml")
    output.write_text(payload, encoding="utf-8")
    click.echo(f"Wrote sidecar template to {output}")


@main.command("doctor")
@click.option("--path", "target_path", default=".", type=click.Path(path_type=Path))
def doctor(target_path: Path) -> None:
    """Check sidecar integrity across a path."""
    issues: list[str] = []
    root = target_path if target_path.is_dir() else target_path.parent
    for sidecar in root.rglob("*.sidecar.yaml"):
        source = sidecar.with_name(sidecar.name.replace(".sidecar.yaml", ""))
        if not source.exists():
            issues.append(f"Orphaned sidecar: {sidecar}")
    for sidecar in root.rglob("*.sidecar.yml"):
        source = sidecar.with_name(sidecar.name.replace(".sidecar.yml", ""))
        if not source.exists():
            issues.append(f"Orphaned sidecar: {sidecar}")
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
    "sidecar",
    "sidecar_init",
    "doctor",
]
