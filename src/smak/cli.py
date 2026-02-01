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
from smak.db.adapter import VectorAdapter
from smak.ingest.embedder import SimpleEmbedder
from smak.ingest.parsers import IssueParser, Parser, PerlParser, PythonParser, SimpleLineParser
from smak.ingest.pipeline import IngestPipeline, IntegrityError
from smak.ingest.sidecar import SidecarManager

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


def _load_registry(base_path: str):
    spec = importlib.util.find_spec("faiss_storage_lib.engine.registry")
    if spec is None:  # pragma: no cover - guard for missing dependency
        raise click.ClickException(
            "Critical dependency 'faiss-storage-lib' not found. "
            "Did you run pip install -e ../faiss-storage-lib?"
        )
    module = importlib.import_module("faiss_storage_lib.engine.registry")
    registry_class = getattr(module, "IndexRegistry")
    return registry_class(base_path)


def _default_config_template() -> str:
    return "\n".join(
        [
            "# SMAK Workspace Configuration",
            "",
            "storage:",
            "  base_path: ./data/faiss_indexes",
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
            "llm:",
            "  provider: openai",
            "  model: gpt-4o",
            "  temperature: 0.0",
            "  # api_base: http://localhost:11434/v1",
            "",
            "embedding_dimensions: 3",
            "",
        ]
    )


def _parser_for_path(path: Path) -> Parser:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return PythonParser()
    if suffix in {".pl", ".pm"}:
        return PerlParser()
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


def _ingest_folder(
    folder: Path,
    index: str,
    config: SmakConfig,
    registry_loader: Callable[[str], object] | None = None,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    show_progress: bool = False,
) -> IngestStats:
    loader = registry_loader or _load_registry
    registry = loader(config.storage.base_path)
    embedder = SimpleEmbedder()
    sidecar_manager = SidecarManager()
    adapter = VectorAdapter(registry=registry, embedder=embedder)

    paths = list(_iter_source_files(folder))
    file_count = 0
    vector_count = 0

    lock = threading.Lock()

    def process_file(file_path: Path) -> tuple[Path, int]:
        parser = _parser_for_path(file_path)
        pipeline = IngestPipeline(
            parser=parser,
            embedder=embedder,
            sidecar_manager=sidecar_manager,
        )
        content = file_path.read_text(encoding="utf-8", errors="replace")
        sidecar = _sidecar_payload(file_path)
        result = pipeline.run(content, source=str(file_path), sidecar_payload=sidecar)
        with lock:
            adapter.save(index, result.units)
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

    click.echo(f"🚀 Starting ingestion for '{folder}' -> Index: '{index}'...")
    try:
        stats = _ingest_folder(folder, index, cfg, max_workers=workers, show_progress=True)
    except IntegrityError as exc:
        raise click.ClickException(f"Sidecar integrity error: {exc}") from exc
    except Exception as exc:
        raise click.ClickException(f"Ingestion failed: {exc}") from exc

    click.echo("✅ Ingestion Complete!")
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
    click.echo(f"📝 Wrote workspace config to {target}")


@main.command()
@click.option("--port", default=7860, help="Port to run the server on")
def server(port: int) -> None:
    """Launch the Agent Demo Server."""
    spec = importlib.util.find_spec("examples.demo_server")
    if spec is None:
        raise click.ClickException("Demo server module not available.")
    module = importlib.import_module("examples.demo_server")
    demo = getattr(module, "demo")
    click.echo(f"🤖 Launching SMAK Agent Server on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


__all__ = [
    "IngestStats",
    "ingest",
    "init",
    "main",
    "server",
]
