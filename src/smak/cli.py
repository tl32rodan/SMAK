"""Command line interface for SMAK."""

from __future__ import annotations

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


@dataclass(frozen=True)
class IngestStats:
    """Statistics returned after ingesting a folder."""

    files: int
    vectors: int


def _load_registry(base_path: str):
    try:
        from faiss_storage_lib.engine.registry import IndexRegistry
    except ImportError as exc:  # pragma: no cover - guard for missing dependency
        raise click.ClickException(
            "faiss-storage-lib is required for ingest. Install it to continue."
        ) from exc
    return IndexRegistry(base_path)


def _default_config_template() -> str:
    return "\n".join(
        [
            "indices:",
            "  - name: source_code",
            "    description: Source code files",
            "llm:",
            "  provider: openai",
            "  model: ",
            "storage:",
            "  base_path: vault",
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
) -> IngestStats:
    loader = registry_loader or _load_registry
    registry = loader(config.storage.base_path)
    embedder = SimpleEmbedder()
    sidecar_manager = SidecarManager()
    adapter = VectorAdapter(registry=registry, embedder=embedder)

    file_count = 0
    vector_count = 0
    for file_path in _iter_source_files(folder):
        parser = _parser_for_path(file_path)
        pipeline = IngestPipeline(
            parser=parser,
            embedder=embedder,
            sidecar_manager=sidecar_manager,
        )
        content = file_path.read_text(encoding="utf-8", errors="replace")
        sidecar = _sidecar_payload(file_path)
        result = pipeline.run(content, source=str(file_path), sidecar_payload=sidecar)
        adapter.save(index, result.units)
        file_count += 1
        vector_count += len(result.units)
    return IngestStats(files=file_count, vectors=vector_count)


@click.group()
def main() -> None:
    """SMAK: Semantic Mesh Agentic Kernel CLI."""


@main.command()
@click.option("--folder", required=True, type=click.Path(path_type=Path), help="Path to ingest")
@click.option("--index", required=True, help="Target index name (e.g., source_code)")
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
def ingest(folder: Path, index: str, config: str) -> None:
    """Ingest a folder into the specified vector index."""
    if not folder.exists() or not folder.is_dir():
        raise click.ClickException(f"Folder not found: {folder}")

    try:
        cfg = load_config(config)
    except Exception as exc:  # pragma: no cover - click prints the exception
        raise click.ClickException(f"Error loading config: {exc}") from exc

    click.echo(f"🚀 Starting ingestion for '{folder}' -> Index: '{index}'...")
    try:
        stats = _ingest_folder(folder, index, cfg)
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
    try:
        from examples.demo_server import demo
    except ImportError as exc:
        raise click.ClickException("Demo server module not available.") from exc
    click.echo(f"🤖 Launching SMAK Agent Server on port {port}...")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


__all__ = [
    "IngestStats",
    "ingest",
    "init",
    "main",
    "server",
]
