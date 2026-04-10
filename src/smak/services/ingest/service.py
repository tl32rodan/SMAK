from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from smak.parsers import get_parser_for_path
from smak.services.ingest.pipeline import IngestPipeline
from smak.utils.embedding import EmbeddingProbe, InternalNomicEmbedding


@dataclass(frozen=True)
class IngestStats:
    files: int
    vectors: int
    skipped: int
    deleted: int = 0


def _load_text_node_class():
    from llama_index.core.schema import TextNode

    return TextNode


def _read_text_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _iter_source_files(
    folder: Path,
    *,
    follow_symlinks: bool = True,
    skip_file: Callable[[Path], bool] | None = None,
) -> Iterable[Path]:
    for root, _, files in os.walk(folder, followlinks=follow_symlinks):
        root_path = Path(root)
        for filename in files:
            path = root_path / filename
            if skip_file is None or not skip_file(path):
                yield path


def _source_key(path: Path, root_path: Path | None = None) -> str:
    if root_path is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(root_path.resolve()))
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


class IngestService:
    def __init__(self, vector_store: object) -> None:
        self.vector_store = vector_store

    def ingest_paths(
        self,
        paths: list[Path],
        *,
        max_workers: int = 4,
        incremental: bool = True,
        node_class_loader: Callable[[], type] | None = None,
        embedder_loader: Callable[[], EmbeddingProbe] | None = None,
        follow_symlinks: bool = True,
        sync: bool = False,
        skip_file: Callable[[Path], bool] | None = None,
        on_ghost_source: Callable[[str, list[Path]], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> IngestStats:
        """Ingest content from multiple paths into the vector store.

        Each entry in *paths* may be a directory (walked recursively) or a
        single file (ingested directly).

        Handles sync correctly across all paths: ghost detection is deferred
        until all paths have been visited, so sources from one path are
        not mistakenly pruned while another path is being processed.
        """
        embedder = (embedder_loader or InternalNomicEmbedding)()
        node_class = (node_class_loader or _load_text_node_class)()
        vector_store = self.vector_store
        tracked_sources = (
            vector_store.get_all_tracked_sources() if sync else {}
        )
        all_visited_sources: set[str] = set()

        # Collect (file_path, root_path) pairs from all paths
        all_files: list[tuple[Path, Path]] = []
        for source_path in paths:
            if source_path.is_file():
                if skip_file is None or not skip_file(source_path):
                    all_files.append((source_path, source_path.parent))
            else:
                for file_path in _iter_source_files(
                    source_path, follow_symlinks=follow_symlinks, skip_file=skip_file
                ):
                    all_files.append((file_path, source_path))

        lock = threading.Lock()
        file_count = vector_count = skipped_count = 0

        def process(file_path: Path, root_folder: Path) -> tuple[int, bool]:
            source_key = _source_key(file_path, root_folder)
            with lock:
                all_visited_sources.add(source_key)
            parser = get_parser_for_path(file_path)
            content = _read_text_with_fallback(file_path)
            parsed_units = parser.parse(content, source=str(file_path), env=env)
            source_mtime = _source_mtime(file_path)
            if (
                incremental
                and parsed_units
                and _unit_up_to_date(
                    vector_store,
                    parsed_units[0].uid,
                    source_mtime,
                )
            ):
                return 0, True

            pipeline = IngestPipeline(parser=parser, embedder=embedder)
            result = pipeline.run(
                content,
                source=str(file_path),
                compute_embeddings=True,
                env=env,
            )
            nodes = []
            for unit, vector in zip(result.units, result.embeddings):
                node = node_class(
                    text=unit.content,
                    id_=unit.uid,
                    metadata={
                        "relations": list(unit.relations),
                        "meta": unit.metadata,
                        "source": source_key,
                        "source_mtime": source_mtime,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                node.embedding = vector
                nodes.append(node)
            with lock:
                if hasattr(vector_store, "delete_by_metadata"):
                    vector_store.delete_by_metadata("source", source_key)
                if nodes:
                    vector_store.add(nodes)
            return len(result.units), False

        max_workers = max(1, min(max_workers, os.cpu_count() or max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process, fp, rf) for fp, rf in all_files]
            for future in as_completed(futures):
                units_count, skipped = future.result()
                skipped_count += int(skipped)
                file_count += 0 if skipped else 1
                vector_count += units_count

        deleted_count = 0
        if sync:
            ghost_sources = set(tracked_sources.keys()) - all_visited_sources
            for ghost_source in ghost_sources:
                vector_store.delete_by_ids(tracked_sources[ghost_source])
                if on_ghost_source:
                    on_ghost_source(ghost_source, paths)
                deleted_count += 1

        return IngestStats(
            files=file_count,
            vectors=vector_count,
            skipped=skipped_count,
            deleted=deleted_count,
        )
