from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from smak.config import SmakConfig
from smak.embedding import (
    EmbeddingProbe,
    InternalNomicEmbedding,
    initialize_embedding_dimensions,
    validate_vector_store_dimension,
)
from smak.ingest.parsers import IssueParser, Parser, PerlParser, PythonParser, SimpleLineParser
from smak.ingest.pipeline import IngestPipeline
from smak.ingest.sidecar import SidecarManager

SIDECAR_SUFFIXES = (".sidecar.yaml", ".sidecar.yml")


@dataclass(frozen=True)
class IngestStats:
    files: int
    vectors: int
    skipped: int


def _load_text_node_class():
    from llama_index.core.schema import TextNode

    return TextNode


def _load_vector_store(index_name: str, config: SmakConfig):
    from smak.storage.faiss_adapter import load_faiss_store

    provider = (config.storage.provider or "faiss").lower()
    if provider != "faiss":
        raise ValueError(f"Unsupported vector store provider: {provider}")
    return load_faiss_store(
        uri=config.storage.uri, collection_name=index_name, dim=config.embedding_dimensions
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
        if path.is_file() and not path.name.endswith(SIDECAR_SUFFIXES):
            yield path


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


class IngestService:
    def __init__(
        self,
        config: SmakConfig,
        vector_store_loader: Callable[[str, SmakConfig], object] | None = None,
    ) -> None:
        self.config = config
        self._vector_store_loader = vector_store_loader or _load_vector_store

    def ingest_folder(
        self,
        folder: Path,
        index: str,
        *,
        max_workers: int = 4,
        workspace_root: Path | None = None,
        incremental: bool = True,
        node_class_loader: Callable[[], type] | None = None,
        embedder_loader: Callable[[], EmbeddingProbe] | None = None,
    ) -> IngestStats:
        embedder = (embedder_loader or InternalNomicEmbedding)()
        config = initialize_embedding_dimensions(self.config, embedder)
        vector_store = self._vector_store_loader(index, config)
        validate_vector_store_dimension(vector_store, config.embedding_dimensions)
        node_class = (node_class_loader or _load_text_node_class)()
        sidecar_manager = SidecarManager()

        paths = list(_iter_source_files(folder))
        lock = threading.Lock()
        file_count = vector_count = skipped_count = 0

        def process(file_path: Path) -> tuple[int, bool]:
            parser = _parser_for_path(file_path, root_path=workspace_root)
            content = file_path.read_text(encoding="utf-8", errors="replace")
            parsed_units = parser.parse(content, source=str(file_path))
            source_mtime = _source_mtime(file_path)
            if (
                incremental
                and parsed_units
                and _unit_up_to_date(vector_store, parsed_units[0].uid, source_mtime)
            ):
                return 0, True

            pipeline = IngestPipeline(
                parser=parser, embedder=embedder, sidecar_manager=sidecar_manager
            )
            result = pipeline.run(
                content,
                source=str(file_path),
                sidecar_payload=_sidecar_payload(file_path),
                compute_embeddings=True,
            )
            nodes = []
            for unit, vector in zip(result.units, result.embeddings):
                node = node_class(
                    text=unit.content,
                    id_=unit.uid,
                    metadata={
                        "relations": list(unit.relations),
                        "meta": unit.metadata,
                        "source": _source_key(file_path, workspace_root),
                        "source_mtime": source_mtime,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                node.embedding = vector
                nodes.append(node)
            with lock:
                if hasattr(vector_store, "delete_by_metadata"):
                    vector_store.delete_by_metadata(
                        "source", _source_key(file_path, workspace_root)
                    )
                if nodes:
                    vector_store.add(nodes)
            return len(result.units), False

        max_workers = max(1, min(max_workers, os.cpu_count() or max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process, path) for path in paths]
            for future in as_completed(futures):
                units_count, skipped = future.result()
                skipped_count += int(skipped)
                file_count += 0 if skipped else 1
                vector_count += units_count
        return IngestStats(files=file_count, vectors=vector_count, skipped=skipped_count)
