"""Factory functions for assembling SMAK services with their dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from smak.config import EmbeddingConfig, IndexConfig, SmakConfig
from smak.services import DoctorService, QueryService, SidecarService
from smak.services.relation_resolver import SidecarRelationResolver
from smak.sidecar import is_sidecar_file
from smak.sidecar.store import YAMLSidecarStore
from smak.utils.embedding import (
    InternalEmbedding,
    initialize_embedding_dimensions,
    validate_vector_store_dimension,
)


def _load_vector_store_for_index(index_config: object, config: object) -> object:
    from smak.storage.faiss_adapter import load_vector_store_for_index

    return load_vector_store_for_index(index_config, config)


def load_and_validate_vector_store(
    index_config: IndexConfig,
    config: SmakConfig,
) -> object:
    """Load a vector store and validate its dimensions match the config."""
    vector_store = _load_vector_store_for_index(index_config, config)
    validate_vector_store_dimension(vector_store, config.embedding_dimensions)
    return vector_store


def init_config(
    config: SmakConfig,
    embedding_config: EmbeddingConfig | None = None,
) -> SmakConfig:
    """Initialise embedding dimensions on a config object."""
    return initialize_embedding_dimensions(
        config,
        InternalEmbedding(embedding_config=embedding_config),
    )


def create_query_service(
    vector_store: object,
    config: SmakConfig,
    index_config: IndexConfig,
    embedding_config: EmbeddingConfig | None = None,
) -> QueryService:
    """Create a QueryService with standard sidecar relation resolution."""
    sidecar_store = YAMLSidecarStore()
    return QueryService(
        vector_store=vector_store,
        config=config,
        index_config=index_config,
        vector_store_loader=_load_vector_store_for_index,
        relation_resolver=SidecarRelationResolver(sidecar_store, env=config.env),
        embedder=InternalEmbedding(embedding_config=embedding_config),
    )


def create_sidecar_service(env: dict[str, str] | None = None) -> SidecarService:
    """Create a SidecarService with the default YAML store."""
    return SidecarService(sidecar_store=YAMLSidecarStore(), env=env)


def create_doctor_service(
    config: SmakConfig,
    *,
    on_missing_index: Callable[[str], None] | None = None,
) -> DoctorService:
    """Create a DoctorService with standard vector store loading."""

    def _load_store(index_name: str) -> object:
        index_config = config.get_index(index_name)
        if index_config is None:
            if on_missing_index:
                on_missing_index(index_name)
            raise ValueError(f"Index '{index_name}' not found in configuration.")
        return load_and_validate_vector_store(index_config, config)

    return DoctorService(config=config, vector_store_loader=_load_store)


def sidecar_skip_file(path: Path) -> bool:
    """File filter predicate that skips sidecar files during iteration."""
    return is_sidecar_file(path)


def on_ghost_source(source: str, folders: list[Path]) -> None:
    """Callback to clean up sidecar files for ghost (deleted) sources."""
    sidecar_store = YAMLSidecarStore()
    for folder in folders:
        sidecar_store.delete_sidecar_for_source(folder / source)
