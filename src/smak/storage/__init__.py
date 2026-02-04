"""Storage adapters for SMAK."""

from .faiss_adapter import (
    FaissVectorSearchIndex,
    FaissVectorStore,
    load_faiss_store,
)

__all__ = ["FaissVectorSearchIndex", "FaissVectorStore", "load_faiss_store"]
