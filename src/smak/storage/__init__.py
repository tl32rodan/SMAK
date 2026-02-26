"""Storage adapters for SMAK."""

from .faiss_adapter import FaissVectorSearchIndex, FaissVectorStore, load_faiss_store
from .vector_adapter import VectorAdapter, VectorDocument

__all__ = [
    "FaissVectorSearchIndex",
    "FaissVectorStore",
    "VectorAdapter",
    "VectorDocument",
    "load_faiss_store",
]
