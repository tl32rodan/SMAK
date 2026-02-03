"""Storage adapters for SMAK."""

from .milvus import MilvusLiteVectorSearchIndex, MilvusLiteVectorStore

__all__ = ["MilvusLiteVectorSearchIndex", "MilvusLiteVectorStore"]
