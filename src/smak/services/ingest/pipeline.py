"""Ingest pipeline combining parsing and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smak.core.domain import KnowledgeUnit
from smak.parsers import Parser
from smak.utils.embedding import EmbeddingProbe, InternalEmbedding

Embedder = EmbeddingProbe


@dataclass
class IngestResult:
    units: list[KnowledgeUnit]
    embeddings: list[list[float]]
    metadata: dict[str, Any]


@dataclass
class IngestPipeline:
    parser: Parser
    embedder: Embedder | None = None

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = InternalEmbedding()

    def run(
        self,
        content: str,
        *,
        source: str | None = None,
        compute_embeddings: bool = False,
        env: dict[str, str] | None = None,
    ) -> IngestResult:
        units = self.parser.parse(content, source=source, env=env)
        embeddings = self._embed_units(units) if compute_embeddings else []
        return IngestResult(units=units, embeddings=embeddings, metadata={})

    def _embed_units(self, units: list[KnowledgeUnit]) -> list[list[float]]:
        texts = [unit.content for unit in units]
        embedder = self.embedder
        if embedder is None:
            return []
        if hasattr(embedder, "embed_documents"):
            return embedder.embed_documents(texts)
        if hasattr(embedder, "embed"):
            return embedder.embed(texts)
        if hasattr(embedder, "get_text_embedding_batch"):
            return embedder.get_text_embedding_batch(list(texts))
        raise AttributeError("Embedder does not support embedding documents.")


__all__ = ["Embedder", "IngestPipeline", "IngestResult"]
