"""Ingest pipeline combining parsing, sidecar metadata, and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from smak.bridge.models import InternalNomicEmbedding
from smak.core.domain import KnowledgeUnit
from smak.ingest.parsers import Parser
from smak.ingest.sidecar import IntegrityError, SidecarManager


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def get_text_embeddings(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class IngestResult:
    """Output of the ingest pipeline."""

    units: list[KnowledgeUnit]
    embeddings: list[list[float]]
    metadata: dict[str, Any]


@dataclass
class IngestPipeline:
    """Coordinate ingest steps for content."""

    parser: Parser
    embedder: Embedder | None = None
    sidecar_manager: SidecarManager | None = None

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = InternalNomicEmbedding()
        if self.sidecar_manager is None:
            self.sidecar_manager = SidecarManager()

    def run(
        self,
        content: str,
        *,
        source: str | None = None,
        sidecar_payload: str | None = None,
        compute_embeddings: bool = False,
    ) -> IngestResult:
        units = self.parser.parse(content, source=source)
        sidecar_manager = self.sidecar_manager
        if sidecar_manager is None:
            sidecar_manager = SidecarManager()
        metadata = sidecar_manager.load(sidecar_payload)
        symbols = [unit.metadata.get("symbol") for unit in units if unit.metadata.get("symbol")]
        sidecar_manager.validate(symbols, metadata)
        enriched_units = sidecar_manager.apply(units, metadata)
        embeddings = self._embed_units(enriched_units) if compute_embeddings else []
        return IngestResult(units=enriched_units, embeddings=embeddings, metadata=metadata)

    def _embed_units(self, units: list[KnowledgeUnit]) -> list[list[float]]:
        texts = [unit.content for unit in units]
        embedder = self.embedder
        if embedder is None:
            return []
        if hasattr(embedder, "embed_documents"):
            return embedder.embed_documents(texts)
        if hasattr(embedder, "embed"):
            return embedder.embed(texts)
        if hasattr(embedder, "get_text_embeddings"):
            return embedder.get_text_embeddings(list(texts))
        raise AttributeError("Embedder does not support embedding documents.")


__all__ = ["Embedder", "IngestPipeline", "IngestResult", "IntegrityError"]
