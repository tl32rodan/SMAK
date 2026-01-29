"""Ingest pipeline combining parsing, sidecar metadata, and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smak.core.domain import KnowledgeUnit
from smak.ingest.embedder import SimpleEmbedder
from smak.ingest.parsers import Parser
from smak.ingest.sidecar import IntegrityError, SidecarManager


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
    embedder: SimpleEmbedder
    sidecar_manager: SidecarManager

    def run(
        self,
        content: str,
        *,
        source: str | None = None,
        sidecar_payload: str | None = None,
    ) -> IngestResult:
        units = self.parser.parse(content, source=source)
        embeddings = self.embedder.embed([unit.content for unit in units])
        metadata = self.sidecar_manager.load(sidecar_payload)
        symbols = [unit.metadata.get("symbol") for unit in units if unit.metadata.get("symbol")]
        self.sidecar_manager.validate(symbols, metadata)
        enriched_units = self.sidecar_manager.apply(units, metadata)
        return IngestResult(units=enriched_units, embeddings=embeddings, metadata=metadata)


__all__ = ["IngestPipeline", "IngestResult", "IntegrityError"]
