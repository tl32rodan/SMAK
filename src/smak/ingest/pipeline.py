"""Ingest pipeline combining parsing, sidecar metadata, and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smak.core.domain import KnowledgeUnit
from smak.ingest.embedder import SimpleEmbedder
from smak.ingest.parsers import Parser
from smak.ingest.sidecar import SidecarLoader


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
    sidecar_loader: SidecarLoader

    def run(
        self,
        content: str,
        *,
        source: str | None = None,
        sidecar_payload: str | None = None,
    ) -> IngestResult:
        units = self.parser.parse(content, source=source)
        embeddings = self.embedder.embed([unit.content for unit in units])
        metadata = self.sidecar_loader.load(sidecar_payload)
        return IngestResult(units=units, embeddings=embeddings, metadata=metadata)
