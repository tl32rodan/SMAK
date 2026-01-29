"""Embedding utilities for SMAK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class SimpleEmbedder:
    """Create deterministic embedding vectors based on string statistics."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Compatibility wrapper for document embedding."""

        return self.embed(texts)

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        length = len(text)
        if length == 0:
            return [0.0, 0.0, 0.0]
        total = sum(ord(char) for char in text)
        average = total / length
        return [float(length), float(total), float(average)]
