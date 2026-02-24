from __future__ import annotations

from collections.abc import Callable
from typing import Any

from smak.config import SmakConfig
from smak.embedding import InternalNomicEmbedding


class QueryService:
    def __init__(
        self,
        vector_store: object,
        config: SmakConfig,
        vector_store_loader: Callable[[str, str, SmakConfig], object],
        embedder: object | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.config = config
        self.vector_store_loader = vector_store_loader
        self.embedder = embedder or InternalNomicEmbedding()
        self._vector_store_cache: dict[str, object] = {}

    def _get_payload_globally(self, uid: str) -> dict[str, Any] | None:
        payload = self.vector_store.get_by_id(uid)
        if isinstance(payload, dict):
            return payload

        for index in self.config.indices:
            if index.name in self._vector_store_cache:
                store = self._vector_store_cache[index.name]
            else:
                index_uri = index.uri or f"./smak_data/{index.name}"
                store = self.vector_store_loader(index.name, index_uri, self.config)
                self._vector_store_cache[index.name] = store
            payload = store.get_by_id(uid)
            if isinstance(payload, dict):
                return payload
        return None

    def search(self, text: str, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
        query_vector = self.embedder.get_text_embedding(text)
        semantic_hits = self.vector_store.search(query_vector, top_k=max(top_k, 1))

        hits: list[dict[str, Any]] = []
        related_context: list[dict[str, Any]] = []
        seen_related: set[str] = set()

        for hit in semantic_hits:
            if not isinstance(hit, dict):
                continue
            uid = str(hit.get("uid", ""))
            metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
            hits.append(
                {
                    "uid": uid,
                    "match_type": "semantic",
                    "score": hit.get("score"),
                    "content": hit.get("content"),
                }
            )
            relations = metadata.get("relations", [])
            if isinstance(relations, str):
                relations = [relations]
            if not isinstance(relations, list):
                continue
            for target_uid in relations:
                target_uid = str(target_uid)
                if not target_uid or target_uid in seen_related:
                    continue
                seen_related.add(target_uid)
                related_payload = self._get_payload_globally(target_uid)
                if not isinstance(related_payload, dict):
                    continue
                related_context.append(
                    {
                        "uid": str(related_payload.get("uid", target_uid)),
                        "match_type": "relation",
                        "source_hit": uid,
                        "content": related_payload.get("content"),
                    }
                )

        return {"hits": hits, "related_context": related_context}
