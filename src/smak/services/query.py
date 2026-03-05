from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from smak.config import IndexConfig, SmakConfig
from smak.services.relation_resolver import SidecarRelationResolver
from smak.utils.embedding import InternalNomicEmbedding


class QueryService:
    def __init__(
        self,
        vector_store: object,
        config: SmakConfig,
        vector_store_loader: Callable[[IndexConfig, SmakConfig], object],
        index_config: IndexConfig,
        embedder: object | None = None,
        relation_resolver: SidecarRelationResolver | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.config = config
        self.vector_store_loader = vector_store_loader
        self.index_config = index_config
        self.embedder = embedder or InternalNomicEmbedding()
        self.relation_resolver = relation_resolver or SidecarRelationResolver()
        self._vector_store_cache: dict[str, object] = {}

    def _build_resolver_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of *metadata* with 'source' made absolute for sidecar lookup.

        The stored source path is relative to the index root. The relation resolver
        needs an absolute path to locate the .sidecar.yaml file on disk.
        """
        resolver_metadata = dict(metadata)
        if "source" in resolver_metadata:
            resolver_metadata["source"] = str(
                Path(self.index_config.path) / resolver_metadata["source"]
            )
        return resolver_metadata

    def _normalize_hit_uid(self, uid: str, metadata: dict[str, Any]) -> str:
        """Normalize hit UID path segment to absolute when metadata source is relative."""

        source = metadata.get("source")
        if not isinstance(source, str) or not source:
            return uid
        if "::" not in uid:
            return uid

        uid_path, symbol = uid.split("::", 1)
        source_path = Path(source)
        if source_path.is_absolute():
            normalized_source = str(source_path)
        else:
            normalized_source = str((Path(self.index_config.path) / source_path).resolve())

        if uid_path == source or uid_path == str(source_path):
            return f"{normalized_source}::{symbol}"
        return uid

    def _get_payload_globally(self, uid: str) -> dict[str, Any] | None:
        payload = self.vector_store.get_by_id(uid)
        if isinstance(payload, dict):
            return payload

        for index in self.config.indices:
            if index.name in self._vector_store_cache:
                store = self._vector_store_cache[index.name]
            else:
                store = self.vector_store_loader(index, self.config)
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
            raw_uid = str(hit.get("uid", ""))
            metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
            uid = self._normalize_hit_uid(raw_uid, metadata)
            hits.append(
                {
                    "uid": uid,
                    "exact_uid": uid,
                    "match_type": "semantic",
                    "score": hit.get("score"),
                    "content": hit.get("content"),
                    "exact_relative_path": metadata.get("source"),
                }
            )
            relations = self.relation_resolver.resolve(uid, self._build_resolver_metadata(metadata))
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
