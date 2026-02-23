"""Thread-local FaissEngine wrapper.

Ensures each thread uses an isolated engine instance (and therefore an isolated
SQLite connection owned by that thread).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from faiss_storage_lib.engine.faiss_engine import FaissEngine as _BaseFaissEngine

logger = logging.getLogger(__name__)


@dataclass
class FaissEngine:
    """Proxy Faiss engine with per-thread engine instances."""

    path: str
    dim: int
    _factory: Callable[[str, int], Any] = _BaseFaissEngine
    _local: threading.local = field(default_factory=threading.local, init=False)

    def __post_init__(self) -> None:
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def _engine(self) -> Any:
        engine = getattr(self._local, "engine", None)
        if engine is None:
            engine = self._factory(self.path, self.dim)
            self._local.engine = engine
            self._apply_sqlite_pragma(engine)
        return engine

    @staticmethod
    def _apply_sqlite_pragma(engine: Any) -> None:
        connection = (
            getattr(engine, "conn", None)
            or getattr(engine, "connection", None)
            or getattr(engine, "_conn", None)
            or getattr(engine, "_connection", None)
        )
        if connection is None:
            return
        try:
            connection.execute("PRAGMA journal_mode=NA;")
        except sqlite3.DatabaseError:
            logger.debug("Unable to apply PRAGMA journal_mode=NA on engine connection.")

    def add(self, docs: list[Any]) -> None:
        self._engine().add(docs)

    def persist(self) -> None:
        persist = getattr(self._engine(), "persist", None)
        if callable(persist):
            persist()

    def search(self, embedding: list[float], top_k: int) -> list[Any]:
        return self._engine().search(embedding, top_k)

    def get_by_id(self, uid: str) -> Any | None:
        return self._engine().get_by_id(uid)

    def delete_by_metadata(self, key: str, value: Any) -> None:
        delete = getattr(self._engine(), "delete_by_metadata", None)
        if callable(delete):
            delete(key, value)

    def count(self) -> int | None:
        count = getattr(self._engine(), "count", None)
        if callable(count):
            return int(count())
        docs = getattr(self._engine(), "docs", None)
        if isinstance(docs, list):
            return len(docs)
        return None

    @property
    def last_update(self) -> Any:
        return getattr(self._engine(), "last_update", None)

    @last_update.setter
    def last_update(self, value: Any) -> None:
        setattr(self._engine(), "last_update", value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine(), name)
