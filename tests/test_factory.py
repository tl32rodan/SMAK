"""Tests for smak.factory service-construction functions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from smak.config import IndexConfig, SmakConfig
from smak.factory import (
    create_query_service,
    create_sidecar_service,
    on_ghost_source,
    sidecar_skip_file,
)
from smak.services.query import QueryService
from smak.services.sidecar import SidecarService


class TestFactory(unittest.TestCase):
    @patch("smak.services.query.InternalNomicEmbedding")
    def test_create_query_service_returns_query_service(self, _mock_embedder: object) -> None:
        store = SimpleNamespace(
            get_by_id=lambda uid: None,
            search=lambda v, top_k=5: [],
        )
        config = SmakConfig(indices=[IndexConfig(name="src", description="d")])
        service = create_query_service(store, config, config.indices[0])
        self.assertIsInstance(service, QueryService)

    def test_create_sidecar_service_returns_sidecar_service(self) -> None:
        service = create_sidecar_service()
        self.assertIsInstance(service, SidecarService)

    def test_sidecar_skip_file_filters_sidecar_files(self) -> None:
        self.assertTrue(sidecar_skip_file(Path("foo/.bar.py.sidecar.yaml")))
        self.assertFalse(sidecar_skip_file(Path("foo/bar.py")))

    def test_on_ghost_source_deletes_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sidecar = root / ".ghost.py.sidecar.yaml"
            sidecar.write_text("symbols: []\n", encoding="utf-8")
            on_ghost_source("ghost.py", [root])
            self.assertFalse(sidecar.exists())


if __name__ == "__main__":
    unittest.main()
