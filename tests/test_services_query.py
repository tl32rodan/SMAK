from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from smak.config import IndexConfig, SmakConfig
from smak.services.query import QueryService


class DummyEmbedder:
    def get_text_embedding(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class TestQueryService(unittest.TestCase):
    def test_query_service_expands_one_hop_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "main.py"
            source.write_text("def a():\n    pass\n", encoding="utf-8")
            source.with_name("main.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: func_A\n"
                "    relations:\n"
                "      - issue_12\n",
                encoding="utf-8",
            )
            store = SimpleNamespace(
                search=lambda vector, top_k=5: [
                    {
                        "uid": "func_A",
                        "score": 0.9,
                        "content": "A",
                        "metadata": {"source": str(source)},
                    }
                ],
                get_by_id=lambda uid: {"uid": uid, "content": "Issue body"},
            )
            config = SmakConfig(indices=[IndexConfig(name="source_code", description="source")])
            payload = QueryService(
                store,
                config=config,
                workspace_root=Path(tmp_dir),
                vector_store_loader=lambda index_config, cfg: store,
                embedder=DummyEmbedder(),
            ).search("query", top_k=1)
            self.assertEqual(payload["hits"][0]["match_type"], "semantic")
            self.assertEqual(payload["related_context"][0]["source_hit"], "func_A")
            self.assertEqual(payload["hits"][0]["content"], "A")

    def test_query_service_resolves_relations_across_indices_without_extra_search(self) -> None:
        class PrimaryStore:
            def __init__(self) -> None:
                self.search_calls = 0
                self.get_by_id_calls: list[str] = []

            def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
                self.search_calls += 1
                return [
                    {
                        "uid": "func_A",
                        "score": 0.95,
                        "content": "def A(): pass",
                        "metadata": {"source": "src/a.py"},
                    }
                ]

            def get_by_id(self, uid: str) -> dict | None:
                self.get_by_id_calls.append(uid)
                return None

        class SecondaryStore:
            def __init__(self) -> None:
                self.get_by_id_calls: list[str] = []

            def get_by_id(self, uid: str) -> dict | None:
                self.get_by_id_calls.append(uid)
                if uid == "issue:42":
                    return {"uid": uid, "content": "Fix login bug"}
                return None

        primary = PrimaryStore()
        secondary = SecondaryStore()
        loader_calls: list[str] = []

        def loader(index_config: IndexConfig, config: SmakConfig) -> object:
            loader_calls.append(index_config.name)
            if index_config.name == "issues":
                return secondary
            return primary

        config = SmakConfig(
            indices=[
                IndexConfig(name="source_code", description="source"),
                IndexConfig(name="issues", description="issues"),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "src" / "a.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("def a():\n    pass\n", encoding="utf-8")
            source.with_name("a.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: func_A\n"
                "    relations:\n"
                "      - issue:42\n",
                encoding="utf-8",
            )
            payload = QueryService(
                primary,
                config=config,
                workspace_root=Path(tmp_dir),
                vector_store_loader=loader,
                embedder=DummyEmbedder(),
            ).search("query", top_k=1)

        self.assertEqual(primary.search_calls, 1)
        self.assertEqual(loader_calls, ["source_code", "issues"])
        self.assertEqual(payload["hits"][0]["content"], "def A(): pass")
        self.assertEqual(payload["related_context"][0]["content"], "Fix login bug")

    def test_query_service_uses_loader_with_index_config(self) -> None:
        class PrimaryStore:
            def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
                return [{"uid": "func_A", "content": "A", "metadata": {"source": "src/a.py"}}]

            def get_by_id(self, uid: str) -> dict | None:
                return None

        class SecondaryStore:
            def get_by_id(self, uid: str) -> dict | None:
                return {"uid": uid, "content": "Issue"} if uid == "issue:42" else None

        loader_calls: list[str] = []

        def loader(index_config: IndexConfig, config: SmakConfig) -> object:
            loader_calls.append(index_config.name)
            return SecondaryStore() if index_config.name == "issues" else PrimaryStore()

        config = SmakConfig(
            indices=[
                IndexConfig(name="source_code", description="source"),
                IndexConfig(name="issues", description="issues"),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "src" / "a.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("def a():\n    pass\n", encoding="utf-8")
            source.with_name("a.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: func_A\n"
                "    relations:\n"
                "      - issue:42\n",
                encoding="utf-8",
            )
            payload = QueryService(
                PrimaryStore(),
                config=config,
                workspace_root=Path(tmp_dir),
                vector_store_loader=loader,
                embedder=DummyEmbedder(),
            ).search("query", top_k=1)

        self.assertEqual(loader_calls, ["source_code", "issues"])
        self.assertEqual(payload["related_context"][0]["content"], "Issue")


    def test_query_service_matches_sidecar_symbol_name_from_ingest_metadata(self) -> None:
        from smak.ingest.parsers.python import PythonParser

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            source = workspace_root / "src" / "csv_editor.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(
                "class CsvEditor:\n"
                "    def updatecell(self):\n"
                "        return True\n",
                encoding="utf-8",
            )

            parser = PythonParser(root_path=str(workspace_root))
            parsed_units = parser.parse(source.read_text(encoding="utf-8"), source=str(source))
            target_unit = next(
                unit
                for unit in parsed_units
                if unit.metadata.get("symbol") == "CsvEditor.updatecell"
            )

            source.with_name("csv_editor.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: CsvEditor.updatecell\n"
                "    relations:\n"
                "      - issue:from-symbol\n",
                encoding="utf-8",
            )

            store = SimpleNamespace(
                search=lambda vector, top_k=5: [
                    {
                        "uid": target_unit.uid,
                        "content": target_unit.content,
                        "metadata": {
                            "source": target_unit.metadata["source"],
                            "symbol": target_unit.metadata["symbol"],
                        },
                    }
                ],
                get_by_id=lambda related_uid: {"uid": related_uid, "content": related_uid},
            )
            config = SmakConfig(indices=[IndexConfig(name="source_code", description="source")])

            payload = QueryService(
                store,
                config=config,
                workspace_root=workspace_root,
                vector_store_loader=lambda index_config, cfg: store,
                embedder=DummyEmbedder(),
            ).search("query", top_k=1)

            self.assertEqual(payload["related_context"][0]["uid"], "issue:from-symbol")

    def test_query_service_matches_uid_sans_source_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            source = workspace_root / "src" / "csv_editor.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("def updatecell():\n    return True\n", encoding="utf-8")
            source.with_name("csv_editor.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: updatecell\n"
                "    relations:\n"
                "      - issue:suffix-match\n",
                encoding="utf-8",
            )

            store = SimpleNamespace(
                search=lambda vector, top_k=5: [
                    {
                        "uid": "src/csv_editor.py::updatecell",
                        "content": "def updatecell",
                        "metadata": {"source": "src/csv_editor.py"},
                    }
                ],
                get_by_id=lambda related_uid: {"uid": related_uid, "content": related_uid},
            )
            config = SmakConfig(indices=[IndexConfig(name="source_code", description="source")])

            payload = QueryService(
                store,
                config=config,
                workspace_root=workspace_root,
                vector_store_loader=lambda index_config, cfg: store,
                embedder=DummyEmbedder(),
            ).search("query", top_k=1)

            self.assertEqual(payload["related_context"][0]["uid"], "issue:suffix-match")

    def test_query_service_ignores_stale_metadata_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "module.py"
            source.write_text("def a():\n    pass\n", encoding="utf-8")
            source.with_name("module.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: func_A\n"
                "    relations:\n"
                "      - issue:fresh\n",
                encoding="utf-8",
            )

            store = SimpleNamespace(
                search=lambda vector, top_k=5: [
                    {
                        "uid": "func_A",
                        "content": "A",
                        "metadata": {
                            "source": str(source),
                            "relations": ["issue:stale"],
                        },
                    }
                ],
                get_by_id=lambda uid: {"uid": uid, "content": uid},
            )
            config = SmakConfig(indices=[IndexConfig(name="source_code", description="source")])

            payload = QueryService(
                store,
                config=config,
                workspace_root=Path(tmp_dir),
                vector_store_loader=lambda index_config, cfg: store,
                embedder=DummyEmbedder(),
            ).search("query", top_k=1)

            self.assertEqual(payload["related_context"][0]["uid"], "issue:fresh")


if __name__ == "__main__":
    unittest.main()
