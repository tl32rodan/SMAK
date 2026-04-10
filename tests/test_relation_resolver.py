from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smak.services.relation_resolver import SidecarRelationResolver, build_symbol_name_candidates
from smak.sidecar.store import YAMLSidecarStore


class TestRelationResolver(unittest.TestCase):
    def test_build_symbol_name_candidates_from_uid_and_metadata(self) -> None:
        candidates = build_symbol_name_candidates(
            "src/csv_editor.py::CsvEditor.updatecell",
            {"symbol": "CsvEditor.updatecell"},
        )
        self.assertEqual(
            candidates,
            {
                "src/csv_editor.py::CsvEditor.updatecell",
                "CsvEditor.updatecell",
            },
        )

    def test_resolve_supports_relative_source_symbol_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            source = workspace_root / "src" / "csv_editor.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("x=1\n", encoding="utf-8")
            source.with_name(".csv_editor.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: CsvEditor.updatecell\n"
                "    relations:\n"
                "      - issue:1\n",
                encoding="utf-8",
            )

            resolver = SidecarRelationResolver(YAMLSidecarStore())
            relations = resolver.resolve(
                "src/csv_editor.py::CsvEditor.updatecell",
                {"source": str(source), "symbol": "CsvEditor.updatecell"},
            )
            self.assertEqual(relations, ["issue:1"])


    def test_resolve_expands_env_source_for_sidecar_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "src" / "a.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("x=1\n", encoding="utf-8")
            source.with_name(".a.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: login\n"
                "    relations:\n"
                "      - issue:env-test\n",
                encoding="utf-8",
            )

            env = {"TEST_DDI_ROOT": tmp_dir}
            resolver = SidecarRelationResolver(YAMLSidecarStore(), env=env)
            env_source = "$TEST_DDI_ROOT/src/a.py"
            relations = resolver.resolve(
                "$TEST_DDI_ROOT/src/a.py::login",
                {"source": env_source, "symbol": "login"},
            )
            self.assertEqual(relations, ["issue:env-test"])

    def test_resolve_matches_uid_with_env_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "src" / "a.py"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("x=1\n", encoding="utf-8")
            source.with_name(".a.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: login\n"
                "    relations:\n"
                "      - issue:uid-match\n",
                encoding="utf-8",
            )

            env = {"TEST_DDI_ROOT": tmp_dir}
            resolver = SidecarRelationResolver(YAMLSidecarStore(), env=env)
            relations = resolver.resolve(
                "$TEST_DDI_ROOT/src/a.py::login",
                {"source": str(source), "symbol": "login"},
            )
            self.assertEqual(relations, ["issue:uid-match"])


if __name__ == "__main__":
    unittest.main()
