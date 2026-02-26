from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smak.services.relation_resolver import SidecarRelationResolver, build_symbol_name_candidates
from smak.services.sidecar_store import SidecarStore


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
            source.with_name("csv_editor.py.sidecar.yaml").write_text(
                "symbols:\n"
                "  - name: CsvEditor.updatecell\n"
                "    relations:\n"
                "      - issue:1\n",
                encoding="utf-8",
            )

            resolver = SidecarRelationResolver(SidecarStore(workspace_root))
            relations = resolver.resolve(
                "src/csv_editor.py::CsvEditor.updatecell",
                {"source": "src/csv_editor.py", "symbol": "CsvEditor.updatecell"},
            )
            self.assertEqual(relations, ["issue:1"])


if __name__ == "__main__":
    unittest.main()
