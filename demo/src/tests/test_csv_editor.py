import tempfile
import unittest
from pathlib import Path

from demo.src.csv_editor import CsvEditor


class CsvEditorTests(unittest.TestCase):
    def test_append_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sample.csv"
            csv_path.write_text("name,score\n", encoding="utf-8")
            editor = CsvEditor(csv_path)

            editor.append_row(["alice", "95"])

            self.assertEqual(editor.read_rows(), [["name", "score"], ["alice", "95"]])

    def test_update_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sample.csv"
            csv_path.write_text("name,score\nalice,95\n", encoding="utf-8")
            editor = CsvEditor(csv_path)

            editor.update_cell(1, 1, "96")

            self.assertEqual(editor.read_rows()[1][1], "96")


if __name__ == "__main__":
    unittest.main()
