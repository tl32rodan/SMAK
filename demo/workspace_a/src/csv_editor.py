"""Simple CSV editor utilities for SMAK demo workflows."""

from __future__ import annotations

import csv
from pathlib import Path


class CsvEditor:
    """Minimal editor that appends rows and updates cells in CSV files."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def append_row(self, row: list[str]) -> None:
        with self.file_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)

    def update_cell(self, row_index: int, column_index: int, value: str) -> None:
        rows = self.read_rows()
        rows[row_index][column_index] = value
        with self.file_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)

    def read_rows(self) -> list[list[str]]:
        with self.file_path.open("r", encoding="utf-8", newline="") as handle:
            return [row for row in csv.reader(handle)]
