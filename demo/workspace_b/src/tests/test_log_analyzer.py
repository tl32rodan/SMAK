"""Unit tests for LogAnalyzer."""

import tempfile
import unittest
from pathlib import Path

from log_analyzer import LogAnalyzer

SAMPLE_LOG = """\
2024-01-15 10:23:01 [INFO] server started on port 8080
2024-01-15 10:23:45 [ERROR] connection refused: host unreachable
2024-01-15 10:24:02 [WARNING] retry attempt 1 of 3
2024-01-15 10:24:10 [ERROR] connection refused: host unreachable
not a valid log line
2024-01-15 10:24:30 [INFO] connection established
"""


class LogAnalyzerTests(unittest.TestCase):
    def _make_log(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_parse_valid_entries(self) -> None:
        log_path = self._make_log(SAMPLE_LOG)
        analyzer = LogAnalyzer(log_path)
        entries = analyzer.parse()
        # 5 valid lines; the malformed line is silently skipped
        self.assertEqual(len(entries), 5)
        self.assertEqual(entries[0].level, "INFO")
        self.assertEqual(entries[1].level, "ERROR")

    def test_count_by_level(self) -> None:
        log_path = self._make_log(SAMPLE_LOG)
        analyzer = LogAnalyzer(log_path)
        counts = analyzer.count_by_level()
        self.assertEqual(counts["ERROR"], 2)
        self.assertEqual(counts["INFO"], 2)
        self.assertEqual(counts["WARNING"], 1)

    def test_filter_by_level(self) -> None:
        log_path = self._make_log(SAMPLE_LOG)
        analyzer = LogAnalyzer(log_path)
        errors = analyzer.filter_by_level("error")  # case-insensitive
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(e.level == "ERROR" for e in errors))


if __name__ == "__main__":
    unittest.main()
