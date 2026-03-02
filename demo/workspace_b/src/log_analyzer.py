"""Log file analysis utilities for SMAK demo workflow B."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s+\[(?P<level>\w+)\]\s+(?P<message>.+)$"
)


@dataclass
class LogEntry:
    """A single parsed log line."""

    timestamp: str
    level: str
    message: str


class LogAnalyzer:
    """Parse and analyze structured log files."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self._entries: list[LogEntry] | None = None

    def parse(self) -> list[LogEntry]:
        """Parse the log file and return all valid entries.

        Lines that do not match the expected format are silently skipped.
        Expected format: YYYY-MM-DD HH:MM:SS [LEVEL] message
        """
        entries = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            match = LOG_PATTERN.match(line.strip())
            if match:
                entries.append(LogEntry(**match.groupdict()))
        self._entries = entries
        return entries

    def count_by_level(self) -> dict[str, int]:
        """Return a count of log entries grouped by severity level.

        Level strings are returned as-is (case preserved).
        Call parse() first or it will be called automatically.
        """
        if self._entries is None:
            self.parse()
        return dict(Counter(entry.level for entry in self._entries))

    def filter_by_level(self, level: str) -> list[LogEntry]:
        """Return only entries matching a specific severity level.

        Comparison is case-insensitive.
        Call parse() first or it will be called automatically.
        """
        if self._entries is None:
            self.parse()
        return [e for e in self._entries if e.level.upper() == level.upper()]
