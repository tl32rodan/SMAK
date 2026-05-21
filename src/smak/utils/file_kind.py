"""File-kind detection utilities."""

from __future__ import annotations

from pathlib import Path


_BINARY_SNIFF_BYTES = 8192


def is_binary_file(path: Path, sniff_bytes: int = _BINARY_SNIFF_BYTES) -> bool:
    """Return True if *path* appears to contain binary data.

    Uses null-byte sniffing on the first ``sniff_bytes`` bytes — the same
    heuristic used by ``git`` and ``grep`` to classify files. A file that
    cannot be opened is reported as non-binary so callers surface the
    underlying I/O error in their own read path.
    """
    try:
        with path.open("rb") as fp:
            chunk = fp.read(sniff_bytes)
    except OSError:
        return False
    return b"\x00" in chunk


__all__ = ["is_binary_file"]
