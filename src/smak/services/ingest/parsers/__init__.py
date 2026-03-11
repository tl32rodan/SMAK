"""Backwards-compatible re-exports from smak.parsers."""

from smak.parsers import (
    NullParser,
    Parser,
    PerlParser,
    PythonParser,
    SimpleLineParser,
    get_parser_for_path,
)

__all__ = [
    "NullParser",
    "Parser",
    "PerlParser",
    "PythonParser",
    "SimpleLineParser",
    "get_parser_for_path",
]
