"""Perl source code parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from smak.core.domain import KnowledgeUnit

_SUB_START_REGEX = re.compile(
    r"(?m)^[ \t]*sub[ \t]+([A-Za-z_][A-Za-z0-9_]*)[^{;\n]*\{"
)
_POD_LINE_START_REGEX = re.compile(r"(?m)^=(head\d*|pod|begin|for|item|over|back|encoding)\b")
_SINGLE_QUOTE_REGEX = re.compile(r"'(?:\\.|[^'\\\n])*'")
_DOUBLE_QUOTE_REGEX = re.compile(r'"(?:\\.|[^"\\\n])*"')


@dataclass
class PerlParser:

    def parse(self, content: str, source: str | None = None) -> list[KnowledgeUnit]:
        original = content or ""
        cleaned = _clean_for_structure_scan(original)
        abs_source = str(Path(source).resolve()) if source else None

        units: list[KnowledgeUnit] = []
        cursor = 0
        while True:
            match = _SUB_START_REGEX.search(cleaned, cursor)
            if match is None:
                break
            name = match.group(1)
            body_start = cleaned.find("{", match.start())
            if body_start == -1:
                cursor = match.end()
                continue
            body_end = _find_matching_brace(cleaned, body_start)
            if body_end is None:
                cursor = match.end()
                continue
            segment = original[match.start() : body_end + 1].strip()
            units.append(
                KnowledgeUnit(
                    uid=f"{abs_source}::{name}" if abs_source else name,
                    content=segment,
                    source_type="source_code",
                    metadata={"language": "perl", "symbol": name, "source": abs_source},
                )
            )
            cursor = body_end + 1
        return units


def _clean_for_structure_scan(content: str) -> str:
    cleaned = _mask_pod_sections(content)
    cleaned = _mask_regex(cleaned, _SINGLE_QUOTE_REGEX)
    cleaned = _mask_regex(cleaned, _DOUBLE_QUOTE_REGEX)
    cleaned = _mask_single_line_comments(cleaned)
    return cleaned


def _mask_pod_sections(content: str) -> str:
    chars = list(content)
    in_pod = False
    line_start = 0

    for index, char in enumerate(content):
        if char != "\n":
            continue
        line = content[line_start:index]
        if in_pod:
            if line.strip() == "=cut":
                in_pod = False
            for pos in range(line_start, index):
                if chars[pos] != "\n":
                    chars[pos] = " "
        else:
            if _POD_LINE_START_REGEX.match(line):
                in_pod = True
                for pos in range(line_start, index):
                    if chars[pos] != "\n":
                        chars[pos] = " "
        line_start = index + 1

    if line_start < len(content):
        line = content[line_start:]
        if in_pod or _POD_LINE_START_REGEX.match(line):
            for pos in range(line_start, len(content)):
                if chars[pos] != "\n":
                    chars[pos] = " "

    return "".join(chars)


def _mask_regex(content: str, pattern: re.Pattern[str]) -> str:
    chars = list(content)
    for match in pattern.finditer(content):
        for index in range(match.start(), match.end()):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _mask_single_line_comments(content: str) -> str:
    lines = []
    for line in content.splitlines(keepends=True):
        hash_index = line.find("#")
        if hash_index == -1:
            lines.append(line)
            continue
        masked = list(line)
        for index in range(hash_index, len(masked)):
            if masked[index] != "\n":
                masked[index] = " "
        lines.append("".join(masked))
    return "".join(lines)


def _find_matching_brace(content: str, start_index: int) -> int | None:
    depth = 0
    for index in range(start_index, len(content)):
        char = content[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


