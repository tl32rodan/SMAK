"""Minimal YAML parsing utilities."""

from __future__ import annotations

from typing import Any


def load_yaml(text: str) -> Any:
    """Parse a minimal subset of YAML for configuration and metadata."""

    lines = [
        line.rstrip("\n")
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return None
    parsed, _ = _parse_block(lines, 0, 0)
    return parsed


def _parse_block(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    result: Any = None
    index = start
    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation at line: {line}")
        stripped = line.strip()
        if stripped.startswith("- "):
            result = [] if result is None else result
            if not isinstance(result, list):
                raise ValueError("Mixed list and mapping in YAML block.")
            item, index = _parse_list_item(lines, index, indent)
            result.append(item)
            continue
        key, value_text = _split_mapping(stripped)
        result = {} if result is None else result
        if not isinstance(result, dict):
            raise ValueError("Mixed mapping and list in YAML block.")
        if value_text:
            result[key] = _parse_scalar(value_text)
            index += 1
            continue
        nested, index = _parse_block(lines, index + 1, indent + 2)
        result[key] = nested
    return result, index


def _parse_list_item(lines: list[str], index: int, indent: int) -> tuple[Any, int]:
    line = lines[index]
    stripped = line.strip()[2:]
    if not stripped:
        return _parse_block(lines, index + 1, indent + 2)
    if ": " in stripped:
        key, value_text = _split_mapping(stripped)
        item: dict[str, Any] = {key: _parse_scalar(value_text) if value_text else None}
        index += 1
        if index < len(lines):
            next_indent = len(lines[index]) - len(lines[index].lstrip(" "))
            if next_indent > indent:
                nested, index = _parse_block(lines, index, indent + 2)
                if isinstance(nested, dict):
                    item.update(nested)
        if item[key] is None:
            item[key] = ""
        return item, index
    return _parse_scalar(stripped), index + 1


def _split_mapping(stripped: str) -> tuple[str, str]:
    if ":" not in stripped:
        return stripped, ""
    key, value = stripped.split(":", 1)
    return key.strip(), value.strip()


def _parse_scalar(value: str) -> Any:
    if not value:
        return ""
    if value.isdigit():
        return int(value)
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value
