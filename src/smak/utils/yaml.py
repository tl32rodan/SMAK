"""YAML parsing helpers."""

from __future__ import annotations

import importlib
from typing import Any


def safe_load(text: str) -> Any:
    """Parse YAML content with PyYAML when available, with a subset fallback parser."""

    try:
        module = importlib.import_module("yaml")
    except ModuleNotFoundError:
        return _fallback_load(text)
    return module.safe_load(text)


def safe_dump(data: Any) -> str:
    """Serialize YAML content with PyYAML when available."""

    try:
        module = importlib.import_module("yaml")
    except ModuleNotFoundError:
        return _fallback_dump(data)
    return module.safe_dump(data, sort_keys=False)


def _parse_scalar(value: str) -> Any:
    raw = value.strip()
    if raw == "":
        return ""
    if raw.isdigit():
        return int(raw)
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    return raw


def _split_key_value(text: str) -> tuple[str, str]:
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def _fallback_load(text: str) -> Any:
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, 0)
    return value


def _parse_block(lines: list[str], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    stripped = lines[index].lstrip(" ")
    current_indent = len(lines[index]) - len(stripped)
    if current_indent < indent:
        return {}, index
    if stripped.startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_dict(lines, index, indent)


def _parse_list(lines: list[str], index: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    i = index
    while i < len(lines):
        raw = lines[i]
        stripped = raw.lstrip(" ")
        current_indent = len(raw) - len(stripped)
        if current_indent < indent or not stripped.startswith("- "):
            break
        content = stripped[2:].strip()
        if content == "":
            nested, i = _parse_block(lines, i + 1, indent + 2)
            items.append(nested)
            continue
        if ": " in content or content.endswith(":"):
            key, value = _split_key_value(content)
            obj: dict[str, Any] = {key: _parse_scalar(value)} if value else {key: None}
            i += 1
            while i < len(lines):
                nested_raw = lines[i]
                nested_stripped = nested_raw.lstrip(" ")
                nested_indent = len(nested_raw) - len(nested_stripped)
                if nested_indent <= indent:
                    break
                if nested_stripped.startswith("- "):
                    break
                nk, nv = _split_key_value(nested_stripped)
                if nv:
                    obj[nk] = _parse_scalar(nv)
                    i += 1
                else:
                    nested, i = _parse_block(lines, i + 1, nested_indent + 2)
                    obj[nk] = nested
            items.append(obj)
            continue
        items.append(_parse_scalar(content))
        i += 1
    return items, i


def _parse_dict(lines: list[str], index: int, indent: int) -> tuple[dict[str, Any], int]:
    obj: dict[str, Any] = {}
    i = index
    while i < len(lines):
        raw = lines[i]
        stripped = raw.lstrip(" ")
        current_indent = len(raw) - len(stripped)
        if current_indent < indent:
            break
        if current_indent > indent:
            i += 1
            continue
        if ":" not in stripped:
            i += 1
            continue
        key, value = _split_key_value(stripped)
        if value:
            obj[key] = _parse_scalar(value)
            i += 1
            continue
        nested, next_i = _parse_block(lines, i + 1, indent + 2)
        obj[key] = nested
        i = next_i
    return obj, i


def _fallback_dump(data: Any, indent: int = 0) -> str:
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{' ' * indent}{key}:")
                lines.append(_fallback_dump(value, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{' ' * indent}{key}: {_dump_scalar(value)}")
        return "\n".join(lines) + "\n"
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, dict) and item:
                first = True
                for key, value in item.items():
                    if first:
                        lines.append(f"{' ' * indent}- {key}: {_dump_scalar(value)}")
                        first = False
                    else:
                        lines.append(f"{' ' * (indent + 2)}{key}: {_dump_scalar(value)}")
            else:
                lines.append(f"{' ' * indent}- {_dump_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{' ' * indent}{_dump_scalar(data)}\n"


def _dump_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    return f'"{text}"' if text == "" else text


__all__ = ["safe_load", "safe_dump"]
