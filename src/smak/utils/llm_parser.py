"""Utilities for parsing LLM outputs."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_substring(text: str) -> str | None:
    """Return the first JSON object substring from text, if any."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_json_from_text(text: str) -> dict[str, Any]:
    """Parse JSON from text, tolerating extra surrounding content."""

    candidate = extract_json_substring(text) or text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON from LLM output: %s", exc)
        raise ValueError(f"Invalid JSON response: {exc}") from exc


def ensure_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate an action payload has the required structure."""

    if "tool" not in payload:
        raise ValueError("Action payload missing 'tool' field.")
    args = payload.get("args", [])
    kwargs = payload.get("kwargs", {})
    if isinstance(args, tuple):
        args = list(args)
    if not isinstance(args, list):
        args = [args]
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("Action payload 'kwargs' must be a dict.")
    payload["args"] = args
    payload["kwargs"] = kwargs
    return payload


__all__ = ["ensure_action_payload", "extract_json_substring", "parse_json_from_text"]
