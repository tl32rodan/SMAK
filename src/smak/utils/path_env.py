"""Environment variable expansion for UIDs and paths."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def expand_env_path(raw: str) -> str:
    """Expand ``$VAR`` references in a path string.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    if not raw:
        return raw

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ValueError(
                f"Environment variable '{var_name}' is not set. "
                f"Cannot expand '${var_name}' in path."
            )
        return value

    return _ENV_PATTERN.sub(_replace, raw)


def contains_env_var(raw: str) -> bool:
    """Return ``True`` if *raw* contains ``$VAR`` references."""
    return bool(_ENV_PATTERN.search(raw))


def collapse_to_env(absolute_path: str, env_var: str) -> str | None:
    """Replace the prefix of *absolute_path* with ``$env_var`` if it matches.

    Returns ``None`` if *env_var* is not set or the path does not start with
    its value.
    """
    value = os.environ.get(env_var)
    if value is None:
        return None

    value = value.rstrip("/")
    if absolute_path == value:
        return f"${env_var}"
    if absolute_path.startswith(value + "/"):
        return f"${env_var}{absolute_path[len(value):]}"
    return None


def expand_uid(uid: str) -> str:
    """Expand environment variables in a UID (``path::symbol`` format).

    Only the path portion (before ``::`` ) is expanded.  If no ``::``
    separator is present the whole string is expanded.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    if "::" in uid:
        path_part, symbol = uid.split("::", 1)
        return f"{expand_env_path(path_part)}::{symbol}"
    return expand_env_path(uid)


def warn_path_mismatch(actual_path: Path, uid_path: str) -> str | None:
    """Return a warning string if *actual_path* does not match *uid_path*.

    Environment variables in *uid_path* are expanded before comparison.
    Returns ``None`` when the paths match.
    """
    try:
        expanded = expand_env_path(uid_path)
    except ValueError:
        expanded = uid_path

    if str(actual_path) == expanded:
        return None
    return (
        f"Path mismatch: actual sidecar location '{actual_path}' "
        f"differs from UID path '{expanded}'"
    )
