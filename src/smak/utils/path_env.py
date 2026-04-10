"""Environment variable expansion for UIDs and paths.

All functions accept an explicit ``env`` dict — they never read from
``os.environ``.  The env dict is populated from the ``env:`` block in
``workspace_config.yaml``.
"""

from __future__ import annotations

import re
from pathlib import Path

_ENV_PATTERN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def expand_env_path(raw: str, env: dict[str, str]) -> str:
    """Expand ``$VAR`` references in *raw* using *env*.

    Raises:
        ValueError: If a referenced variable is not defined in *env*.
    """
    if not raw:
        return raw

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = env.get(var_name)
        if value is None:
            raise ValueError(
                f"Variable '{var_name}' is not defined in the workspace env block. "
                f"Cannot expand '${var_name}' in path."
            )
        return value

    return _ENV_PATTERN.sub(_replace, raw)


def contains_env_var(raw: str) -> bool:
    """Return ``True`` if *raw* contains ``$VAR`` references."""
    return bool(_ENV_PATTERN.search(raw))


def collapse_to_env(absolute_path: str, env: dict[str, str]) -> str | None:
    """Replace the longest matching env-value prefix with ``$key``.

    Returns ``None`` if no env value is a prefix of *absolute_path*.
    """
    best_key: str | None = None
    best_len = 0

    for key, value in env.items():
        value = value.rstrip("/")
        if not value:
            continue
        if absolute_path == value and len(value) > best_len:
            best_key = key
            best_len = len(value)
        elif absolute_path.startswith(value + "/") and len(value) > best_len:
            best_key = key
            best_len = len(value)

    if best_key is None:
        return None
    value = env[best_key].rstrip("/")
    if absolute_path == value:
        return f"${best_key}"
    return f"${best_key}{absolute_path[len(value):]}"


def expand_uid(uid: str, env: dict[str, str]) -> str:
    """Expand variables in a UID (``path::symbol`` format).

    Only the path portion (before ``::`` ) is expanded.  If no ``::``
    separator is present the whole string is expanded.

    Raises:
        ValueError: If a referenced variable is not defined in *env*.
    """
    if "::" in uid:
        path_part, symbol = uid.split("::", 1)
        return f"{expand_env_path(path_part, env)}::{symbol}"
    return expand_env_path(uid, env)


def warn_path_mismatch(actual_path: Path, uid_path: str, env: dict[str, str]) -> str | None:
    """Return a warning string if *actual_path* does not match *uid_path*.

    Variables in *uid_path* are expanded using *env* before comparison.
    Returns ``None`` when the paths match.
    """
    try:
        expanded = expand_env_path(uid_path, env)
    except ValueError:
        expanded = uid_path

    if str(actual_path) == expanded:
        return None
    return (
        f"Path mismatch: actual sidecar location '{actual_path}' "
        f"differs from UID path '{expanded}'"
    )
