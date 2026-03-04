from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SidecarLoader(Protocol):
    """Read-only sidecar access. Sufficient for relation resolution."""

    def load_payload_for_source(self, source_path: Path) -> dict[str, Any]: ...

    def load_symbols_for_source(self, source_path: Path) -> list[dict[str, Any]]: ...


@runtime_checkable
class SidecarStore(Protocol):
    """Full read/write sidecar access. Required for init/update operations."""

    def load_payload_for_source(self, source_path: Path) -> dict[str, Any]: ...

    def load_symbols_for_source(self, source_path: Path) -> list[dict[str, Any]]: ...

    def save_symbols_for_source(self, source_path: Path, symbols: list[dict[str, Any]]) -> Path: ...

    def merge_symbols_for_source(
        self,
        source_path: Path,
        updates: list[dict[str, Any]],
    ) -> tuple[Path, int]: ...
