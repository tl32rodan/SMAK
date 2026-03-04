"""Sidecar components.

Available persistence backends:
- ``YAMLSidecarStore``: per-source hidden YAML sidecar files adjacent to sources.
- ``CentralizedYAMLSidecarStore``: single workspace file at ``.smak/sidecars.yaml``.
- ``SidecarDirectoryStore`` (documented option, not yet implemented): mirror sources under
  ``.smak/sidecars/`` to isolate per-file sidecars from source directories.
"""

from smak.sidecar.centralized_store import CentralizedYAMLSidecarStore
from smak.sidecar.manager import IntegrityError, SidecarManager
from smak.sidecar.paths import (
    PRIMARY_SIDECAR_SUFFIX,
    SIDECAR_SUFFIXES,
    is_sidecar_file,
    iter_sidecar_files,
    sidecar_path_for_source,
    source_path_from_sidecar,
)
from smak.sidecar.protocols import SidecarLoader, SidecarStore
from smak.sidecar.store import YAMLSidecarStore

__all__ = [
    "CentralizedYAMLSidecarStore",
    "IntegrityError",
    "PRIMARY_SIDECAR_SUFFIX",
    "SIDECAR_SUFFIXES",
    "SidecarLoader",
    "SidecarManager",
    "SidecarStore",
    "YAMLSidecarStore",
    "is_sidecar_file",
    "iter_sidecar_files",
    "sidecar_path_for_source",
    "source_path_from_sidecar",
]
