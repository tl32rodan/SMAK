from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

SIDECAR_SUFFIXES: tuple[str, ...] = (".sidecar.yaml", ".sidecar.yml")
PRIMARY_SIDECAR_SUFFIX = SIDECAR_SUFFIXES[0]


def is_sidecar_file(path: Path) -> bool:
    return path.name.endswith(SIDECAR_SUFFIXES)


def iter_sidecar_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and is_sidecar_file(path):
            yield path


def sidecar_path_for_source(source_path: Path) -> Path:
    return source_path.with_name(f".{source_path.name}{PRIMARY_SIDECAR_SUFFIX}")


def source_path_from_sidecar(sidecar_path: Path) -> Path:
    for suffix in SIDECAR_SUFFIXES:
        if sidecar_path.name.endswith(suffix):
            source_name = sidecar_path.name[: -len(suffix)]
            if source_name.startswith("."):
                source_name = source_name[1:]
            return sidecar_path.with_name(source_name)
    return sidecar_path
