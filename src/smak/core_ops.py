"""Core operations shared by MCP server and CLI.

Every public ``do_*`` function corresponds to one MCP tool / CLI command.
Both the MCP adapter and the CLI delegate here so the logic stays DRY.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from smak.config import EmbeddingConfig, SmakConfig, load_config, load_embedding_config
from smak.factory import (
    create_doctor_service,
    create_query_service,
    create_sidecar_service,
    init_config,
    load_and_validate_vector_store,
    on_ghost_source,
    sidecar_skip_file,
)
from smak.services import IngestService
from smak.utils.embedding import InternalNomicEmbedding

_STALE_SYMBOL_RE = re.compile(r'--symbol "([^"]+)"')


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

def setup_config(
    config_path: str | Path,
    embedding_setup: str | Path | None = None,
) -> tuple[SmakConfig, EmbeddingConfig]:
    """Load workspace config and embedding config from paths.

    Returns ``(SmakConfig, EmbeddingConfig)`` — the config has
    ``embedding_dimensions`` populated.
    """
    emb_cfg = load_embedding_config(embedding_setup)
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}.")
    cfg = init_config(load_config(path), embedding_config=emb_cfg)
    return cfg, emb_cfg


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _get_index_config(config: SmakConfig, index: str) -> object:
    index_config = config.get_index(index)
    if index_config is None:
        raise ValueError(f"Index '{index}' not found in configuration.")
    return index_config


def _load_index_vector_store(
    config: SmakConfig, index: str,
) -> tuple[object, object]:
    index_config = _get_index_config(config, index)
    vector_store = load_and_validate_vector_store(index_config, config)
    return vector_store, index_config


def _create_query_service(
    config: SmakConfig, index: str,
    embedding_config: EmbeddingConfig | None = None,
) -> object:
    vs, ic = _load_index_vector_store(config, index)
    return create_query_service(vs, config, ic, embedding_config=embedding_config)


def _format_candidates(candidates: list[str]) -> str:
    if len(candidates) == 1:
        return f"'{candidates[0]}'"
    if len(candidates) == 2:
        return f"'{candidates[0]}' or '{candidates[1]}'"
    quoted = [f"'{c}'" for c in candidates]
    return ", ".join(quoted[:-1]) + f", or {quoted[-1]}"


def resolve_source_path(
    index: str, index_config: object, file_path: str,
) -> Path:
    """Resolve a file path relative to the index roots.

    Tries in order: absolute path, relative to each root, filename search.
    Raises ``FileNotFoundError`` or ``ValueError`` (ambiguous) on failure.
    """
    index_roots = [Path(p).resolve() for p in index_config.paths if Path(p).is_dir()]
    index_files = [Path(p).resolve() for p in index_config.paths if Path(p).is_file()]
    raw = Path(file_path)

    if raw.is_absolute() and raw.exists():
        return raw
    if not raw.is_absolute():
        for root in index_roots:
            candidate = (root / raw).resolve()
            if candidate.exists():
                return candidate

    name = raw.name
    all_candidates: list[Path] = []
    for f in index_files:
        if f.name == name:
            all_candidates.append(f)
    for root in index_roots:
        all_candidates.extend(root.rglob(f"*{name}") if name else [])
    candidates = sorted(all_candidates)

    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        rel: list[str] = []
        for c in candidates:
            for root in index_roots:
                try:
                    rel.append(str(c.resolve().relative_to(root)))
                    break
                except ValueError:
                    continue
            else:
                rel.append(str(c.resolve()))
        raise ValueError(
            f"Ambiguous file path '{file_path}'. "
            f"Did you mean {_format_candidates(rel)}?"
        )

    roots_display = ", ".join(f"'{r}'" for r in index_roots)
    raise FileNotFoundError(
        f"File path resolution failed under index roots [{roots_display}] "
        f"(index='{index}', file_path='{file_path}'). "
        "Try using semantic_search first and pass one of the returned file paths."
    )


def _resolve_file(
    config: SmakConfig, index: str, file_path: str,
) -> tuple[object, Path]:
    """Return ``(SidecarService, resolved_source_path)``."""
    ic = _get_index_config(config, index)
    return create_sidecar_service(env=config.env), resolve_source_path(index, ic, file_path)


def _add_reverse_relation(
    cfg: SmakConfig, svc: object, target_uid: str, source_uid: str,
) -> dict[str, Any]:
    """Add a reverse relation from *target* back to *source*."""
    if "::" not in target_uid:
        return {"target": target_uid, "status": "skip", "message": "No :: separator in UID"}

    path_part, symbol_part = target_uid.rsplit("::", 1)

    target_index = None
    for idx_cfg in cfg.indices:
        for idx_path in idx_cfg.paths:
            resolved = Path(idx_path).resolve()
            target_resolved = Path(path_part).resolve()
            try:
                target_resolved.relative_to(resolved)
                target_index = idx_cfg.name
                break
            except ValueError:
                continue
        if target_index:
            break

    if not target_index:
        return {"target": target_uid, "status": "skip", "message": "Target index not found"}

    try:
        target_path = resolve_source_path(
            target_index,
            _get_index_config(cfg, target_index),
            path_part,
        )
    except (FileNotFoundError, ValueError):
        return {"target": target_uid, "status": "skip", "message": "Target file not found"}

    try:
        svc.update(target_path, symbol=symbol_part, relations=[source_uid])
        return {"target": target_uid, "status": "ok"}
    except Exception as exc:
        return {"target": target_uid, "status": "error", "message": str(exc)}


# ------------------------------------------------------------------
# Public operations
# ------------------------------------------------------------------

def do_describe(config: SmakConfig) -> dict[str, Any]:
    """Describe workspace indices."""
    return {
        "indices": [
            {
                "name": idx.name,
                "description": idx.description,
                "paths": idx.paths,
            }
            for idx in config.indices
        ],
    }


def do_search(
    config: SmakConfig, query: str,
    index: str = "source_code", top_k: int = 5,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any]:
    """Semantic search within a single index."""
    svc = _create_query_service(config, index, embedding_config=embedding_config)
    result = svc.search(query, top_k=top_k)
    return result if isinstance(result, dict) else {}


def do_search_all(
    config: SmakConfig, query: str,
    indices: list[str] | None = None, top_k: int = 3,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any]:
    """Search across multiple (or all) indices."""
    target = indices or [idx.name for idx in config.indices]
    results: dict[str, Any] = {}
    for name in target:
        try:
            svc = _create_query_service(config, name, embedding_config=embedding_config)
            results[name] = svc.search(query, top_k=top_k)
        except ValueError:
            results[name] = {"error": f"Index '{name}' not found"}
        except Exception as exc:
            results[name] = {"error": str(exc)}
    return results


def do_lookup(
    config: SmakConfig, uid: str,
    index: str = "source_code",
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any]:
    """Look up a specific UID in the vector store."""
    svc = _create_query_service(config, index, embedding_config=embedding_config)
    return svc.lookup(uid)


def do_ingest(
    config: SmakConfig, index: str = "source_code",
    follow_symlinks: bool = True,
    embedding_config: EmbeddingConfig | None = None,
) -> dict[str, Any]:
    """Re-ingest source files into a vector store index."""
    vs, ic = _load_index_vector_store(config, index)
    emb_cfg = embedding_config
    stats = IngestService(vector_store=vs).ingest_paths(
        [Path(p) for p in ic.paths],
        follow_symlinks=follow_symlinks,
        skip_file=sidecar_skip_file,
        on_ghost_source=on_ghost_source,
        embedder_loader=lambda: InternalNomicEmbedding(embedding_config=emb_cfg),
        env=config.env or None,
    )
    return {
        "files": stats.files,
        "skipped": stats.skipped,
        "vectors": stats.vectors,
    }


def do_enrich_symbol(
    config: SmakConfig, file_path: str, symbol: str,
    intent: str | None = None, relations: list[str] | None = None,
    index: str = "source_code",
    bidirectional: bool = False,
) -> dict[str, Any]:
    """Annotate a single code symbol with intent and/or relations."""
    svc, source_path = _resolve_file(config, index, file_path)

    valid_symbols = svc.inspect(source_path)
    if symbol not in valid_symbols:
        return {
            "status": "error",
            "message": (
                f"Symbol '{symbol}' not found in {file_path}. "
                f"Valid symbols: {valid_symbols}"
            ),
            "valid_symbols": valid_symbols,
        }

    try:
        svc.update(source_path)
    except ValueError as exc:
        stale = _STALE_SYMBOL_RE.findall(str(exc))
        if not stale:
            return {"status": "error", "message": str(exc)}
        for s in stale:
            try:
                svc.clear_symbol(source_path, s)
            except ValueError:
                pass
        svc.update(source_path)

    if intent is None and relations is None:
        return {
            "status": "ok",
            "message": f"Sidecar synced for {file_path}. "
            f"No intent/relations provided for '{symbol}'.",
        }
    svc.update(source_path, symbol=symbol, intent=intent, relations=relations)

    reverse_results: list[dict[str, Any]] = []
    if bidirectional and relations:
        source_uid = f"{source_path}::{symbol}"
        for rel_uid in relations:
            try:
                result = _add_reverse_relation(config, svc, rel_uid, source_uid)
                reverse_results.append(result)
            except Exception as exc:
                reverse_results.append({
                    "target": rel_uid, "status": "error", "message": str(exc),
                })

    result: dict[str, Any] = {
        "status": "ok",
        "file_path": str(source_path),
        "symbol": symbol,
        "intent": intent,
        "relations": relations,
    }
    if reverse_results:
        result["reverse_relations"] = reverse_results
    return result


def do_enrich_file(
    config: SmakConfig, file_path: str,
    index: str = "source_code",
) -> dict[str, Any]:
    """Sync a file's sidecar — create stubs for every symbol."""
    svc, source = _resolve_file(config, index, file_path)
    return svc.update(source)


def do_enrich_batch(
    config: SmakConfig, file_paths: list[str],
    index: str = "source_code",
) -> list[dict[str, Any]]:
    """Sync sidecars for multiple files. Continues on error."""
    ic = _get_index_config(config, index)
    svc = create_sidecar_service(env=config.env)
    results: list[dict[str, Any]] = []
    for fp in file_paths:
        try:
            source = resolve_source_path(index, ic, fp)
            results.append(svc.update(source))
        except Exception as exc:
            results.append({"file_path": fp, "error": str(exc)})
    return results


def do_health(config: SmakConfig) -> dict[str, Any]:
    """Run mesh integrity diagnostics."""
    service = create_doctor_service(config)
    try:
        service.validate_all()
    except RuntimeError as exc:
        return {"status": "unhealthy", "issues": str(exc).split("\n")}
    return {"status": "healthy", "issues": []}


def do_graph_stats(config: SmakConfig) -> dict[str, Any]:
    """Return knowledge graph coverage statistics."""
    service = create_doctor_service(config)
    return service.graph_stats()
