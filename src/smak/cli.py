"""SMAK CLI — full-featured command-line interface.

Mirrors all MCP tools so callers (including All-Might) can use SMAK
via ``smak <command> --json`` without MCP.

Commands:
  init          Generate a workspace_config.yaml template
  doctor        Run mesh integrity diagnostics
  search        Semantic search within one index
  search-all    Semantic search across all indices
  lookup        Look up a UID in the vector store
  enrich        Annotate a symbol with intent/relations
  enrich-file   Sync a file's sidecar
  ingest        Re-ingest files into a vector store index
  describe      Describe workspace indices
  health        Run health checks (structured output)
  stats         Knowledge graph coverage statistics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from smak.config import load_config, load_embedding_config
from smak.core_ops import (
    do_describe,
    do_enrich_file,
    do_enrich_symbol,
    do_graph_stats,
    do_health,
    do_ingest,
    do_lookup,
    do_search,
    do_search_all,
    setup_config,
)
from smak.factory import create_doctor_service, init_config

_DEFAULT_EMBEDDING_SETUP = str(Path(__file__).resolve().parent / "embedding_setup.yaml")


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def _output(data: Any, as_json: bool) -> None:
    """Print *data* as JSON or human-readable text."""
    if as_json:
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        _pretty(data)


def _pretty(data: Any, indent: int = 0) -> None:
    """Recursively pretty-print dicts/lists for human consumption."""
    prefix = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                click.echo(f"{prefix}{k}:")
                _pretty(v, indent + 1)
            else:
                click.echo(f"{prefix}{k}: {v}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _pretty(item, indent)
                click.echo()
            else:
                click.echo(f"{prefix}- {item}")
    else:
        click.echo(f"{prefix}{data}")


def _default_config_template() -> str:
    return "\n".join(
        [
            "# SMAK Workspace Configuration",
            "#",
            "# Indices are not limited to these defaults — define any number",
            "# with any names. Write precise descriptions so agents can select",
            "# the right index.",
            "#",
            "# Optional: set path_env to use $ENV_VAR in UIDs instead of",
            "# absolute paths (useful for CliosoftSOS or multi-root setups).",
            "",
            "indices:",
            "  - name: source_code",
            "    description: Contains the project's source code (Python, Perl), "
            "function definitions, and logic.",
            "    paths:",
            "      - ./src",
            "    uri: ./smak_data/source_code",
            "    # path_env: DDI_ROOT_PATH",
            "  - name: issues",
            "    description: Contains historical bug reports, GitHub issues, "
            "and Jira tickets describing known problems.",
            "    paths:",
            "      - ./issues",
            "    uri: ./smak_data/issues",
            "  - name: tests",
            "    description: Contains unit tests, integration tests, and test cases.",
            "    paths:",
            "      - ./tests",
            "    uri: ./smak_data/tests",
            "  - name: documentation",
            "    description: Contains architecture diagrams, API docs, and "
            "general knowledge base.",
            "    paths:",
            "      - ./documentation",
            "    uri: ./smak_data/documentation",
            "",
        ]
    )


# ------------------------------------------------------------------
# Shared options
# ------------------------------------------------------------------

def _config_option(default: str = "workspace_config.yaml"):
    return click.option("--config", default=default, help="Path to workspace_config.yaml")


def _json_option():
    return click.option("--json", "as_json", is_flag=True, help="Output as JSON")


def _embedding_option():
    return click.option(
        "--embedding-setup", default=_DEFAULT_EMBEDDING_SETUP,
        help="Path to embedding_setup.yaml",
    )


def _index_option(default: str = "source_code"):
    return click.option("--index", default=default, help="Target index name")


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------

@click.group()
def main() -> None:
    """SMAK: Semantic Mesh Agentic Kernel CLI."""


# ------------------------------------------------------------------
# Bootstrap commands (unchanged)
# ------------------------------------------------------------------

@main.command()
@click.option("--path", "config_path", default="workspace_config.yaml", help="Config path")
@click.option("--force", is_flag=True, help="Overwrite existing config file")
def init(config_path: str, force: bool) -> None:
    """Generate a workspace_config.yaml template."""
    target = Path(config_path)
    if target.exists() and not force:
        raise click.ClickException(f"Config already exists: {target}")
    target.write_text(_default_config_template(), encoding="utf-8")
    click.echo(f"Wrote workspace config to {target}")


@main.command("doctor")
@_config_option()
@_embedding_option()
def doctor(config: str, embedding_setup: str) -> None:
    """Run mesh integrity diagnostics."""
    emb_cfg = load_embedding_config(embedding_setup)
    cfg = init_config(load_config(config), embedding_config=emb_cfg)
    service = create_doctor_service(cfg)
    try:
        service.validate_all()
    except RuntimeError as exc:
        for issue in str(exc).split("\n"):
            click.echo(issue)
        raise click.ClickException("Mesh diagnostics found problems.") from exc
    click.echo("Mesh diagnostics passed.")


# ------------------------------------------------------------------
# Search commands
# ------------------------------------------------------------------

@main.command()
@_config_option()
@_embedding_option()
@_index_option()
@_json_option()
@click.option("--top-k", default=5, type=int, help="Max results")
@click.argument("query")
def search(config: str, embedding_setup: str, index: str, as_json: bool,
           top_k: int, query: str) -> None:
    """Semantic search within a single index."""
    cfg, emb_cfg = setup_config(config, embedding_setup)
    result = do_search(cfg, query, index=index, top_k=top_k, embedding_config=emb_cfg)
    _output(result, as_json)


@main.command("search-all")
@_config_option()
@_embedding_option()
@_json_option()
@click.option("--top-k", default=3, type=int, help="Max results per index")
@click.argument("query")
def search_all(config: str, embedding_setup: str, as_json: bool,
               top_k: int, query: str) -> None:
    """Search across all indices at once."""
    cfg, emb_cfg = setup_config(config, embedding_setup)
    result = do_search_all(cfg, query, top_k=top_k, embedding_config=emb_cfg)
    _output(result, as_json)


@main.command()
@_config_option()
@_embedding_option()
@_index_option()
@_json_option()
@click.argument("uid")
def lookup(config: str, embedding_setup: str, index: str, as_json: bool,
           uid: str) -> None:
    """Look up a UID in the vector store."""
    cfg, emb_cfg = setup_config(config, embedding_setup)
    result = do_lookup(cfg, uid, index=index, embedding_config=emb_cfg)
    _output(result, as_json)


# ------------------------------------------------------------------
# Enrichment commands
# ------------------------------------------------------------------

@main.command()
@_config_option()
@_index_option()
@_json_option()
@click.option("--file", "file_path", required=True, help="Source file path")
@click.option("--symbol", required=True, help="Symbol name")
@click.option("--intent", default=None, help="Human description of symbol purpose")
@click.option("--relation", "relations", multiple=True, help="UID to link (repeatable)")
@click.option("--bidirectional", is_flag=True, help="Add reverse relations")
def enrich(config: str, index: str, as_json: bool,
           file_path: str, symbol: str, intent: str | None,
           relations: tuple[str, ...], bidirectional: bool) -> None:
    """Annotate a symbol with intent and/or relations."""
    cfg, emb_cfg = setup_config(config)
    rel_list = list(relations) if relations else None
    result = do_enrich_symbol(
        cfg, file_path, symbol,
        intent=intent, relations=rel_list,
        index=index, bidirectional=bidirectional,
    )
    _output(result, as_json)
    if not as_json and result.get("status") == "error":
        sys.exit(1)


@main.command("enrich-file")
@_config_option()
@_index_option()
@_json_option()
@click.option("--file", "file_path", required=True, help="Source file path")
def enrich_file(config: str, index: str, as_json: bool, file_path: str) -> None:
    """Sync a file's sidecar — create stubs for all symbols."""
    cfg, emb_cfg = setup_config(config)
    result = do_enrich_file(cfg, file_path, index=index)
    _output(result, as_json)


# ------------------------------------------------------------------
# Management commands
# ------------------------------------------------------------------

@main.command()
@_config_option()
@_embedding_option()
@_index_option()
@_json_option()
@click.option("--follow-symlinks/--no-follow-symlinks", default=True, help="Follow symlinks")
def ingest(config: str, embedding_setup: str, index: str, as_json: bool,
           follow_symlinks: bool) -> None:
    """Re-ingest source files into a vector store index."""
    cfg, emb_cfg = setup_config(config, embedding_setup)
    result = do_ingest(cfg, index=index, follow_symlinks=follow_symlinks,
                       embedding_config=emb_cfg)
    _output(result, as_json)


@main.command()
@_config_option()
@_json_option()
def describe(config: str, as_json: bool) -> None:
    """Describe workspace indices."""
    cfg, emb_cfg = setup_config(config)
    result = do_describe(cfg)
    _output(result, as_json)


@main.command()
@_config_option()
@_json_option()
def health(config: str, as_json: bool) -> None:
    """Run health checks (structured output)."""
    cfg, emb_cfg = setup_config(config)
    result = do_health(cfg)
    _output(result, as_json)
    if not as_json and result.get("status") == "unhealthy":
        sys.exit(1)


@main.command()
@_config_option()
@_json_option()
def stats(config: str, as_json: bool) -> None:
    """Knowledge graph coverage statistics."""
    cfg, emb_cfg = setup_config(config)
    result = do_graph_stats(cfg)
    _output(result, as_json)


__all__ = ["main"]
