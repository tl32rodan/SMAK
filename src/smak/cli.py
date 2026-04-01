"""Minimal CLI for SMAK — bootstrapping and debugging only.

All regular operations (search, ingest, sidecar enrichment) go through
the MCP server.  This CLI exists for two purposes:

  smak init     — generate a workspace_config.yaml template
  smak doctor   — run mesh integrity diagnostics from the terminal
"""

from __future__ import annotations

from pathlib import Path

import click

from smak.config import load_config, load_embedding_config
from smak.factory import create_doctor_service, init_config

_DEFAULT_EMBEDDING_SETUP = str(Path(__file__).resolve().parent / "embedding_setup.yaml")


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
            "  - name: tests",
            "    description: Contains unit tests, integration tests, and test cases.",
            "    paths:",
            "      - ./tests",
            "  - name: documentation",
            "    description: Contains architecture diagrams, API docs, and "
            "general knowledge base.",
            "    paths:",
            "      - ./documentation",
            "",
        ]
    )


@click.group()
def main() -> None:
    """SMAK: Semantic Mesh Agentic Kernel CLI."""


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
@click.option("--config", default="workspace_config.yaml", help="Path to workspace config")
@click.option(
    "--embedding-setup",
    default=_DEFAULT_EMBEDDING_SETUP,
    help="Path to embedding_setup.yaml",
)
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


__all__ = ["init", "main", "doctor"]
