"""Tests for the slimmed-down CLI (init + doctor only)."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import click
from click.testing import CliRunner


class TestCli(unittest.TestCase):
    def test_default_config_template_includes_index_uri(self) -> None:
        cli = importlib.import_module("smak.cli")
        template = cli._default_config_template()
        self.assertIn("uri: ./smak_data/source_code", template)
        self.assertIn("paths:", template)
        self.assertIn("path_env", template)

    def test_init_creates_config_file(self) -> None:
        runner = CliRunner()
        cli = importlib.import_module("smak.cli")
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "workspace_config.yaml"
            result = runner.invoke(cli.main, ["init", "--path", str(target)])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(target.exists())
            self.assertIn("indices:", target.read_text())

    def test_init_refuses_overwrite_without_force(self) -> None:
        runner = CliRunner()
        cli = importlib.import_module("smak.cli")
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "workspace_config.yaml"
            target.write_text("existing", encoding="utf-8")
            result = runner.invoke(cli.main, ["init", "--path", str(target)])
            self.assertNotEqual(result.exit_code, 0)
            self.assertEqual(target.read_text(), "existing")

    def test_init_force_overwrites(self) -> None:
        runner = CliRunner()
        cli = importlib.import_module("smak.cli")
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "workspace_config.yaml"
            target.write_text("old", encoding="utf-8")
            result = runner.invoke(cli.main, ["init", "--path", str(target), "--force"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("indices:", target.read_text())

    def test_doctor_command_accepts_embedding_setup_option(self) -> None:
        runner = CliRunner()
        cli = importlib.import_module("smak.cli")
        result = runner.invoke(cli.main, ["doctor", "--help"])
        self.assertIn("--embedding-setup", result.output)
        self.assertIn("--config", result.output)

    def test_removed_commands_not_present(self) -> None:
        runner = CliRunner()
        cli = importlib.import_module("smak.cli")
        help_output = runner.invoke(cli.main, ["--help"]).output
        self.assertNotIn("ingest", help_output)
        self.assertNotIn("query", help_output)
        self.assertNotIn("sidecar", help_output)


if __name__ == "__main__":
    unittest.main()
