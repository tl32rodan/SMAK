"""Tests for the SMAK CLI."""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner


class TestCli(unittest.TestCase):
    def test_default_config_template_includes_index_uri(self) -> None:
        cli = importlib.import_module("smak.cli")
        template = cli._default_config_template()
        self.assertIn("uri: ./smak/source_code", template)
        self.assertIn("uri: ./smak/issues", template)
        self.assertIn("uri: ./smak/tests", template)
        self.assertIn("uri: ./smak/documentation", template)
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

    def test_new_commands_present(self) -> None:
        runner = CliRunner()
        cli = importlib.import_module("smak.cli")
        help_output = runner.invoke(cli.main, ["--help"]).output
        for cmd in ("search", "search-all", "lookup", "enrich", "enrich-file",
                    "ingest", "describe", "health", "stats"):
            self.assertIn(cmd, help_output, f"Missing command: {cmd}")
        # Old fine-grained commands should not be present as top-level commands
        self.assertNotIn("query", help_output)


class TestNewCliCommands(unittest.TestCase):
    """Tests for CLI commands added in Phase 7A — all mock core_ops."""

    def setUp(self) -> None:
        self.runner = CliRunner()
        self.cli = importlib.import_module("smak.cli")

    def _patch_setup(self):
        mock_cfg = MagicMock()
        mock_emb = MagicMock()
        return patch("smak.cli.setup_config", return_value=(mock_cfg, mock_emb))

    def test_describe_json(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_describe", return_value={"indices": [{"name": "src"}]}):
            result = self.runner.invoke(self.cli.main, ["describe", "--json"])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertEqual(data["indices"][0]["name"], "src")

    def test_describe_human(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_describe", return_value={"indices": [{"name": "src"}]}):
            result = self.runner.invoke(self.cli.main, ["describe"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("src", result.output)

    def test_search_json(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_search", return_value={"hits": [{"uid": "a::b", "score": 0.9}]}):
            result = self.runner.invoke(self.cli.main, ["search", "--json", "test query"])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertEqual(data["hits"][0]["uid"], "a::b")

    def test_search_all_json(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_search_all", return_value={"src": {"hits": []}}):
            result = self.runner.invoke(self.cli.main, ["search-all", "--json", "query"])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertIn("src", data)

    def test_lookup_json(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_lookup", return_value={"found": True, "uid": "x::y"}):
            result = self.runner.invoke(self.cli.main, ["lookup", "--json", "x::y"])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertTrue(data["found"])

    def test_enrich_json(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_enrich_symbol", return_value={"status": "ok", "symbol": "foo"}):
            result = self.runner.invoke(self.cli.main, [
                "enrich", "--json", "--file", "a.py", "--symbol", "foo", "--intent", "does stuff",
            ])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertEqual(data["status"], "ok")

    def test_enrich_error_exits_nonzero(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_enrich_symbol", return_value={"status": "error", "message": "bad"}):
            result = self.runner.invoke(self.cli.main, [
                "enrich", "--file", "a.py", "--symbol", "foo",
            ])
            self.assertNotEqual(result.exit_code, 0)

    def test_enrich_file_json(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_enrich_file", return_value={"total_symbols": 3}):
            result = self.runner.invoke(self.cli.main, [
                "enrich-file", "--json", "--file", "a.py",
            ])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertEqual(data["total_symbols"], 3)

    def test_ingest_json(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_ingest", return_value={"files": 5, "skipped": 0, "vectors": 20}):
            result = self.runner.invoke(self.cli.main, ["ingest", "--json"])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertEqual(data["files"], 5)

    def test_health_json_healthy(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_health", return_value={"status": "healthy", "issues": []}):
            result = self.runner.invoke(self.cli.main, ["health", "--json"])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertEqual(data["status"], "healthy")

    def test_health_unhealthy_exits_nonzero(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_health", return_value={"status": "unhealthy", "issues": ["bad"]}):
            result = self.runner.invoke(self.cli.main, ["health"])
            self.assertNotEqual(result.exit_code, 0)

    def test_stats_json(self) -> None:
        with self._patch_setup(), \
             patch("smak.cli.do_graph_stats", return_value={"by_index": {}, "overall_coverage_pct": 50}):
            result = self.runner.invoke(self.cli.main, ["stats", "--json"])
            self.assertEqual(result.exit_code, 0)
            data = json.loads(result.output)
            self.assertEqual(data["overall_coverage_pct"], 50)

    def test_search_help_shows_options(self) -> None:
        result = self.runner.invoke(self.cli.main, ["search", "--help"])
        self.assertIn("--config", result.output)
        self.assertIn("--index", result.output)
        self.assertIn("--top-k", result.output)
        self.assertIn("--json", result.output)
        self.assertIn("QUERY", result.output)

    def test_enrich_help_shows_options(self) -> None:
        result = self.runner.invoke(self.cli.main, ["enrich", "--help"])
        self.assertIn("--file", result.output)
        self.assertIn("--symbol", result.output)
        self.assertIn("--intent", result.output)
        self.assertIn("--relation", result.output)
        self.assertIn("--bidirectional", result.output)


if __name__ == "__main__":
    unittest.main()
