"""Tests for sidecar path mismatch warnings."""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smak.services.sidecar import SidecarService
from smak.sidecar.store import YAMLSidecarStore


class TestSidecarPathMismatch(unittest.TestCase):
    def test_update_sidecar_warns_when_workspace_path_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace" / "src"
            workspace.mkdir(parents=True)
            source = workspace / "a.py"
            source.write_text("def hello():\n    pass\n", encoding="utf-8")

            env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
            service = SidecarService(sidecar_store=YAMLSidecarStore(), env=env)
            # First create the sidecar
            service.update(source)

            # Update with a relation that uses env var
            with self.assertLogs("smak.services.sidecar", level="WARNING") as cm:
                service.update(
                    source,
                    symbol="hello",
                    intent="greeting",
                    relations=["$DDI_ROOT_PATH/issues/bug.md::*"],
                )

            self.assertTrue(any("mismatch" in msg.lower() for msg in cm.output))

    def test_no_warning_when_no_env_vars_in_relations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "a.py"
            source.write_text("def hello():\n    pass\n", encoding="utf-8")

            service = SidecarService(sidecar_store=YAMLSidecarStore())
            service.update(source)

            # Plain absolute path relations — no warning expected
            logger = logging.getLogger("smak.services.sidecar")
            with patch.object(logger, "warning") as mock_warn:
                service.update(
                    source,
                    symbol="hello",
                    intent="greeting",
                    relations=["/abs/path/issues/bug.md::*"],
                )
                mock_warn.assert_not_called()

    def test_no_warning_when_paths_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "a.py"
            source.write_text("def hello():\n    pass\n", encoding="utf-8")

            env = {"DDI_ROOT_PATH": tmp_dir}
            service = SidecarService(sidecar_store=YAMLSidecarStore(), env=env)
            service.update(source)

            # Env var resolves to the same directory as the source
            logger = logging.getLogger("smak.services.sidecar")
            with patch.object(logger, "warning") as mock_warn:
                service.update(
                    source,
                    symbol="hello",
                    intent="greeting",
                    relations=["$DDI_ROOT_PATH/issues/bug.md::*"],
                )
                mock_warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
