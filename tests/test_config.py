from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smak.config import SmakConfig, load_config


class TestConfig(unittest.TestCase):
    def test_smak_config_defaults(self) -> None:
        config = SmakConfig()

        self.assertIsNone(config.embedding_dimensions)
        self.assertEqual(config.llm.provider, "openai")
        self.assertEqual(config.llm.temperature, 0.0)

    def test_load_config_reads_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    uri: data/source_code\n"
                "llm:\n"
                "  provider: ollama\n"
                "  model: llama3\n"
                "  temperature: 0.4\n"
                "  api_base: http://localhost:11434/v1\n"
                "embedding_dimensions: 12\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(config.indices[0].name, "source_code")
            self.assertEqual(config.llm.provider, "ollama")
            self.assertEqual(config.llm.temperature, 0.4)
            self.assertEqual(config.llm.api_base, "http://localhost:11434/v1")
            self.assertEqual(config.indices[0].uri, "data/source_code")
            self.assertIsNone(config.embedding_dimensions)

    def test_demo_workspace_config_omits_llm_block(self) -> None:
        demo_config = Path(__file__).resolve().parents[1] / "demo" / "workspace_config.yaml"

        config = load_config(demo_config)

        self.assertEqual(config.llm.provider, "openai")
        self.assertIsNone(config.llm.model)

    def test_demo_workspace_config_source_code_uri(self) -> None:
        demo_config = Path(__file__).resolve().parents[1] / "demo" / "workspace_config.yaml"

        config = load_config(demo_config)

        self.assertEqual(config.indices[0].uri, "./smak_data/source_code")

    def test_load_config_defaults_index_uri_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n  - name: docs\n    description: docs index\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertIsNone(config.indices[0].uri)


if __name__ == "__main__":
    unittest.main()
