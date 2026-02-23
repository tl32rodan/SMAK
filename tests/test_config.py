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
        self.assertEqual(config.indices, [])

    def test_load_config_reads_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    uri: ./shared/source_code\n"
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
            self.assertEqual(config.indices[0].uri, "./shared/source_code")
            self.assertEqual(config.llm.provider, "ollama")
            self.assertEqual(config.llm.temperature, 0.4)
            self.assertEqual(config.llm.api_base, "http://localhost:11434/v1")
            self.assertIsNone(config.embedding_dimensions)

    def test_load_config_index_uri_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(config.indices[0].name, "source_code")
            self.assertIsNone(config.indices[0].uri)


if __name__ == "__main__":
    unittest.main()
