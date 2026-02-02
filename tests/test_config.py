from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smak.config import SmakConfig, load_config


class TestConfig(unittest.TestCase):
    def test_smak_config_defaults(self) -> None:
        config = SmakConfig()

        self.assertEqual(config.embedding_dimensions, 3)
        self.assertEqual(config.llm.provider, "openai")
        self.assertEqual(config.llm.temperature, 0.0)
        self.assertEqual(config.storage.provider, "milvus_lite")
        self.assertEqual(config.storage.uri, "./milvus_data.db")

    def test_load_config_reads_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "llm:\n"
                "  provider: ollama\n"
                "  model: llama3\n"
                "  temperature: 0.4\n"
                "  api_base: http://localhost:11434/v1\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(config.indices[0].name, "source_code")
            self.assertEqual(config.llm.provider, "ollama")
            self.assertEqual(config.llm.temperature, 0.4)
            self.assertEqual(config.llm.api_base, "http://localhost:11434/v1")

    def test_load_config_reads_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "storage:\n  provider: milvus_lite\n  uri: data/vault.db\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(config.storage.provider, "milvus_lite")
            self.assertEqual(config.storage.uri, "data/vault.db")

    def test_load_config_reads_legacy_base_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text("storage:\n  base_path: data/legacy\n", encoding="utf-8")

            config = load_config(path)

            self.assertEqual(config.storage.uri, "data/legacy")


if __name__ == "__main__":
    unittest.main()
