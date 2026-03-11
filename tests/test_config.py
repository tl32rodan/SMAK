from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smak.config import SmakConfig, load_config


class TestConfig(unittest.TestCase):
    def test_smak_config_defaults(self) -> None:
        config = SmakConfig()

        self.assertIsNone(config.embedding_dimensions)
        self.assertEqual(config.indices, [])

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
            self.assertTrue(config.indices[0].uri.endswith("data/source_code"))
            self.assertIsNone(config.embedding_dimensions)
            self.assertFalse(hasattr(config, "llm"))

    def test_demo_workspace_config_loads_without_llm_field(self) -> None:
        demo_config = (
            Path(__file__).resolve().parents[1]
            / "demo"
            / "workspace_a"
            / "workspace_config.yaml"
        )

        config = load_config(demo_config)

        self.assertFalse(hasattr(config, "llm"))

    def test_load_config_ignores_llm_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "llm:\n"
                "  provider: legacy\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertFalse(hasattr(config, "llm"))

    def test_demo_workspace_config_source_code_uri(self) -> None:
        demo_config = (
            Path(__file__).resolve().parents[1]
            / "demo"
            / "workspace_a"
            / "workspace_config.yaml"
        )

        config = load_config(demo_config)

        self.assertTrue(config.indices[0].uri.endswith("smak_data/source_code"))

    def test_load_config_defaults_index_uri_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n  - name: docs\n    description: docs index\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertIsNotNone(config.indices[0].uri)
            self.assertTrue(config.indices[0].uri.endswith("smak_data/docs"))


    def test_load_config_supports_paths_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    paths:\n"
                "      - ./src\n"
                "      - ./lib\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(len(config.indices[0].paths), 2)
            self.assertTrue(config.indices[0].paths[0].endswith("src"))
            self.assertTrue(config.indices[0].paths[1].endswith("lib"))

    def test_load_config_default_path_is_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: docs\n"
                "    description: docs index\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertIsInstance(config.indices[0].paths, list)
            self.assertEqual(len(config.indices[0].paths), 1)


    def test_glob_pattern_expands_to_matching_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create directories: modules/auth, modules/billing, modules/README.txt (file)
            (Path(tmp_dir) / "modules" / "auth").mkdir(parents=True)
            (Path(tmp_dir) / "modules" / "billing").mkdir(parents=True)
            (Path(tmp_dir) / "modules" / "README.txt").touch()

            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    paths:\n"
                "      - ./modules/*\n",
                encoding="utf-8",
            )

            config = load_config(path)

            resolved = config.indices[0].paths
            # Only directories should be included (not README.txt)
            self.assertEqual(len(resolved), 2)
            names = sorted(Path(p).name for p in resolved)
            self.assertEqual(names, ["auth", "billing"])

    def test_glob_pattern_no_match_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    paths:\n"
                "      - ./nonexistent_*\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("matched zero directories", str(ctx.exception))

    def test_glob_mixed_with_literal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "src").mkdir()
            (Path(tmp_dir) / "plugins" / "alpha").mkdir(parents=True)
            (Path(tmp_dir) / "plugins" / "beta").mkdir(parents=True)

            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    paths:\n"
                "      - ./src\n"
                "      - ./plugins/*\n",
                encoding="utf-8",
            )

            config = load_config(path)

            resolved = config.indices[0].paths
            self.assertEqual(len(resolved), 3)
            names = sorted(Path(p).name for p in resolved)
            self.assertEqual(names, ["alpha", "beta", "src"])

    def test_literal_path_unchanged_by_glob_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "src").mkdir()

            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    paths:\n"
                "      - ./src\n",
                encoding="utf-8",
            )

            config = load_config(path)

            resolved = config.indices[0].paths
            self.assertEqual(len(resolved), 1)
            self.assertTrue(resolved[0].endswith("src"))


if __name__ == "__main__":
    unittest.main()
