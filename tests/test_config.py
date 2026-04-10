from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smak.config import EmbeddingConfig, SmakConfig, load_config, load_embedding_config


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
                "    uri: /tmp/smak_test_data/source_code\n"
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
            self.assertEqual(config.indices[0].uri, "/tmp/smak_test_data/source_code")
            self.assertIsNone(config.embedding_dimensions)
            self.assertFalse(hasattr(config, "llm"))

    def test_demo_workspace_config_loads_without_llm_field(self) -> None:
        import os
        from unittest.mock import patch as mock_patch

        demo_config = (
            Path(__file__).resolve().parents[1]
            / "demo"
            / "workspace_a"
            / "workspace_config.yaml"
        )
        smak_data = str(demo_config.parent / "smak_data")

        with mock_patch.dict(os.environ, {"SMAK_DATA": smak_data}):
            config = load_config(demo_config)

        self.assertFalse(hasattr(config, "llm"))

    def test_load_config_ignores_llm_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    uri: /tmp/smak_test_data\n"
                "llm:\n"
                "  provider: legacy\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertFalse(hasattr(config, "llm"))

    def test_demo_workspace_config_source_code_uri(self) -> None:
        import os
        from unittest.mock import patch as mock_patch

        demo_config = (
            Path(__file__).resolve().parents[1]
            / "demo"
            / "workspace_a"
            / "workspace_config.yaml"
        )
        smak_data = str(demo_config.parent / "smak_data")

        with mock_patch.dict(os.environ, {"SMAK_DATA": smak_data}):
            config = load_config(demo_config)

        self.assertTrue(config.indices[0].uri.endswith("smak_data/source_code"))

    def test_load_config_raises_when_uri_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n  - name: docs\n    description: docs\n    uri: ./relative/path\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("relative", str(ctx.exception).lower())

    def test_load_config_raises_when_uri_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n  - name: docs\n    description: docs index\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("uri", str(ctx.exception))


    def test_load_config_supports_paths_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    uri: /tmp/smak_test_data\n"
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
                "    description: docs index\n"
                "    uri: /tmp/smak_test_data\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertIsInstance(config.indices[0].paths, list)
            self.assertEqual(len(config.indices[0].paths), 1)


    def test_glob_pattern_expands_to_matching_files_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create directories and a file under modules/
            (Path(tmp_dir) / "modules" / "auth").mkdir(parents=True)
            (Path(tmp_dir) / "modules" / "billing").mkdir(parents=True)
            (Path(tmp_dir) / "modules" / "README.txt").touch()

            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    uri: /tmp/smak_test_data\n"
                "    paths:\n"
                "      - ./modules/*\n",
                encoding="utf-8",
            )

            config = load_config(path)

            resolved = config.indices[0].paths
            # Both files and directories should be included
            self.assertEqual(len(resolved), 3)
            names = sorted(Path(p).name for p in resolved)
            self.assertEqual(names, ["README.txt", "auth", "billing"])

    def test_glob_pattern_no_match_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    uri: /tmp/smak_test_data\n"
                "    paths:\n"
                "      - ./nonexistent_*\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("matched zero files or directories", str(ctx.exception))

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
                "    uri: /tmp/smak_test_data\n"
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
                "    uri: /tmp/smak_test_data\n"
                "    paths:\n"
                "      - ./src\n",
                encoding="utf-8",
            )

            config = load_config(path)

            resolved = config.indices[0].paths
            self.assertEqual(len(resolved), 1)
            self.assertTrue(resolved[0].endswith("src"))

    def test_glob_pattern_expands_to_matching_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scripts_dir = Path(tmp_dir) / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "app.py").touch()
            (scripts_dir / "utils.py").touch()
            (scripts_dir / "readme.txt").touch()

            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: scripts\n"
                "    description: Python scripts\n"
                "    uri: /tmp/smak_test_data\n"
                "    paths:\n"
                "      - ./scripts/*.py\n",
                encoding="utf-8",
            )

            config = load_config(path)

            resolved = config.indices[0].paths
            self.assertEqual(len(resolved), 2)
            names = sorted(Path(p).name for p in resolved)
            self.assertEqual(names, ["app.py", "utils.py"])

    def test_glob_mixed_files_and_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "data" / "subdir").mkdir(parents=True)
            (Path(tmp_dir) / "data" / "file.csv").touch()

            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: data\n"
                "    description: Data files\n"
                "    uri: /tmp/smak_test_store\n"
                "    paths:\n"
                "      - ./data/*\n",
                encoding="utf-8",
            )

            config = load_config(path)

            resolved = config.indices[0].paths
            self.assertEqual(len(resolved), 2)
            names = sorted(Path(p).name for p in resolved)
            self.assertEqual(names, ["file.csv", "subdir"])

    def test_literal_file_path_passes_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "specific.py").touch()

            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: single\n"
                "    description: Single file\n"
                "    uri: /tmp/smak_test_data\n"
                "    paths:\n"
                "      - ./specific.py\n",
                encoding="utf-8",
            )

            config = load_config(path)

            resolved = config.indices[0].paths
            self.assertEqual(len(resolved), 1)
            self.assertTrue(resolved[0].endswith("specific.py"))


    def test_load_config_reads_path_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "src").mkdir()
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source code files\n"
                "    uri: /tmp/smak_test_data\n"
                "    paths:\n"
                "      - ./src\n"
                "    path_env: DDI_ROOT_PATH\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertEqual(config.indices[0].path_env, "DDI_ROOT_PATH")

    def test_load_config_path_env_defaults_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n  - name: docs\n    description: docs index\n    uri: /tmp/smak_test_data\n",
                encoding="utf-8",
            )

            config = load_config(path)

            self.assertIsNone(config.indices[0].path_env)

    def test_load_config_expands_env_var_in_paths(self) -> None:
        import os
        from unittest.mock import patch as mock_patch

        with tempfile.TemporaryDirectory() as tmp_dir:
            src_dir = Path(tmp_dir) / "src"
            src_dir.mkdir()
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source\n"
                "    uri: /tmp/smak_test_data\n"
                "    paths:\n"
                f'      - "$TEST_SMAK_ROOT/src"\n',
                encoding="utf-8",
            )

            with mock_patch.dict(os.environ, {"TEST_SMAK_ROOT": tmp_dir}):
                config = load_config(path)

            self.assertTrue(config.indices[0].paths[0].endswith("src"))
            self.assertTrue(Path(config.indices[0].paths[0]).is_absolute())

    def test_load_config_raises_on_undefined_env_in_paths(self) -> None:
        import os

        key = "_SMAK_TEST_UNDEF_CONFIG_"
        os.environ.pop(key, None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workspace.yaml"
            path.write_text(
                "indices:\n"
                "  - name: source_code\n"
                "    description: Source\n"
                "    uri: /tmp/smak_test_data\n"
                "    paths:\n"
                f'      - "${key}/src"\n',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(path)


class TestEmbeddingConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = EmbeddingConfig()
        self.assertEqual(cfg.api_base, "http://f15dtpai1:11434")
        self.assertEqual(cfg.model, "nomic_embed_text:latest")
        self.assertEqual(cfg.timeout, 600.0)
        self.assertEqual(cfg.batch_size, 64)

    def test_load_embedding_config_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "embedding_setup.yaml"
            path.write_text(
                "api_base: http://custom:9999\n"
                "model: custom-model\n"
                "timeout: 300.0\n"
                "batch_size: 128\n",
                encoding="utf-8",
            )
            cfg = load_embedding_config(path)
            self.assertEqual(cfg.api_base, "http://custom:9999")
            self.assertEqual(cfg.model, "custom-model")
            self.assertEqual(cfg.timeout, 300.0)
            self.assertEqual(cfg.batch_size, 128)

    def test_load_embedding_config_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "embedding_setup.yaml"
            path.write_text("api_base: http://other:5555\n", encoding="utf-8")
            cfg = load_embedding_config(path)
            self.assertEqual(cfg.api_base, "http://other:5555")
            self.assertEqual(cfg.model, "nomic_embed_text:latest")  # default
            self.assertEqual(cfg.timeout, 600.0)  # default

    def test_load_embedding_config_missing_file_returns_defaults(self) -> None:
        cfg = load_embedding_config("/nonexistent/path.yaml")
        self.assertEqual(cfg, EmbeddingConfig())

    def test_load_embedding_config_ignores_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "embedding_setup.yaml"
            path.write_text(
                "api_base: http://x:1\n"
                "unknown_key: should_be_ignored\n",
                encoding="utf-8",
            )
            cfg = load_embedding_config(path)
            self.assertEqual(cfg.api_base, "http://x:1")
            self.assertFalse(hasattr(cfg, "unknown_key"))

    def test_load_embedding_config_default_path(self) -> None:
        cfg = load_embedding_config()
        self.assertIsInstance(cfg, EmbeddingConfig)


if __name__ == "__main__":
    unittest.main()
