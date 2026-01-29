from pathlib import Path

from smak.config import SmakConfig, load_config


def test_smak_config_defaults() -> None:
    config = SmakConfig()

    assert config.embedding_dimensions == 3
    assert config.llm.provider == "openai"


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text(
        "indices:\n"
        "  - name: source_code\n"
        "    description: Source code files\n"
        "llm:\n"
        "  provider: ollama\n"
        "  model: llama3\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.indices[0].name == "source_code"
    assert config.llm.provider == "ollama"
