from pathlib import Path

from smak.config import SmakConfig, load_config


def test_smak_config_defaults() -> None:
    config = SmakConfig()

    assert config.embedding_dimensions == 3
    assert config.llm.provider == "openai"
    assert config.llm.temperature == 0.0
    assert config.storage.provider == "milvus_lite"
    assert config.storage.uri == "./milvus_data.db"


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
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

    assert config.indices[0].name == "source_code"
    assert config.llm.provider == "ollama"
    assert config.llm.temperature == 0.4
    assert config.llm.api_base == "http://localhost:11434/v1"


def test_load_config_reads_storage(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text(
        "storage:\n  provider: milvus_lite\n  uri: data/vault.db\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.storage.provider == "milvus_lite"
    assert config.storage.uri == "data/vault.db"


def test_load_config_reads_legacy_base_path(tmp_path: Path) -> None:
    path = tmp_path / "workspace.yaml"
    path.write_text("storage:\n  base_path: data/legacy\n", encoding="utf-8")

    config = load_config(path)

    assert config.storage.uri == "data/legacy"
