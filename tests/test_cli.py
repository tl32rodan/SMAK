from pathlib import Path

from click.testing import CliRunner

from smak.cli import _default_config_template, _ingest_folder, main
from smak.config import SmakConfig, StorageConfig


class FakeNode:
    def __init__(self, text: str, id_: str, metadata: dict) -> None:
        self.text = text
        self.id_ = id_
        self.metadata = metadata
        self.embedding: list[float] | None = None


class FakeVectorStore:
    def __init__(self, saved: list, index_name: str) -> None:
        self._saved = saved
        self.index_name = index_name

    def add(self, nodes: list) -> None:
        self._saved.extend(nodes)


def test_default_config_template_includes_storage() -> None:
    template = _default_config_template()

    assert "storage:" in template
    assert "provider: milvus_lite" in template
    assert "uri: ./milvus_data.db" in template


def test_ingest_folder_processes_files(tmp_path: Path) -> None:
    folder = tmp_path / "src"
    folder.mkdir()
    source = folder / "example.py"
    source.write_text("def foo():\n    return 1\n", encoding="utf-8")
    sidecar = folder / "example.py.sidecar.yaml"
    sidecar.write_text(
        "symbols:\n  foo:\n    relations:\n      - issue::1\n",
        encoding="utf-8",
    )

    saved: list = []
    created: dict[str, str] = {}

    def loader(index_name: str, config: SmakConfig) -> FakeVectorStore:
        assert config.storage.uri == "vault.db"
        created["index"] = index_name
        return FakeVectorStore(saved, index_name)

    config = SmakConfig(storage=StorageConfig(uri="vault.db"))

    stats = _ingest_folder(
        folder,
        "code",
        config,
        vector_store_loader=loader,
        node_class_loader=lambda: FakeNode,
    )

    assert stats.files == 1
    assert stats.vectors == 1
    assert created["index"] == "code"
    assert saved[0].metadata["relations"] == ["issue::1"]


def test_cli_init_and_ingest(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    folder = tmp_path / "repo"
    folder.mkdir()
    source = folder / "note.md"
    source.write_text("---\nid: issue-1\n---\nbody\n", encoding="utf-8")
    config_path = tmp_path / "workspace_config.yaml"
    config_path.write_text("storage:\n  uri: vault.db\n", encoding="utf-8")

    saved: list = []
    monkeypatch.setattr(
        "smak.cli._load_vector_store",
        lambda index, config: FakeVectorStore(saved, index),
    )
    monkeypatch.setattr("smak.cli._load_text_node_class", lambda: FakeNode)

    init_result = runner.invoke(
        main,
        [
            "init",
            "--path",
            str(tmp_path / "generated.yaml"),
        ],
    )

    assert init_result.exit_code == 0
    assert "workspace config" in init_result.output
    assert (tmp_path / "generated.yaml").exists()

    result = runner.invoke(
        main,
        [
            "ingest",
            "--folder",
            str(folder),
            "--index",
            "issues",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert "Ingestion Complete" in result.output
    assert saved


def test_load_vector_store_missing_dependency(monkeypatch) -> None:
    from smak import cli

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: None)

    try:
        cli._load_vector_store("code", SmakConfig())
    except Exception as exc:
        assert "llama-index-vector-stores-milvus" in str(exc)


def test_load_text_node_missing_dependency(monkeypatch) -> None:
    from smak import cli

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: None)

    try:
        cli._load_text_node_class()
    except Exception as exc:
        assert "llama-index-core" in str(exc)
