from pathlib import Path

from click.testing import CliRunner

from smak.cli import _default_config_template, _ingest_folder, main
from smak.config import SmakConfig, StorageConfig


class FakeIndex:
    def __init__(self, saved: list) -> None:
        self._saved = saved

    def add(self, docs: list) -> None:
        self._saved.extend(docs)


class FakeRegistry:
    def __init__(self, saved: list) -> None:
        self._saved = saved
        self.last_index: str | None = None

    def get_index(self, name: str) -> FakeIndex:
        self.last_index = name
        return FakeIndex(self._saved)


def test_default_config_template_includes_storage() -> None:
    template = _default_config_template()

    assert "storage:" in template
    assert "base_path: vault" in template


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
    registry = FakeRegistry(saved)

    def loader(base_path: str) -> FakeRegistry:
        assert base_path == "vault"
        return registry

    config = SmakConfig(storage=StorageConfig(base_path="vault"))

    stats = _ingest_folder(folder, "code", config, registry_loader=loader)

    assert stats.files == 1
    assert stats.vectors == 1
    assert registry.last_index == "code"
    assert saved[0].payload["relations"] == ["issue::1"]


def test_cli_init_and_ingest(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    folder = tmp_path / "repo"
    folder.mkdir()
    source = folder / "note.md"
    source.write_text("---\nid: issue-1\n---\nbody\n", encoding="utf-8")
    config_path = tmp_path / "workspace_config.yaml"
    config_path.write_text("storage:\n  base_path: vault\n", encoding="utf-8")

    saved: list = []
    registry = FakeRegistry(saved)
    monkeypatch.setattr("smak.cli._load_registry", lambda _: registry)

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
