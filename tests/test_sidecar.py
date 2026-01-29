import pytest

from smak.core.domain import KnowledgeUnit
from smak.ingest.sidecar import IntegrityError, SidecarManager


def test_sidecar_loader_none_returns_empty() -> None:
    loader = SidecarManager()

    assert loader.load(None) == {}


def test_sidecar_loader_parses_yaml() -> None:
    loader = SidecarManager()

    assert loader.load("key: value") == {"key": "value"}


def test_sidecar_loader_wraps_non_dict_yaml() -> None:
    loader = SidecarManager()

    assert loader.load("- a\n- b\n") == {"value": ["a", "b"]}


def test_sidecar_validate_missing_symbol() -> None:
    loader = SidecarManager()
    metadata = {"symbols": {"missing": {"relations": ["issue::1"]}}}

    with pytest.raises(IntegrityError):
        loader.validate(["present"], metadata)


def test_sidecar_apply_enriches_units() -> None:
    loader = SidecarManager()
    units = [
        KnowledgeUnit(
            uid="main.py::login",
            content="def login(): pass",
            source_type="source_code",
            metadata={"symbol": "login"},
        )
    ]
    metadata = {"symbols": {"login": {"relations": ["issue::404"], "owner": "team"}}}

    enriched = loader.apply(units, metadata)

    assert enriched[0].relations == ("issue::404",)
    assert enriched[0].metadata["owner"] == "team"
