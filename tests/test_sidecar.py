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
    metadata = {
        "symbols": [
            {"name": "missing", "relations": ["issue::1"], "intent": "missing-intent"}
        ]
    }

    with pytest.raises(IntegrityError):
        loader.validate(["present"], metadata)


def test_sidecar_validate_rejects_invalid_schema() -> None:
    loader = SidecarManager()
    metadata = {"symbols": [{"intent": "missing-name"}]}

    with pytest.raises(IntegrityError):
        loader.validate(["present"], metadata)


def test_sidecar_apply_enriches_units() -> None:
    loader = SidecarManager()
    units = [
        KnowledgeUnit(
            uid="main.py::login",
            content="def login(): pass",
            source_type="source_code",
            metadata={"symbol": "login", "source": "main.py"},
        )
    ]
    metadata = {
        "symbols": [
            {
                "name": "login",
                "relations": ["issue::404"],
                "intent": "auth",
                "owner": "team",
            }
        ]
    }

    enriched = loader.apply(units, metadata)

    assert enriched[0].relations == ("issue::404",)
    assert enriched[0].metadata["owner"] == "team"
    assert enriched[0].metadata["file_name"] == "main.py"
    assert enriched[0].metadata["symbol_name"] == "login"
    assert enriched[0].metadata["intent"] == "auth"
    assert enriched[0].metadata["mesh_relations"] == ["issue::404"]


def test_sidecar_apply_uses_existing_relations_when_missing() -> None:
    loader = SidecarManager()
    units = [
        KnowledgeUnit(
            uid="main.py::logout",
            content="def logout(): pass",
            source_type="source_code",
            relations=("issue::200",),
            metadata={"symbol": "logout", "source": "main.py"},
        )
    ]
    metadata = {"symbols": []}

    enriched = loader.apply(units, metadata)

    assert enriched[0].relations == ("issue::200",)
    assert enriched[0].metadata["mesh_relations"] == ["issue::200"]
