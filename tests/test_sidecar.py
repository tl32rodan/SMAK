from smak.ingest.sidecar import SidecarLoader


def test_sidecar_loader_none_returns_empty() -> None:
    loader = SidecarLoader()

    assert loader.load(None) == {}


def test_sidecar_loader_parses_dict_json() -> None:
    loader = SidecarLoader()

    assert loader.load('{"key": "value"}') == {"key": "value"}


def test_sidecar_loader_wraps_non_dict_json() -> None:
    loader = SidecarLoader()

    assert loader.load("[1, 2]") == {"value": [1, 2]}


def test_sidecar_loader_wraps_invalid_json() -> None:
    loader = SidecarLoader()

    assert loader.load("not json") == {"raw": "not json"}
