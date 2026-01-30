from smak.utils.yaml import safe_load


def test_safe_load_parses_mapping_and_list() -> None:
    text = "a: 1\nb:\n  - x\n  - y\n"

    parsed = safe_load(text)

    assert parsed == {"a": 1, "b": ["x", "y"]}


def test_safe_load_parses_nested_mapping_list_items() -> None:
    text = "items:\n  - name: alpha\n    value: 2\n"

    parsed = safe_load(text)

    assert parsed == {"items": [{"name": "alpha", "value": 2}]}
