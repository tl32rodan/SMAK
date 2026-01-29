from smak.utils.yaml import load_yaml


def test_load_yaml_parses_mapping_and_list() -> None:
    text = "a: 1\nb:\n  - x\n  - y\n"

    parsed = load_yaml(text)

    assert parsed == {"a": 1, "b": ["x", "y"]}


def test_load_yaml_parses_nested_mapping_list_items() -> None:
    text = "items:\n  - name: alpha\n    value: 2\n"

    parsed = load_yaml(text)

    assert parsed == {"items": [{"name": "alpha", "value": 2}]}
