import unittest

from smak.utils.yaml import safe_load


class TestYaml(unittest.TestCase):
    def test_safe_load_parses_mapping_and_list(self) -> None:
        text = "a: 1\nb:\n  - x\n  - y\n"

        parsed = safe_load(text)

        self.assertEqual(parsed, {"a": 1, "b": ["x", "y"]})

    def test_safe_load_parses_nested_mapping_list_items(self) -> None:
        text = "items:\n  - name: alpha\n    value: 2\n"

        parsed = safe_load(text)

        self.assertEqual(parsed, {"items": [{"name": "alpha", "value": 2}]})


if __name__ == "__main__":
    unittest.main()
