import unittest
from pathlib import Path

from smak.parsers import (
    NullParser,
    PerlParser,
    PythonParser,
    SimpleLineParser,
    get_parser_for_path,
)


class TestParsers(unittest.TestCase):
    def test_simple_line_parser_creates_single_unit_with_file_wildcard_symbol_uid(self) -> None:
        parser = SimpleLineParser()
        units = parser.parse("one\n\n two ", source="file.txt")
        expected_origin = str(Path("file.txt").resolve())

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].content, "one\n\n two")
        self.assertEqual(units[0].uid, f"{expected_origin}::*")
        self.assertEqual(units[0].metadata["symbol"], "*")

    def test_simple_line_parser_returns_empty_for_empty_content(self) -> None:
        parser = SimpleLineParser()
        self.assertEqual(parser.parse("   \n\n", source="file.txt"), [])

    def test_python_parser_extracts_symbols(self) -> None:
        units = PythonParser().parse(
            "def login():\n    return True\n\nclass User:\n    pass\n", source="main.py"
        )
        expected_source = str(Path("main.py").resolve())
        self.assertEqual(
            [unit.uid for unit in units],
            [f"{expected_source}::login", f"{expected_source}::User"],
        )

    def test_perl_parser_extracts_subs(self) -> None:
        units = PerlParser().parse("sub login {\n}\n\nsub logout {\n}\n", source="main.pl")
        expected_source = str(Path("main.pl").resolve())
        self.assertEqual(
            [unit.uid for unit in units],
            [f"{expected_source}::login", f"{expected_source}::logout"],
        )

    def test_perl_parser_extracts_sub_blocks_with_nested_braces(self) -> None:
        content = (
            "sub login {\n"
            "    my $v = 1;\n"
            "    if ($v) {\n"
            "        return $v;\n"
            "    }\n"
            "}\n"
        )

        units = PerlParser().parse(content, source="main.pl")

        self.assertEqual(len(units), 1)
        self.assertTrue(units[0].content.startswith("sub login {"))
        self.assertIn("if ($v) {", units[0].content)
        self.assertTrue(units[0].content.endswith("}"))

    def test_perl_parser_ignores_braces_in_strings_comments_and_pod(self) -> None:
        content = (
            "=head\n"
            "sub fake_from_pod {\n"
            "}\n"
            "=cut\n"
            "sub alpha {\n"
            "    my $s1 = \"} not real brace\";\n"
            "    my $s2 = '\\{ still string';\n"
            "    # sub fake_comment {\n"
            "    return 1;\n"
            "}\n"
            "sub beta {\n"
            "    return 2;\n"
            "}\n"
        )

        units = PerlParser().parse(content, source="main.pl")

        expected_source = str(Path("main.pl").resolve())
        self.assertEqual([unit.uid for unit in units], [f"{expected_source}::alpha", f"{expected_source}::beta"])
        self.assertNotIn("fake_from_pod", "\n".join(unit.content for unit in units))

    def test_perl_parser_handles_single_quote_with_double_quote_char(self) -> None:
        content = (
            "sub build_csv {\n"
            "    return Text::CSV->new({ quote_char => '\"' });\n"
            "}\n"
            "sub other {\n"
            "    my $x = \"ok\";\n"
            "    return $x;\n"
            "}\n"
        )

        units = PerlParser().parse(content, source="csv.pl")

        expected_source = str(Path("csv.pl").resolve())
        self.assertEqual(
            [unit.uid for unit in units],
            [f"{expected_source}::build_csv", f"{expected_source}::other"],
        )
        self.assertIn("quote_char => '\"'", units[0].content)

    def test_get_parser_for_path_routes_by_suffix(self) -> None:
        self.assertIsInstance(get_parser_for_path(Path("a.py")), PythonParser)
        self.assertIsInstance(get_parser_for_path(Path("a.pm")), PerlParser)
        self.assertIsInstance(get_parser_for_path(Path("a.t")), PerlParser)
        self.assertIsInstance(get_parser_for_path(Path("a.md")), SimpleLineParser)
        self.assertIsInstance(get_parser_for_path(Path("a.markdown")), SimpleLineParser)
        self.assertIsInstance(get_parser_for_path(Path("a.txt")), SimpleLineParser)
        self.assertIsInstance(get_parser_for_path(Path("a.csv")), SimpleLineParser)
        self.assertIsInstance(get_parser_for_path(Path("a.il")), SimpleLineParser)

    def test_get_parser_for_path_defaults_to_simple_line_parser(self) -> None:
        for suffix in (".json", ".patch", ".sh", ".yaml", ".cfg", ""):
            self.assertIsInstance(
                get_parser_for_path(Path(f"a{suffix}")),
                SimpleLineParser,
                msg=f"suffix {suffix!r} should default to SimpleLineParser",
            )

    def test_null_parser_skips_content(self) -> None:
        parser = NullParser()
        self.assertEqual(parser.parse("anything", source="a.bin"), [])

    def test_python_parser_always_uses_absolute_source(self) -> None:
        parser = PythonParser()
        units = parser.parse("def login():\n    return True\n", source="/repo/src/auth.py")
        self.assertEqual(units[0].uid, "/repo/src/auth.py::login")

    def test_python_parser_uses_env_var_in_uid_when_env_set(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        parser = PythonParser()
        units = parser.parse(
            "def login():\n    return True\n",
            source="/opt/ddi/online/src/auth.py",
            env=env,
        )
        self.assertEqual(units[0].uid, "$DDI_ROOT_PATH/src/auth.py::login")
        self.assertEqual(units[0].metadata["source"], "$DDI_ROOT_PATH/src/auth.py")

    def test_perl_parser_uses_env_var_in_uid_when_env_set(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        parser = PerlParser()
        units = parser.parse(
            "sub login {\n}\n",
            source="/opt/ddi/online/src/main.pl",
            env=env,
        )
        self.assertEqual(units[0].uid, "$DDI_ROOT_PATH/src/main.pl::login")
        self.assertEqual(units[0].metadata["source"], "$DDI_ROOT_PATH/src/main.pl")

    def test_simple_line_parser_uses_env_var_in_uid_when_env_set(self) -> None:
        env = {"DDI_ROOT_PATH": "/opt/ddi/online"}
        parser = SimpleLineParser()
        units = parser.parse(
            "some content",
            source="/opt/ddi/online/docs/readme.txt",
            env=env,
        )
        self.assertEqual(units[0].uid, "$DDI_ROOT_PATH/docs/readme.txt::*")
        self.assertEqual(units[0].metadata["source"], "$DDI_ROOT_PATH/docs/readme.txt")

    def test_python_parser_preserves_absolute_when_no_env(self) -> None:
        parser = PythonParser()
        units = parser.parse(
            "def login():\n    return True\n",
            source="/repo/src/auth.py",
            env=None,
        )
        self.assertEqual(units[0].uid, "/repo/src/auth.py::login")


if __name__ == "__main__":
    unittest.main()
