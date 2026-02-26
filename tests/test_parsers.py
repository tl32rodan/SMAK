import unittest
from pathlib import Path

from smak.services.ingest.parsers import (
    IssueParser,
    PerlParser,
    PythonParser,
    SimpleLineParser,
    get_parser_for_path,
)


class TestParsers(unittest.TestCase):
    def test_simple_line_parser_creates_units(self) -> None:
        parser = SimpleLineParser()
        units = parser.parse("one\n\n two ", source="file.txt")
        self.assertEqual([unit.content for unit in units], ["one", "two"])
        self.assertEqual([unit.uid for unit in units], ["file.txt:1", "file.txt:2"])

    def test_python_parser_extracts_symbols(self) -> None:
        units = PythonParser().parse(
            "def login():\n    return True\n\nclass User:\n    pass\n", source="main.py"
        )
        self.assertEqual([unit.uid for unit in units], ["main.py::login", "main.py::User"])

    def test_perl_parser_extracts_subs(self) -> None:
        units = PerlParser().parse("sub login {\n}\n\nsub logout {\n}\n", source="main.pl")
        self.assertEqual([unit.uid for unit in units], ["main.pl::login", "main.pl::logout"])

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

        self.assertEqual([unit.uid for unit in units], ["main.pl::alpha", "main.pl::beta"])
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

        self.assertEqual([unit.uid for unit in units], ["csv.pl::build_csv", "csv.pl::other"])
        self.assertIn("quote_char => '\"'", units[0].content)

    def test_issue_parser_uses_symbol_from_frontmatter(self) -> None:
        parser = IssueParser()
        content = (
            "---\n"
            "symbol: csv-editor-known-issues\n"
            "relations:\n"
            "  - src/csv_editor.py::edit\n"
            "---\n"
            "# CSV Editor Known Issues\n"
            "Body"
        )
        units = parser.parse(content, source="issue.md")
        self.assertEqual(units[0].uid, "csv-editor-known-issues")
        self.assertEqual(units[0].relations, ("src/csv_editor.py::edit",))

    def test_issue_parser_falls_back_to_header_slug(self) -> None:
        units = IssueParser().parse("# Login Issue Tracker\nDetails", source="issue.md")
        self.assertEqual(units[0].uid, "login-issue-tracker")

    def test_issue_parser_falls_back_to_filename_slug(self) -> None:
        units = IssueParser().parse("No markdown header", source="docs/My Issue.md")
        self.assertEqual(units[0].uid, "my-issue")

    def test_get_parser_for_path_routes_by_suffix(self) -> None:
        self.assertIsInstance(get_parser_for_path(Path("a.py")), PythonParser)
        self.assertIsInstance(get_parser_for_path(Path("a.pm")), PerlParser)
        self.assertIsInstance(get_parser_for_path(Path("a.t")), PerlParser)
        self.assertIsInstance(get_parser_for_path(Path("a.md")), IssueParser)
        self.assertIsInstance(get_parser_for_path(Path("a.txt")), SimpleLineParser)

    def test_python_parser_uses_relative_source_with_root(self) -> None:
        parser = PythonParser(root_path="/repo")
        units = parser.parse("def login():\n    return True\n", source="/repo/src/auth.py")
        self.assertEqual(units[0].uid, "src/auth.py::login")


if __name__ == "__main__":
    unittest.main()
