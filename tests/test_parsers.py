import unittest

from smak.ingest.parsers import IssueParser, PerlParser, PythonParser, SimpleLineParser


class TestParsers(unittest.TestCase):
    def test_simple_line_parser_creates_units(self) -> None:
        parser = SimpleLineParser()

        units = parser.parse("one\n\n two ", source="file.txt")

        self.assertEqual([unit.content for unit in units], ["one", "two"])
        self.assertEqual([unit.uid for unit in units], ["file.txt:1", "file.txt:2"])
        self.assertTrue(all(unit.metadata["source"] == "file.txt" for unit in units))

    def test_python_parser_extracts_symbols(self) -> None:
        parser = PythonParser()
        content = "def login():\n    return True\n\nclass User:\n    pass\n"

        units = parser.parse(content, source="main.py")

        self.assertEqual(
            [unit.uid for unit in units],
            ["main.py::login", "main.py::User"],
        )
        self.assertEqual({unit.metadata["symbol"] for unit in units}, {"login", "User"})
        self.assertTrue(all(unit.source_type == "source_code" for unit in units))

    def test_perl_parser_extracts_subs(self) -> None:
        parser = PerlParser()
        content = "sub login {\n}\n\nsub logout {\n}\n"

        units = parser.parse(content, source="main.pl")

        self.assertEqual(
            [unit.uid for unit in units],
            ["main.pl::login", "main.pl::logout"],
        )
        self.assertEqual([unit.content for unit in units], ["sub login {", "sub logout {"])

    def test_issue_parser_reads_frontmatter(self) -> None:
        parser = IssueParser()
        content = "---\nid: 101\ntitle: Login issue\nrelations:\n  - code::login\n---\nBody"

        units = parser.parse(content, source="issue.md")

        self.assertEqual(units[0].uid, "issue:101")
        self.assertEqual(units[0].metadata["title"], "Login issue")
        self.assertEqual(units[0].relations, ("code::login",))

    def test_python_parser_uses_relative_source_with_root(self) -> None:
        parser = PythonParser(root_path="/repo")

        units = parser.parse("def login():\n    return True\n", source="/repo/src/auth.py")

        self.assertEqual(units[0].uid, "src/auth.py::login")
        self.assertEqual(units[0].metadata["source"], "src/auth.py")


if __name__ == "__main__":
    unittest.main()
