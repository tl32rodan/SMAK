from smak.ingest.parsers import IssueParser, PerlParser, PythonParser, SimpleLineParser


def test_simple_line_parser_creates_units() -> None:
    parser = SimpleLineParser()

    units = parser.parse("one\n\n two ", source="file.txt")

    assert [unit.content for unit in units] == ["one", "two"]
    assert [unit.uid for unit in units] == ["file.txt:1", "file.txt:2"]
    assert all(unit.metadata["source"] == "file.txt" for unit in units)


def test_python_parser_extracts_symbols() -> None:
    parser = PythonParser()
    content = "def login():\n    return True\n\nclass User:\n    pass\n"

    units = parser.parse(content, source="main.py")

    assert [unit.uid for unit in units] == ["main.py::login", "main.py::User"]
    assert {unit.metadata["symbol"] for unit in units} == {"login", "User"}
    assert all(unit.source_type == "source_code" for unit in units)


def test_perl_parser_extracts_subs() -> None:
    parser = PerlParser()
    content = "sub login {\n}\n\nsub logout {\n}\n"

    units = parser.parse(content, source="main.pl")

    assert [unit.uid for unit in units] == ["main.pl::login", "main.pl::logout"]
    assert [unit.content for unit in units] == ["sub login {", "sub logout {"]


def test_issue_parser_reads_frontmatter() -> None:
    parser = IssueParser()
    content = "---\nid: 101\ntitle: Login issue\nrelations:\n  - code::login\n---\nBody"

    units = parser.parse(content, source="issue.md")

    assert units[0].uid == "issue::101"
    assert units[0].metadata["title"] == "Login issue"
    assert units[0].relations == ("code::login",)
