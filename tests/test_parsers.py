from smak.ingest.parsers import SimpleLineParser


def test_simple_line_parser_creates_units() -> None:
    parser = SimpleLineParser()

    units = parser.parse("one\n\n two ", source="file.txt")

    assert [unit.content for unit in units] == ["one", "two"]
    assert [unit.uid for unit in units] == ["file.txt:1", "file.txt:2"]
    assert all(unit.source == "file.txt" for unit in units)
