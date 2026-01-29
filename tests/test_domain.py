from smak.core.domain import KnowledgeUnit


def test_knowledge_unit_with_metadata_merges() -> None:
    unit = KnowledgeUnit(uid="u1", content="hello", metadata={"a": 1}, source="src")

    updated = unit.with_metadata({"b": 2})

    assert updated.metadata == {"a": 1, "b": 2}
    assert updated.uid == unit.uid
    assert updated.content == unit.content
    assert updated.source == unit.source
