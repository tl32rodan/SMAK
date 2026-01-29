from smak.core.domain import KnowledgeUnit


def test_knowledge_unit_with_metadata_merges() -> None:
    unit = KnowledgeUnit(
        uid="u1",
        content="hello",
        source_type="source_code",
        relations=("issue::1",),
        metadata={"a": 1},
    )

    updated = unit.with_metadata({"b": 2})

    assert updated.metadata == {"a": 1, "b": 2}
    assert updated.uid == unit.uid
    assert updated.content == unit.content
    assert updated.source_type == unit.source_type
    assert updated.relations == unit.relations


def test_knowledge_unit_with_relations_overrides() -> None:
    unit = KnowledgeUnit(uid="u1", content="hello", source_type="issue")

    updated = unit.with_relations(["issue::2", "issue::3"])

    assert updated.relations == ("issue::2", "issue::3")
    assert updated.metadata == unit.metadata
