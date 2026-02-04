import unittest

from smak.core.domain import KnowledgeUnit


class TestKnowledgeUnit(unittest.TestCase):
    def test_knowledge_unit_with_metadata_merges(self) -> None:
        unit = KnowledgeUnit(
            uid="u1",
            content="hello",
            source_type="source_code",
            relations=("issue:1",),
            metadata={"a": 1},
        )

        updated = unit.with_metadata({"b": 2})

        self.assertEqual(updated.metadata, {"a": 1, "b": 2})
        self.assertEqual(updated.uid, unit.uid)
        self.assertEqual(updated.content, unit.content)
        self.assertEqual(updated.source_type, unit.source_type)
        self.assertEqual(updated.relations, unit.relations)

    def test_knowledge_unit_with_relations_overrides(self) -> None:
        unit = KnowledgeUnit(uid="u1", content="hello", source_type="issue")

        updated = unit.with_relations(["issue:2", "issue:3"])

        self.assertEqual(updated.relations, ("issue:2", "issue:3"))
        self.assertEqual(updated.metadata, unit.metadata)


if __name__ == "__main__":
    unittest.main()
