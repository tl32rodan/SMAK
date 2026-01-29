from smak.core.domain import KnowledgeUnit
from smak.db.adapter import InMemoryAdapter


def test_in_memory_adapter_saves_and_loads() -> None:
    adapter = InMemoryAdapter()
    unit = KnowledgeUnit(uid="u1", content="content")

    adapter.save_units([unit])

    assert adapter.load_units() == [unit]
