from __future__ import annotations

from memory.memory_manager import MemoryManager
from shared.models import MemoryKind, MemoryRecord, ProjectFact, UserPreference


def test_memory_save_and_search(tmp_path) -> None:
    db_path = tmp_path / "mem.db"
    manager = MemoryManager(str(db_path))
    item = MemoryRecord(
        id="1",
        content="hello memory world",
        tags=["test"],
        timestamp="2024-01-01",
        kind=MemoryKind.NOTE,
        meta={"source": "test"},
    )
    manager.save(item)

    results = manager.search("memory", kind=MemoryKind.NOTE)
    assert len(results) == 1
    assert results[0].content == item.content
    assert results[0].tags == item.tags


def test_user_pref_and_project_fact(tmp_path) -> None:
    db_path = tmp_path / "mem.db"
    manager = MemoryManager(str(db_path))

    pref = UserPreference(
        id="pref1",
        key="theme",
        value="dark",
        timestamp="2024-01-01",
        tags=["ui"],
    )
    fact = ProjectFact(
        id="fact1",
        project="projA",
        content="Uses FastAPI",
        timestamp="2024-01-02",
        tags=["stack"],
        meta={"author": "dev"},
    )

    manager.save_user_pref(pref)
    manager.save_project_fact(fact)

    prefs = manager.get_user_prefs("theme")
    assert len(prefs) == 1
    assert prefs[0].meta.get("value") == "dark"

    facts = manager.get_project_facts("projA")
    assert len(facts) == 1
    assert "FastAPI" in facts[0].content
