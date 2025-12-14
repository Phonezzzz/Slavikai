from __future__ import annotations

import pytest

from memory.memory_manager import MemoryManager
from shared.models import MemoryKind, MemoryRecord, ProjectFact, UserPreference


def test_memory_record_validation_rejects_empty_content(tmp_path) -> None:
    manager = MemoryManager(str(tmp_path / "mem.db"))
    bad = MemoryRecord(
        id="1",
        kind=MemoryKind.NOTE,
        content="",
        tags=[],
        timestamp="2024-01-01",
    )
    with pytest.raises(ValueError):
        manager.save(bad)


def test_user_pref_validation(tmp_path) -> None:
    manager = MemoryManager(str(tmp_path / "mem.db"))
    pref = UserPreference(
        id="pref1",
        key="theme",
        value="dark",
        timestamp="2024-01-01",
        tags=["ui"],
    )
    manager.save_user_pref(pref)
    prefs = manager.get_user_prefs("theme")
    assert prefs
    assert prefs[0].meta.get("value") == "dark"


def test_project_fact_validation(tmp_path) -> None:
    manager = MemoryManager(str(tmp_path / "mem.db"))
    fact = ProjectFact(
        id="f1",
        project="proj",
        content="uses FastAPI",
        timestamp="2024-02-02",
        tags=["tech"],
        meta={"author": "me"},
    )
    manager.save_project_fact(fact)
    facts = manager.get_project_facts("proj")
    assert len(facts) == 1
    assert "FastAPI" in facts[0].content
