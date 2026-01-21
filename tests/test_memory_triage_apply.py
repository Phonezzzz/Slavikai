from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory.categorized_memory_store import CategorizedMemoryStore
from memory.triage import TriageApplyPolicy, apply_triage, triage_preview, undo_last_triage
from shared.memory_category_models import MemoryCategory, MemorySource


def _seed_inbox(store: CategorizedMemoryStore) -> tuple[str, str]:
    rule = store.add_item(MemoryCategory.INBOX, "rule: always run tests", "agent")
    note = store.add_item(MemoryCategory.INBOX, "note: tool failed", "agent")
    return rule.id, note.id


def test_apply_triage_moves_and_marks(tmp_path) -> None:
    store = CategorizedMemoryStore(tmp_path / "mem.db")
    rule_id, note_id = _seed_inbox(store)
    inbox_items = store.list_items(MemoryCategory.INBOX, limit=10).items
    now = datetime(2025, 1, 1, tzinfo=UTC)
    plan = triage_preview(inbox_items, now=now, plan_id="plan-apply")

    result = apply_triage(
        store,
        plan,
        policy=TriageApplyPolicy(allow_dangerous=True, keep_inbox=True),
        now=now,
    )
    assert result.created_item_ids
    assert set(result.marked_inbox_ids) == {rule_id, note_id}

    rule_item = store.list_items(MemoryCategory.RULES, limit=10).items[0]
    rule_source: MemorySource = rule_item.source
    assert rule_source == "triage"
    assert rule_item.triaged_from == rule_id
    assert rule_item.triaged_at == now
    assert now <= rule_item.created_at <= now + timedelta(microseconds=10)

    note_item = store.list_items(MemoryCategory.NOTES, limit=10).items[0]
    note_source: MemorySource = note_item.source
    assert note_source == "triage"
    assert note_item.triaged_from == note_id
    assert note_item.triaged_at == now
    assert now <= note_item.created_at <= now + timedelta(microseconds=10)

    original_rule = store.get_item(rule_id)
    original_note = store.get_item(note_id)
    assert original_rule is not None and original_rule.triaged_at == now
    assert original_note is not None and original_note.triaged_at == now


def test_undo_last_triage_restores(tmp_path) -> None:
    store = CategorizedMemoryStore(tmp_path / "mem.db")
    rule_id, note_id = _seed_inbox(store)
    inbox_items = store.list_items(MemoryCategory.INBOX, limit=10).items
    now = datetime(2025, 1, 2, tzinfo=UTC)
    plan = triage_preview(inbox_items, now=now, plan_id="plan-undo")
    apply_triage(
        store,
        plan,
        policy=TriageApplyPolicy(allow_dangerous=True, keep_inbox=True),
        now=now,
    )

    undo = undo_last_triage(store)
    assert undo.triaged_at == now
    assert store.list_items(MemoryCategory.RULES, limit=10).items == []
    assert store.list_items(MemoryCategory.NOTES, limit=10).items == []

    inbox_ids = {item.id for item in store.list_items(MemoryCategory.INBOX, limit=10).items}
    assert inbox_ids == {rule_id, note_id}


def test_apply_triage_blocks_dangerous_without_flag(tmp_path) -> None:
    store = CategorizedMemoryStore(tmp_path / "mem.db")
    _seed_inbox(store)
    inbox_items = store.list_items(MemoryCategory.INBOX, limit=10).items
    plan = triage_preview(inbox_items, plan_id="plan-block")

    with pytest.raises(ValueError, match="allow_dangerous"):
        apply_triage(store, plan, policy=TriageApplyPolicy())

    assert store.list_items(MemoryCategory.RULES, limit=10).items == []
    for item in store.list_items(MemoryCategory.INBOX, limit=10).items:
        assert item.triaged_at is None


def test_apply_triage_idempotent(tmp_path) -> None:
    store = CategorizedMemoryStore(tmp_path / "mem.db")
    _seed_inbox(store)
    inbox_items = store.list_items(MemoryCategory.INBOX, limit=10).items
    now = datetime(2025, 1, 3, tzinfo=UTC)
    plan = triage_preview(inbox_items, now=now, plan_id="plan-repeat")

    apply_triage(
        store,
        plan,
        policy=TriageApplyPolicy(allow_dangerous=True, keep_inbox=True),
        now=now,
    )
    apply_triage(
        store,
        plan,
        policy=TriageApplyPolicy(allow_dangerous=True, keep_inbox=True),
        now=now,
    )

    assert len(store.list_items(MemoryCategory.RULES, limit=10).items) == 1
    assert len(store.list_items(MemoryCategory.NOTES, limit=10).items) == 1
