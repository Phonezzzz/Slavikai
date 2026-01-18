from __future__ import annotations

from datetime import UTC, datetime

from memory.triage import triage_preview
from shared.memory_category_models import MemoryCategory, MemoryItem


def _make_item(item_id: str, content: str, *, title: str | None = None) -> MemoryItem:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    return MemoryItem(
        id=item_id,
        category=MemoryCategory.INBOX,
        created_at=now,
        updated_at=None,
        title=title,
        content=content,
        tags=[],
        source="agent",
        fingerprint=f"fp-{item_id}",
        meta={},
        triaged_from=None,
        triaged_at=None,
    )


def test_triage_preview_stable_plan() -> None:
    items = [
        _make_item("1", "rule: always run tests"),
        _make_item("2", "preference: concise replies"),
        _make_item("3", "term: MWV означает Manager Worker Verifier"),
        _make_item("4", "note: tool failed in workspace"),
        _make_item("5", "random text without markers"),
    ]
    now = datetime(2025, 2, 1, tzinfo=UTC)
    plan = triage_preview(items, now=now, plan_id="plan-1")

    assert plan.plan_id == "plan-1"
    assert plan.created_at == now
    assert plan.source_category is MemoryCategory.INBOX
    assert plan.counts[MemoryCategory.RULES.value] == 1
    assert plan.counts[MemoryCategory.PREFERENCES.value] == 1
    assert plan.counts[MemoryCategory.GLOSSARY.value] == 1
    assert plan.counts[MemoryCategory.NOTES.value] == 1
    assert plan.counts[MemoryCategory.INBOX.value] == 1

    suggestions = plan.suggestions
    assert suggestions[0].proposed_category is MemoryCategory.RULES
    assert suggestions[0].confidence == "medium"
    assert suggestions[0].dangerous is True
    assert suggestions[1].proposed_category is MemoryCategory.PREFERENCES
    assert suggestions[1].confidence == "medium"
    assert suggestions[1].dangerous is True
    assert suggestions[2].proposed_category is MemoryCategory.GLOSSARY
    assert suggestions[2].confidence == "medium"
    assert suggestions[2].dangerous is False
    assert suggestions[3].proposed_category is MemoryCategory.NOTES
    assert suggestions[3].confidence == "medium"
    assert suggestions[4].proposed_category is MemoryCategory.INBOX
    assert suggestions[4].confidence == "low"

    plan_repeat = triage_preview(items, now=now, plan_id="plan-1")
    assert plan_repeat.suggestions == plan.suggestions
    assert plan_repeat.counts == plan.counts


def test_rules_preferences_not_high_without_marker() -> None:
    items = [
        _make_item("1", "always run tests"),
        _make_item("2", "i prefer tea"),
    ]
    now = datetime(2025, 3, 1, tzinfo=UTC)
    plan = triage_preview(items, now=now, plan_id="plan-2")

    for suggestion in plan.suggestions:
        assert suggestion.proposed_category is MemoryCategory.INBOX
        assert suggestion.confidence != "high"
        assert suggestion.dangerous is False
