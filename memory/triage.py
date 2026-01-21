from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from memory.categorized_memory_store import CategorizedMemoryStore
from shared.memory_category_models import MemoryCategory, MemoryItem
from shared.models import JSONValue

TriageConfidence = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class TriageSuggestion:
    item_id: str
    proposed_category: MemoryCategory
    confidence: TriageConfidence
    reason: str
    dangerous: bool

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("TriageSuggestion.item_id должен быть непустым")
        if not self.reason.strip():
            raise ValueError("TriageSuggestion.reason должен быть непустым")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "item_id": self.item_id,
            "proposed_category": self.proposed_category.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "dangerous": self.dangerous,
        }


@dataclass(frozen=True)
class TriagePlan:
    plan_id: str
    created_at: datetime
    source_category: MemoryCategory = MemoryCategory.INBOX
    suggestions: list[TriageSuggestion] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("TriagePlan.plan_id должен быть непустым")
        if self.source_category != MemoryCategory.INBOX:
            raise ValueError("TriagePlan.source_category должен быть inbox")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at.isoformat(),
            "source_category": self.source_category.value,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "counts": self.counts,
        }


@dataclass(frozen=True)
class TriagePolicy:
    rules_markers: tuple[str, ...] = ("rule:", "правило:", "[rule]")
    preferences_markers: tuple[str, ...] = ("preference:", "предпочтение:", "[pref]")
    glossary_markers: tuple[str, ...] = ("term:", "термин:", "определение:")
    facts_markers: tuple[str, ...] = ("fact:", "факт:")
    notes_markers: tuple[str, ...] = ("note:", "заметка:", "наблюдение:")


@dataclass(frozen=True)
class TriageApplyPolicy:
    allow_dangerous: bool = False
    keep_inbox: bool = True


@dataclass(frozen=True)
class TriageApplyResult:
    plan_id: str
    applied_at: datetime
    created_item_ids: list[str] = field(default_factory=list)
    marked_inbox_ids: list[str] = field(default_factory=list)
    deleted_inbox_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TriageUndoResult:
    triaged_at: datetime | None
    restored_inbox_ids: list[str] = field(default_factory=list)
    removed_item_ids: list[str] = field(default_factory=list)


def triage_preview(
    inbox_items: list[MemoryItem],
    policy: TriagePolicy | None = None,
    *,
    now: datetime | None = None,
    plan_id: str | None = None,
) -> TriagePlan:
    resolved_policy = policy or TriagePolicy()
    created_at = _ensure_utc(now) if now else datetime.now(UTC)
    plan_id = plan_id or uuid.uuid4().hex

    suggestions: list[TriageSuggestion] = []
    counts: dict[str, int] = {category.value: 0 for category in MemoryCategory}

    for item in inbox_items:
        suggestion: TriageSuggestion = _suggest_for_item(item, resolved_policy)
        suggestions.append(suggestion)
        counts[suggestion.proposed_category.value] += 1

    return TriagePlan(
        plan_id=plan_id,
        created_at=created_at,
        source_category=MemoryCategory.INBOX,
        suggestions=suggestions,
        counts=counts,
    )


def apply_triage(
    store: CategorizedMemoryStore,
    plan: TriagePlan,
    overrides: dict[str, MemoryCategory | str] | None = None,
    policy: TriageApplyPolicy | None = None,
    *,
    now: datetime | None = None,
) -> TriageApplyResult:
    if plan.source_category != MemoryCategory.INBOX:
        raise ValueError("apply_triage ожидает план из inbox")
    resolved_policy = policy or TriageApplyPolicy()
    overrides = overrides or {}
    applied_at = _ensure_utc(now) if now else datetime.now(UTC)
    created_item_ids: list[str] = []
    marked_inbox_ids: list[str] = []
    deleted_inbox_ids: list[str] = []

    targets: list[tuple[TriageSuggestion, MemoryCategory]] = []
    for suggestion in plan.suggestions:
        target = overrides.get(suggestion.item_id, suggestion.proposed_category)
        target_category = _resolve_category(target)
        targets.append((suggestion, target_category))
    if not resolved_policy.allow_dangerous:
        for _, target_category in targets:
            if target_category in (MemoryCategory.RULES, MemoryCategory.PREFERENCES):
                raise ValueError("Перенос в rules/preferences требует allow_dangerous=True")

    seq = 0
    for suggestion, target_category in targets:
        if target_category == MemoryCategory.INBOX:
            continue
        original = store.get_item(suggestion.item_id)
        if original is None:
            raise ValueError(f"MemoryItem {suggestion.item_id} не найден")
        if original.triaged_at is not None:
            continue
        created_at = applied_at + timedelta(microseconds=seq)
        seq += 1
        new_item = store.add_item(
            target_category,
            original.content,
            "triage",
            meta=dict(original.meta),
            title=original.title,
            tags=list(original.tags),
            created_at=created_at,
        )
        store.update_item(
            new_item.id,
            triaged_from=original.id,
            triaged_at=applied_at,
        )
        created_item_ids.append(new_item.id)
        if resolved_policy.keep_inbox:
            store.update_item(
                original.id,
                triaged_from=None,
                triaged_at=applied_at,
            )
            marked_inbox_ids.append(original.id)
        else:
            if store.delete_item(original.id):
                deleted_inbox_ids.append(original.id)

    return TriageApplyResult(
        plan_id=plan.plan_id,
        applied_at=applied_at,
        created_item_ids=created_item_ids,
        marked_inbox_ids=marked_inbox_ids,
        deleted_inbox_ids=deleted_inbox_ids,
    )


def undo_last_triage(store: CategorizedMemoryStore) -> TriageUndoResult:
    all_items: list[MemoryItem] = []
    for category in MemoryCategory:
        all_items.extend(store.list_items(category, limit=10_000).items)
    latest = max(
        (item.triaged_at for item in all_items if item.triaged_at is not None),
        default=None,
    )
    if latest is None:
        return TriageUndoResult(triaged_at=None)
    restored_inbox_ids: list[str] = []
    removed_item_ids: list[str] = []
    for item in all_items:
        if item.triaged_at != latest:
            continue
        if item.source == "triage":
            if store.delete_item(item.id):
                removed_item_ids.append(item.id)
    return TriageUndoResult(
        triaged_at=latest,
        restored_inbox_ids=restored_inbox_ids,
        removed_item_ids=removed_item_ids,
    )


def _suggest_for_item(item: MemoryItem, policy: TriagePolicy) -> TriageSuggestion:
    text = _normalize_text(item)
    rule_marker = _match_marker(text, policy.rules_markers)
    if rule_marker:
        return _build_suggestion(item, MemoryCategory.RULES, "high", f"marker:{rule_marker}")

    pref_marker = _match_marker(text, policy.preferences_markers)
    if pref_marker:
        return _build_suggestion(item, MemoryCategory.PREFERENCES, "high", f"marker:{pref_marker}")

    glossary_marker = _match_marker(text, policy.glossary_markers)
    if glossary_marker:
        return _build_suggestion(
            item, MemoryCategory.GLOSSARY, "medium", f"marker:{glossary_marker}"
        )

    facts_marker = _match_marker(text, policy.facts_markers)
    if facts_marker:
        return _build_suggestion(item, MemoryCategory.FACTS, "medium", f"marker:{facts_marker}")

    notes_marker = _match_marker(text, policy.notes_markers)
    if notes_marker:
        return _build_suggestion(item, MemoryCategory.NOTES, "medium", f"marker:{notes_marker}")

    return _build_suggestion(item, MemoryCategory.INBOX, "low", "no_match")


def _build_suggestion(
    item: MemoryItem,
    category: MemoryCategory,
    confidence: TriageConfidence,
    reason: str,
) -> TriageSuggestion:
    dangerous = category in (MemoryCategory.RULES, MemoryCategory.PREFERENCES)
    if dangerous and confidence == "high":
        confidence = "medium"
    return TriageSuggestion(
        item_id=item.id,
        proposed_category=category,
        confidence=confidence,
        reason=reason,
        dangerous=dangerous,
    )


def _normalize_text(item: MemoryItem) -> str:
    parts = [item.title or "", item.content, " ".join(item.tags)]
    return " ".join(part for part in parts if part).strip().lower()


def _match_marker(text: str, markers: tuple[str, ...]) -> str | None:
    for marker in markers:
        if text.startswith(marker):
            return marker
    return None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _resolve_category(value: MemoryCategory | str) -> MemoryCategory:
    if isinstance(value, MemoryCategory):
        return value
    try:
        return MemoryCategory(value)
    except ValueError as exc:
        raise ValueError(f"Неизвестная категория памяти: {value}") from exc
