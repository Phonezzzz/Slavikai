from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from memory.categorized_memory_store import ListPage
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


class TriageStoreProtocol(Protocol):
    def list_items(
        self,
        category: MemoryCategory,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ListPage: ...

    def get_item(self, item_id: str) -> MemoryItem | None: ...

    def update_item(self, item_id: str, **kwargs: object) -> MemoryItem | None: ...


@dataclass(frozen=True)
class TriageApplyResult:
    tx_id: str
    plan_id: str
    applied: list[str] = field(default_factory=list)
    blocked_dangerous: list[str] = field(default_factory=list)
    skipped_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "tx_id": self.tx_id,
            "plan_id": self.plan_id,
            "applied": list(self.applied),
            "blocked_dangerous": list(self.blocked_dangerous),
            "skipped_missing": list(self.skipped_missing),
        }


@dataclass(frozen=True)
class TriageUndoResult:
    tx_id: str
    restored: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "tx_id": self.tx_id,
            "restored": list(self.restored),
        }


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


def triage_apply(
    *,
    store: TriageStoreProtocol,
    plan: TriagePlan,
    allow_dangerous: bool = False,
    now: datetime | None = None,
    tx_id: str | None = None,
) -> TriageApplyResult:
    timestamp = _ensure_utc(now) if now else datetime.now(UTC)
    resolved_tx_id = tx_id or f"tx-{uuid.uuid4().hex}"
    applied: list[str] = []
    blocked_dangerous: list[str] = []
    skipped_missing: list[str] = []
    for suggestion in plan.suggestions:
        if suggestion.proposed_category == MemoryCategory.INBOX:
            continue
        if suggestion.dangerous and not allow_dangerous:
            blocked_dangerous.append(suggestion.item_id)
            continue
        item = store.get_item(suggestion.item_id)
        if item is None:
            skipped_missing.append(suggestion.item_id)
            continue
        if item.category != MemoryCategory.INBOX:
            skipped_missing.append(suggestion.item_id)
            continue
        next_meta = dict(item.meta)
        next_meta["triage_tx_id"] = resolved_tx_id
        next_meta["triage_plan_id"] = plan.plan_id
        next_meta["triage_applied_at"] = timestamp.isoformat()
        updated = store.update_item(
            suggestion.item_id,
            category=suggestion.proposed_category,
            source="triage",
            meta=next_meta,
            triaged_from=MemoryCategory.INBOX.value,
            triaged_at=timestamp,
        )
        if updated is None:
            skipped_missing.append(suggestion.item_id)
            continue
        applied.append(updated.id)
    return TriageApplyResult(
        tx_id=resolved_tx_id,
        plan_id=plan.plan_id,
        applied=applied,
        blocked_dangerous=blocked_dangerous,
        skipped_missing=skipped_missing,
    )


def triage_undo(
    *,
    store: TriageStoreProtocol,
    tx_id: str,
    now: datetime | None = None,
) -> TriageUndoResult:
    resolved_tx_id = tx_id.strip()
    if not resolved_tx_id:
        raise ValueError("tx_id обязателен")
    timestamp = _ensure_utc(now) if now else datetime.now(UTC)
    restored: list[str] = []
    for item in _iter_all_items(store):
        if item.category == MemoryCategory.INBOX:
            continue
        meta = item.meta
        tx_meta = meta.get("triage_tx_id")
        if not isinstance(tx_meta, str) or tx_meta != resolved_tx_id:
            continue
        if meta.get("triage_undone_at"):
            continue
        if item.triaged_from != MemoryCategory.INBOX.value:
            continue
        next_meta = dict(meta)
        next_meta["triage_undone_at"] = timestamp.isoformat()
        updated = store.update_item(
            item.id,
            category=MemoryCategory.INBOX,
            source="triage",
            meta=next_meta,
            triaged_from=None,
            triaged_at=None,
        )
        if updated is not None:
            restored.append(updated.id)
    return TriageUndoResult(tx_id=resolved_tx_id, restored=restored)


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


def _iter_all_items(store: TriageStoreProtocol) -> list[MemoryItem]:
    items: list[MemoryItem] = []
    for category in MemoryCategory:
        cursor: str | None = None
        while True:
            page = store.list_items(category, limit=200, cursor=cursor)
            items.extend(page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
    return items
