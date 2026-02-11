from __future__ import annotations

from memory.canonical_aggregator import CanonicalAggregator
from memory.canonical_atom_store import CanonicalAtomStore
from shared.canonical_atom_models import AtomStatus, Claim, ClaimType, utc_now_iso


def _claim(*, value: str, created_at: str) -> Claim:
    return Claim(
        claim_type=ClaimType.PREFERENCE,
        stable_key="preference:response_length",
        value_json={"value": value},
        confidence=0.7,
        summary_text=f"preference:response_length={value}",
        is_explicit=True,
        source_kind="chat.user_input",
        source_id="s-1",
        created_at=created_at,
    )


def test_aggregator_support_growth(tmp_path) -> None:
    store = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    aggregator = CanonicalAggregator(store)

    first = aggregator.upsert_claim(_claim(value="short", created_at=utc_now_iso()))
    second = aggregator.upsert_claim(_claim(value="short", created_at=utc_now_iso()))

    assert first.support_count == 1
    assert second.support_count == 2
    assert second.status is AtomStatus.ACTIVE


def test_aggregator_marks_conflict_on_contradiction(tmp_path) -> None:
    store = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    aggregator = CanonicalAggregator(store)

    initial = aggregator.upsert_claim(_claim(value="short", created_at=utc_now_iso()))
    conflicted = aggregator.upsert_claim(_claim(value="long", created_at=utc_now_iso()))

    assert initial.value_json == {"value": "short"}
    assert conflicted.value_json == {"value": "short"}
    assert conflicted.contradict_count == 1
    assert conflicted.status is AtomStatus.CONFLICT


def test_aggregator_updates_last_seen(tmp_path) -> None:
    store = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    aggregator = CanonicalAggregator(store)

    first_ts = "2026-01-01T00:00:00+00:00"
    second_ts = "2026-01-02T00:00:00+00:00"

    first = aggregator.upsert_claim(_claim(value="short", created_at=first_ts))
    second = aggregator.upsert_claim(_claim(value="short", created_at=second_ts))

    assert first.last_seen_at == first_ts
    assert second.last_seen_at == second_ts


def test_aggregator_resolve_conflict(tmp_path) -> None:
    store = CanonicalAtomStore(str(tmp_path / "atoms.db"))
    aggregator = CanonicalAggregator(store)

    aggregator.upsert_claim(_claim(value="short", created_at=utc_now_iso()))
    aggregator.upsert_claim(_claim(value="long", created_at=utc_now_iso()))

    resolved = aggregator.resolve_conflict(
        stable_key="preference:response_length",
        action="set_value",
        value_json={"value": "concise"},
    )
    assert resolved is not None
    assert resolved.status is AtomStatus.ACTIVE
    assert resolved.value_json == {"value": "concise"}
