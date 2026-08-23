from __future__ import annotations

import json
import uuid

from memory.canonical_atom_store import CanonicalAtomStore
from shared.canonical_atom_models import AtomStatus, CanonicalAtom, Claim, utc_now_iso
from shared.models import JSONValue


class CanonicalAggregator:
    def __init__(self, store: CanonicalAtomStore) -> None:
        self._store = store

    def upsert_claim(self, claim: Claim) -> CanonicalAtom:
        existing = self._store.get_by_stable_key(claim.stable_key)
        atom = self._merge_claim(existing, claim)
        return self._store.upsert(atom)

    def upsert_claims(self, claims: list[Claim]) -> list[CanonicalAtom]:
        with self._store.conn.transaction(immediate=True):
            pending: dict[str, CanonicalAtom] = {}
            order: list[str] = []
            for claim in claims:
                existing = pending.get(claim.stable_key)
                if existing is None:
                    existing = self._store.get_by_stable_key(claim.stable_key)
                    order.append(claim.stable_key)
                pending[claim.stable_key] = self._merge_claim(existing, claim)
            return self._store.upsert_many([pending[stable_key] for stable_key in order])

    @staticmethod
    def _merge_claim(existing: CanonicalAtom | None, claim: Claim) -> CanonicalAtom:
        if existing is None:
            return CanonicalAtom(
                atom_id=f"atom-{uuid.uuid4().hex}",
                stable_key=claim.stable_key,
                claim_type=claim.claim_type,
                value_json=claim.value_json,
                confidence=claim.confidence,
                support_count=1,
                contradict_count=0,
                last_seen_at=claim.created_at,
                status=AtomStatus.ACTIVE,
                summary_text=claim.summary_text,
            )

        if _values_equal(existing.value_json, claim.value_json):
            boosted_confidence = min(1.0, max(existing.confidence, claim.confidence) + 0.03)
            next_status = (
                AtomStatus.ACTIVE if existing.status is AtomStatus.CONFLICT else existing.status
            )
            return CanonicalAtom(
                atom_id=existing.atom_id,
                stable_key=existing.stable_key,
                claim_type=existing.claim_type,
                value_json=existing.value_json,
                confidence=boosted_confidence,
                support_count=existing.support_count + 1,
                contradict_count=existing.contradict_count,
                last_seen_at=claim.created_at,
                status=next_status,
                summary_text=_deterministic_summary(existing.stable_key, existing.value_json),
            )

        downgraded_confidence = max(0.0, min(existing.confidence, claim.confidence) - 0.15)
        return CanonicalAtom(
            atom_id=existing.atom_id,
            stable_key=existing.stable_key,
            claim_type=existing.claim_type,
            value_json=existing.value_json,
            confidence=downgraded_confidence,
            support_count=existing.support_count,
            contradict_count=existing.contradict_count + 1,
            last_seen_at=claim.created_at,
            status=AtomStatus.CONFLICT,
            summary_text=_deterministic_summary(existing.stable_key, existing.value_json),
        )

    def resolve_conflict(
        self,
        *,
        stable_key: str,
        action: str,
        value_json: JSONValue | None = None,
        summary_text: str | None = None,
    ) -> CanonicalAtom | None:
        current = self._store.get_by_stable_key(stable_key)
        if current is None:
            return None
        if action == "activate":
            return self._store.resolve_conflict(
                stable_key=stable_key,
                resolution=AtomStatus.ACTIVE,
                value_json=current.value_json,
                summary_text=summary_text
                or _deterministic_summary(current.stable_key, current.value_json),
            )
        if action == "deprecate":
            return self._store.resolve_conflict(
                stable_key=stable_key,
                resolution=AtomStatus.DEPRECATED,
                value_json=current.value_json,
                summary_text=summary_text or current.summary_text,
            )
        if action == "set_value":
            if value_json is None:
                raise ValueError("value_json обязателен для action=set_value")
            return self._store.resolve_conflict(
                stable_key=stable_key,
                resolution=AtomStatus.ACTIVE,
                value_json=value_json,
                summary_text=summary_text or _deterministic_summary(current.stable_key, value_json),
            )
        raise ValueError(f"Неизвестный action: {action}")


def _values_equal(left: object, right: object) -> bool:
    return json.dumps(left, ensure_ascii=False, sort_keys=True) == json.dumps(
        right,
        ensure_ascii=False,
        sort_keys=True,
    )


def _deterministic_summary(stable_key: str, value_json: object) -> str:
    compact_value = json.dumps(value_json, ensure_ascii=False, sort_keys=True)
    if len(compact_value) > 200:
        compact_value = compact_value[:200]
    return f"{stable_key}={compact_value}"


def touch_last_seen_iso() -> str:
    return utc_now_iso()
