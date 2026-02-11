from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from memory.canonical_atom_store import CanonicalAtomStore
from memory.vector_index import VectorIndex
from shared.canonical_atom_models import AtomStatus, CanonicalAtom, ClaimType
from shared.models import JSONValue


@dataclass(frozen=True)
class RetrievalConfig:
    min_confidence: float = 0.45
    top_k: int = 8
    max_context_chars: int = 1800
    recency_days: int = 365


_DEFAULT_CHAT_TYPES = {
    ClaimType.PREFERENCE,
    ClaimType.POLICY,
    ClaimType.FACT,
    ClaimType.GOAL,
    ClaimType.CONSTRAINT,
    ClaimType.ENVIRONMENT,
}

_DEFAULT_MWV_TYPES = {
    ClaimType.PREFERENCE,
    ClaimType.POLICY,
    ClaimType.CONSTRAINT,
    ClaimType.ENVIRONMENT,
    ClaimType.DECISION,
    ClaimType.FACT,
}


def filter_atoms(
    atoms: list[CanonicalAtom],
    *,
    min_confidence: float,
    allowed_types: set[ClaimType],
    include_conflicts: bool,
    recency_days: int,
    now: datetime | None = None,
) -> list[CanonicalAtom]:
    reference = now or datetime.now(UTC)
    cutoff = reference - timedelta(days=recency_days)
    allowed_statuses = {AtomStatus.ACTIVE}
    if include_conflicts:
        allowed_statuses.add(AtomStatus.CONFLICT)
    filtered: list[CanonicalAtom] = []
    for atom in atoms:
        if atom.status not in allowed_statuses:
            continue
        if atom.claim_type not in allowed_types:
            continue
        if atom.confidence < min_confidence:
            continue
        try:
            seen_at = datetime.fromisoformat(atom.last_seen_at)
        except ValueError:
            continue
        if seen_at.tzinfo is None:
            continue
        if seen_at < cutoff:
            continue
        filtered.append(atom)
    filtered.sort(
        key=lambda item: (item.confidence, item.last_seen_at, item.support_count, item.stable_key),
        reverse=True,
    )
    return filtered


def rank_atoms(
    query: str,
    atoms: list[CanonicalAtom],
    *,
    vector_index: VectorIndex,
    namespace: str,
    top_k: int,
) -> list[CanonicalAtom]:
    if not atoms:
        return []
    key_to_atom = {atom.stable_key: atom for atom in atoms}
    ranked: list[CanonicalAtom] = []
    seen: set[str] = set()
    try:
        semantic = vector_index.search(query, namespace=namespace, top_k=max(top_k * 3, top_k))
    except Exception:  # noqa: BLE001
        semantic = []

    for result in semantic:
        atom = key_to_atom.get(result.path)
        if atom is None or atom.stable_key in seen:
            continue
        ranked.append(atom)
        seen.add(atom.stable_key)
        if len(ranked) >= top_k:
            return ranked

    for atom in atoms:
        if atom.stable_key in seen:
            continue
        ranked.append(atom)
        seen.add(atom.stable_key)
        if len(ranked) >= top_k:
            break
    return ranked


def pack_context(
    atoms: list[CanonicalAtom],
    *,
    max_chars: int,
) -> tuple[str, list[dict[str, JSONValue]]]:
    lines: list[str] = []
    items: list[dict[str, JSONValue]] = []
    used = 0
    for atom in atoms:
        compact_value = json.dumps(atom.value_json, ensure_ascii=False, sort_keys=True)
        line = (
            f"- [{atom.claim_type.value}] {atom.stable_key}: {compact_value} "
            f"(confidence={atom.confidence:.2f}, support={atom.support_count}, "
            f"contradict={atom.contradict_count}, status={atom.status.value})"
        )
        line_len = len(line) + 1
        if used + line_len > max_chars:
            break
        lines.append(line)
        used += line_len
        items.append(
            {
                "stable_key": atom.stable_key,
                "claim_type": atom.claim_type.value,
                "value_json": atom.value_json,
                "confidence": atom.confidence,
                "status": atom.status.value,
                "support_count": atom.support_count,
                "contradict_count": atom.contradict_count,
                "last_seen_at": atom.last_seen_at,
                "summary_text": atom.summary_text,
            }
        )
    return "\n".join(lines), items


def build_memory_capsule(
    *,
    query: str,
    store: CanonicalAtomStore,
    vector_index: VectorIndex,
    for_mwv: bool,
    config: RetrievalConfig,
    include_conflicts: bool = False,
) -> dict[str, JSONValue]:
    allowed_types = _DEFAULT_MWV_TYPES if for_mwv else _DEFAULT_CHAT_TYPES
    all_atoms = store.list_atoms(statuses={AtomStatus.ACTIVE, AtomStatus.CONFLICT}, limit=400)
    filtered = filter_atoms(
        all_atoms,
        min_confidence=config.min_confidence,
        allowed_types=allowed_types,
        include_conflicts=include_conflicts,
        recency_days=config.recency_days,
    )
    ranked = rank_atoms(
        query,
        filtered,
        vector_index=vector_index,
        namespace="atoms",
        top_k=config.top_k,
    )
    packed_text, packed_items = pack_context(ranked, max_chars=config.max_context_chars)
    return {
        "query": query,
        "items": packed_items,
        "count": len(packed_items),
        "text": packed_text,
    }
