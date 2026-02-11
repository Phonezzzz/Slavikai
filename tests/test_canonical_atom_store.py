from __future__ import annotations

from memory.canonical_atom_store import CanonicalAtomStore
from shared.canonical_atom_models import AtomStatus, ClaimType, utc_now_iso


def test_canonical_atom_store_create_read_update(tmp_path) -> None:
    db_path = tmp_path / "canonical_atoms.db"
    store = CanonicalAtomStore(str(db_path))

    created = store.create_atom(
        atom_id="atom-1",
        stable_key="preference:response_length",
        claim_type=ClaimType.PREFERENCE,
        value_json={"value": "short"},
        confidence=0.75,
        summary_text="preference:response_length=short",
        support_count=1,
        contradict_count=0,
        status=AtomStatus.ACTIVE,
        last_seen_at=utc_now_iso(),
    )
    fetched = store.get_by_stable_key("preference:response_length")
    assert fetched is not None
    assert fetched.atom_id == created.atom_id
    assert fetched.value_json == {"value": "short"}

    updated = store.create_atom(
        atom_id="atom-2",
        stable_key="preference:response_length",
        claim_type=ClaimType.PREFERENCE,
        value_json={"value": "concise"},
        confidence=0.8,
        summary_text="preference:response_length=concise",
        support_count=2,
        contradict_count=1,
        status=AtomStatus.CONFLICT,
        last_seen_at=utc_now_iso(),
    )
    assert updated.atom_id == "atom-2"
    assert updated.status is AtomStatus.CONFLICT
    assert updated.value_json == {"value": "concise"}

    rows = store.conn.execute("SELECT COUNT(*) FROM canonical_atom").fetchone()
    assert rows is not None
    assert int(rows[0]) == 1


def test_canonical_atom_store_value_json_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "canonical_atoms.db"
    store = CanonicalAtomStore(str(db_path))

    payload = {
        "nested": {"bool": True, "count": 3, "list": ["a", "b", {"k": "v"}]},
        "text": "hello",
    }
    store.create_atom(
        atom_id="atom-rt",
        stable_key="fact:runtime_context",
        claim_type=ClaimType.FACT,
        value_json=payload,
        confidence=0.6,
        summary_text="fact:runtime_context",
        last_seen_at=utc_now_iso(),
    )

    fetched = store.get_by_id("atom-rt")
    assert fetched is not None
    assert fetched.value_json == payload


def test_canonical_atom_store_timestamps_are_utc_iso(tmp_path) -> None:
    db_path = tmp_path / "canonical_atoms.db"
    store = CanonicalAtomStore(str(db_path))

    atom = store.create_atom(
        atom_id="atom-ts",
        stable_key="environment:os",
        claim_type=ClaimType.ENVIRONMENT,
        value_json={"value": "linux"},
        confidence=0.7,
        summary_text="environment:os=linux",
        last_seen_at=utc_now_iso(),
    )
    assert "T" in atom.last_seen_at
    assert atom.last_seen_at.endswith("+00:00")
