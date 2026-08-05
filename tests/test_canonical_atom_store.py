from __future__ import annotations

import sqlite3

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


def test_canonical_atom_store_pinned_roundtrip_and_listing(tmp_path) -> None:
    db_path = tmp_path / "canonical_atoms.db"
    store = CanonicalAtomStore(str(db_path))

    store.create_atom(
        atom_id="atom-pin",
        stable_key="policy:avoid_emoji",
        claim_type=ClaimType.POLICY,
        value_json={"rule": "avoid emoji"},
        confidence=0.9,
        summary_text="policy:avoid_emoji",
    )

    assert store.list_pinned() == []
    assert store.set_pinned("policy:avoid_emoji", True) is True
    pinned = store.list_pinned()
    assert len(pinned) == 1
    assert pinned[0].stable_key == "policy:avoid_emoji"
    assert pinned[0].pinned is True

    store.create_atom(
        atom_id="atom-pin-updated",
        stable_key="policy:avoid_emoji",
        claim_type=ClaimType.POLICY,
        value_json={"rule": "avoid emoji always"},
        confidence=0.91,
        summary_text="policy:avoid_emoji=updated",
    )

    updated = store.get_by_stable_key("policy:avoid_emoji")
    assert updated is not None
    assert updated.pinned is True

    assert store.set_pinned("policy:avoid_emoji", False) is True
    assert store.list_pinned() == []


def test_canonical_atom_store_list_atoms_filters_by_stable_key_prefix(tmp_path) -> None:
    db_path = tmp_path / "canonical_atoms.db"
    store = CanonicalAtomStore(str(db_path))

    store.create_atom(
        atom_id="atom-session",
        stable_key="session:20260509_120000",
        claim_type=ClaimType.FACT,
        value_json={"text": "session summary"},
        confidence=0.9,
        summary_text="session summary 20260509_120000",
    )
    store.create_atom(
        atom_id="atom-fact",
        stable_key="fact:project_stack",
        claim_type=ClaimType.FACT,
        value_json={"text": "python"},
        confidence=0.8,
        summary_text="project stack",
    )

    session_atoms = store.list_atoms(
        claim_types={ClaimType.FACT},
        stable_key_prefix="session:",
    )

    assert [atom.stable_key for atom in session_atoms] == ["session:20260509_120000"]


def test_canonical_atom_store_migrates_existing_db_with_pinned_column(tmp_path) -> None:
    db_path = tmp_path / "canonical_atoms.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE canonical_atom (
            atom_id TEXT PRIMARY KEY,
            stable_key TEXT NOT NULL UNIQUE,
            claim_type TEXT NOT NULL,
            value_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            support_count INTEGER NOT NULL,
            contradict_count INTEGER NOT NULL,
            last_seen_at TEXT NOT NULL,
            status TEXT NOT NULL,
            summary_text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO canonical_atom VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "atom-old",
            "fact:old",
            "fact",
            '{"text":"old"}',
            0.7,
            1,
            0,
            utc_now_iso(),
            "active",
            "fact:old",
        ),
    )
    conn.commit()
    conn.close()

    store = CanonicalAtomStore(str(db_path))
    columns = {
        row["name"] for row in store.conn.execute("PRAGMA table_info(canonical_atom)").fetchall()
    }
    assert "pinned" in columns
    atom = store.get_by_stable_key("fact:old")
    assert atom is not None
    assert atom.pinned is False


def test_delete_atom_marks_deprecated_excluded_from_list(tmp_path) -> None:
    db_path = tmp_path / "atoms.db"
    store = CanonicalAtomStore(str(db_path))

    store.create_atom(
        atom_id="a-del-1",
        stable_key="fact:to_delete",
        claim_type=ClaimType.FACT,
        value_json={"text": "secret"},
        confidence=0.9,
        summary_text="fact:to_delete",
        support_count=1,
        contradict_count=0,
        status=AtomStatus.ACTIVE,
        last_seen_at=utc_now_iso(),
    )
    assert store.get_by_stable_key("fact:to_delete") is not None

    deleted = store.delete_atom("fact:to_delete")
    assert deleted is not None
    assert deleted.status == AtomStatus.DEPRECATED

    active = store.list_atoms(statuses={AtomStatus.ACTIVE})
    assert all(a.stable_key != "fact:to_delete" for a in active)


def test_update_atom_new_value_replaces_old(tmp_path) -> None:
    db_path = tmp_path / "atoms.db"
    store = CanonicalAtomStore(str(db_path))

    store.create_atom(
        atom_id="a-upd-1",
        stable_key="pref:lang",
        claim_type=ClaimType.PREFERENCE,
        value_json={"value": "python"},
        confidence=0.8,
        summary_text="pref:lang=python",
        support_count=1,
        contradict_count=0,
        status=AtomStatus.ACTIVE,
        last_seen_at=utc_now_iso(),
    )

    store.create_atom(
        atom_id="a-upd-2",
        stable_key="pref:lang",
        claim_type=ClaimType.PREFERENCE,
        value_json={"value": "rust"},
        confidence=0.9,
        summary_text="pref:lang=rust",
        support_count=2,
        contradict_count=0,
        status=AtomStatus.ACTIVE,
        last_seen_at=utc_now_iso(),
    )

    updated = store.get_by_stable_key("pref:lang")
    assert updated is not None
    assert updated.value_json == {"value": "rust"}
    assert updated.confidence == 0.9
