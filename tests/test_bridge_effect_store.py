"""Bridge effect journal store (E03-T08, rev. 2 + owner conditions).

Tests are written against the schema (rev. 1 §6.2 as updated by
rev. 2): identity, trichotomy persistence, retention, gap
detection, bounded persistence. The recorder tests cover the
classification; these cover the store's own guarantees.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.bridge_effect_store import BridgeEffectStore

_OWNER = "owner"
_BOOT = "b" * 32


def _effect(*, boot_id: str = _BOOT, seq: int, kind: str = "fs.write") -> dict[str, object]:
    return {
        "seq": seq,
        "boot_id": boot_id,
        "kind": kind,
        "at": "2026-11-25T00:00:00+00:00",
        "principal_id": _OWNER,
        "run_id": None,
        "root_id": "work",
        "path": "a.txt",
        "size": 1,
        "bytes_written": 1,
        "committed": True,
        "content_sha256": "0" * 64,
    }


def _record(
    store: BridgeEffectStore,
    *,
    request_id: str,
    capability: str = "host.fs.write",
    outcome: str = "effect",
    effect: dict[str, object] | None = None,
    error_code: str | None = None,
    unknown_reason: str | None = None,
    principal_id: str | None = _OWNER,
    run_id: str | None = None,
    recorded_at: str | None = None,
) -> dict[str, object]:
    effect_json = json.dumps(effect, sort_keys=True) if effect is not None else None
    return store.record_call(
        capability=capability,
        request_id=request_id,
        principal_id=principal_id,
        run_id=run_id,
        params_json=json.dumps({"root_id": "work", "path": "a.txt"}, sort_keys=True),
        effect_json=effect_json,
        daemon_boot_id=effect["boot_id"] if isinstance(effect, dict) else None,
        daemon_seq=effect["seq"] if isinstance(effect, dict) else None,
        outcome=outcome,
        error_code=error_code,
        unknown_reason=unknown_reason,
        recorded_at=recorded_at,
    )


def _observation(*, subject: str = "a.txt", state: str = "exists") -> dict[str, object]:
    return {"subject": subject, "observed_at": "2026-11-25T00:00:00+00:00", "state": state}


def _attach(
    store: BridgeEffectStore, effect_seq: int, *, source: str = "poll-loop", **payload: object
) -> int:
    observation = _observation(**payload) if payload else _observation()
    observation_json = json.dumps(observation, sort_keys=True)
    return store.attach_observation(
        effect_seq=effect_seq,
        source=source,
        source_ref=None,
        observation_json=observation_json,
        observation_sha256=hashlib.sha256(observation_json.encode()).hexdigest(),
    )


# --- schema ------------------------------------------------------------------


def test_schema_created_v3(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
    assert tables == {
        "effect_meta",
        "effect_records",
        "effect_observations",
        "pruned_daemon_ranges",
    }


def test_v1_database_migrates_to_v3(tmp_path: Path) -> None:
    """A journal written by the v1 schema (PR #209 head) opens
    under v3: the chain v1 -> v2 (pruned-ranges table) -> v3
    (prune-safe resolution trigger) runs, existing rows are
    untouched, and recording keeps working."""
    path = tmp_path / "effects.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE effect_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE TABLE effect_records (
                seq INTEGER PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                daemon_boot_id TEXT,
                daemon_seq INTEGER,
                request_id TEXT NOT NULL,
                capability TEXT NOT NULL,
                principal_id TEXT,
                run_id TEXT,
                params_json TEXT NOT NULL,
                effect_json TEXT,
                outcome TEXT NOT NULL CHECK (outcome IN ('effect', 'no_effect', 'unknown')),
                error_code TEXT,
                unknown_reason TEXT,
                resolved_by_seq INTEGER REFERENCES effect_records(seq),
                resolved_at TEXT,
                CHECK ((daemon_boot_id IS NULL) = (daemon_seq IS NULL)),
                CHECK ((outcome = 'effect') = (effect_json IS NOT NULL)),
                CHECK ((outcome = 'unknown') = (unknown_reason IS NOT NULL))
            );
            CREATE UNIQUE INDEX effect_emission_identity
                ON effect_records (daemon_boot_id, daemon_seq) WHERE daemon_seq IS NOT NULL;
            CREATE TABLE effect_observations (
                id INTEGER PRIMARY KEY,
                effect_seq INTEGER NOT NULL REFERENCES effect_records(seq) ON DELETE CASCADE,
                attached_at TEXT NOT NULL,
                source TEXT NOT NULL,
                source_ref TEXT,
                observation_json TEXT NOT NULL,
                observation_sha256 TEXT NOT NULL,
                UNIQUE (effect_seq, source, observation_sha256)
            );
            INSERT INTO effect_meta (key, value) VALUES ('next_seq', 2);
            INSERT INTO effect_records
                (seq, recorded_at, daemon_boot_id, daemon_seq, request_id, capability,
                 principal_id, run_id, params_json, effect_json, outcome, error_code,
                 unknown_reason)
                VALUES (1, '2026-11-25T00:00:00+00:00', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                        1, 'req-1', 'host.fs.write', 'owner', NULL,
                        '{"path": "a.txt"}', '{"kind": "fs.write"}', 'effect', NULL, NULL);
            PRAGMA user_version = 1;
            """
        )
    store = BridgeEffectStore(path)
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    record = store.get_record(1)
    assert record is not None
    assert record["daemon_seq"] == 1
    stored = _record(store, request_id="req-2", effect=_effect(seq=2))
    assert stored["seq"] == 2
    assert store.count() == 2


def test_v2_database_migrates_to_v3(tmp_path: Path) -> None:
    """A journal written by the v2 schema (PR #209 rework 1) opens
    under v3: only the immutability trigger is replaced (the
    prune may drop a resolution pointer once its target is
    pruned); existing rows — including a resolved unknown — are
    untouched, and a prune past a resolution link works on the
    migrated journal."""
    path = tmp_path / "effects.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE effect_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE TABLE effect_records (
                seq INTEGER PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                daemon_boot_id TEXT,
                daemon_seq INTEGER,
                request_id TEXT NOT NULL,
                capability TEXT NOT NULL,
                principal_id TEXT,
                run_id TEXT,
                params_json TEXT NOT NULL,
                effect_json TEXT,
                outcome TEXT NOT NULL CHECK (outcome IN ('effect', 'no_effect', 'unknown')),
                error_code TEXT,
                unknown_reason TEXT,
                resolved_by_seq INTEGER REFERENCES effect_records(seq),
                resolved_at TEXT,
                CHECK ((daemon_boot_id IS NULL) = (daemon_seq IS NULL)),
                CHECK ((outcome = 'effect') = (effect_json IS NOT NULL)),
                CHECK ((outcome = 'unknown') = (unknown_reason IS NOT NULL))
            );
            CREATE UNIQUE INDEX effect_emission_identity
                ON effect_records (daemon_boot_id, daemon_seq) WHERE daemon_seq IS NOT NULL;
            CREATE TABLE effect_observations (
                id INTEGER PRIMARY KEY,
                effect_seq INTEGER NOT NULL REFERENCES effect_records(seq) ON DELETE CASCADE,
                attached_at TEXT NOT NULL,
                source TEXT NOT NULL,
                source_ref TEXT,
                observation_json TEXT NOT NULL,
                observation_sha256 TEXT NOT NULL,
                UNIQUE (effect_seq, source, observation_sha256)
            );
            CREATE TABLE pruned_daemon_ranges (
                daemon_boot_id TEXT NOT NULL,
                seq_from INTEGER NOT NULL,
                seq_to INTEGER NOT NULL,
                PRIMARY KEY (daemon_boot_id, seq_from)
            );
            CREATE TRIGGER immutable_effect_record_update
            BEFORE UPDATE ON effect_records
            BEGIN SELECT RAISE(ABORT, 'effect record is immutable except resolution')
            WHERE NOT (
                OLD.outcome = 'unknown'
                AND NEW.outcome = OLD.outcome
                AND NEW.recorded_at = OLD.recorded_at
                AND NEW.daemon_boot_id IS OLD.daemon_boot_id
                AND NEW.daemon_seq IS OLD.daemon_seq
                AND NEW.request_id = OLD.request_id
                AND NEW.capability = OLD.capability
                AND NEW.principal_id IS OLD.principal_id
                AND NEW.run_id IS OLD.run_id
                AND NEW.params_json = OLD.params_json
                AND NEW.effect_json IS OLD.effect_json
                AND NEW.error_code IS OLD.error_code
                AND NEW.unknown_reason IS OLD.unknown_reason
                AND OLD.resolved_by_seq IS NULL
                AND NEW.resolved_by_seq IS NOT NULL
                AND OLD.resolved_at IS NULL
                AND NEW.resolved_at IS NOT NULL
            ); END;
            INSERT INTO effect_meta (key, value) VALUES ('next_seq', 3);
            INSERT INTO effect_records
                (seq, recorded_at, daemon_boot_id, daemon_seq, request_id, capability,
                 principal_id, run_id, params_json, effect_json, outcome, error_code,
                 unknown_reason)
                VALUES (1, '2026-11-25T00:00:00+00:00', 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                        1, 'req-1', 'host.fs.write', 'owner', NULL,
                        '{"path": "a.txt"}', '{"kind": "fs.write"}', 'effect', NULL, NULL);
            INSERT INTO effect_records
                (seq, recorded_at, daemon_boot_id, daemon_seq, request_id, capability,
                 principal_id, run_id, params_json, effect_json, outcome, error_code,
                 unknown_reason, resolved_by_seq, resolved_at)
                VALUES (2, '2026-11-25T00:01:00+00:00', NULL, NULL,
                        'req-1', 'host.fs.write', 'owner', NULL,
                        '{"path": "a.txt"}', NULL, 'unknown', NULL, 'timeout',
                        1, '2026-11-25T00:02:00+00:00');
            PRAGMA user_version = 2;
            """
        )
    store = BridgeEffectStore(path, effect_cap=2)
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    attempt = store.get_record(2)
    assert attempt is not None
    assert attempt["resolved_by_seq"] == 1
    assert attempt["resolved_at"] == "2026-11-25T00:02:00+00:00"
    # The migrated journal prunes past the resolution link.
    _record(store, request_id="req-2", effect=_effect(seq=2))
    assert store.count() == 2
    assert store.get_record(1) is None
    attempt = store.get_record(2)
    assert attempt is not None
    assert attempt["resolved_at"] == "2026-11-25T00:02:00+00:00"
    assert attempt["resolved_by_seq"] is None


def test_unknown_schema_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "effects.db"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA user_version = 99")
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(RuntimeError):
        BridgeEffectStore(path)


def test_occupied_file_without_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "effects.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(RuntimeError):
        BridgeEffectStore(path)


# --- identity and dedupe (owner condition 1) ---------------------------------


def test_record_effect_and_read_back(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    stored = _record(store, request_id="req-1", effect=_effect(seq=1))
    assert stored["seq"] == 1
    assert stored["outcome"] == "effect"
    assert stored["duplicate"] is False
    record = store.get_record(1)
    assert record is not None
    assert record["daemon_boot_id"] == _BOOT
    assert record["daemon_seq"] == 1
    assert record["outcome"] == "effect"
    assert record["effect"] == _effect(seq=1)
    assert record["resolved_by_seq"] is None
    assert record["observations"] == []


def test_emission_identity_dedupe(tmp_path: Path) -> None:
    """The same (boot_id, daemon_seq) twice is one row, not two —
    a duplicate delivery is an idempotent no-op."""
    store = BridgeEffectStore(tmp_path / "effects.db")
    first = _record(store, request_id="req-1", effect=_effect(seq=1))
    second = _record(store, request_id="req-1", effect=_effect(seq=1))
    assert first["seq"] == second["seq"] == 1
    assert second["duplicate"] is True
    assert store.count() == 1


def test_subject_is_not_a_dedupe_key(tmp_path: Path) -> None:
    """Owner condition 1: two effects with the same subject
    (same process/run) but different daemon seqs are two rows.
    The subject is correlation only."""
    store = BridgeEffectStore(tmp_path / "effects.db")
    first = _record(store, request_id="req-1", effect=_effect(seq=1))
    second = _record(store, request_id="req-2", effect=_effect(seq=2))
    assert first["seq"] != second["seq"]
    assert store.count() == 2


def test_different_boots_may_reuse_seq(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    first = _record(store, request_id="req-1", effect=_effect(boot_id="a" * 32, seq=1))
    second = _record(store, request_id="req-2", effect=_effect(boot_id="b" * 32, seq=1))
    assert first["seq"] != second["seq"]
    assert store.count() == 2


# --- unknown resolution (owner condition 2) -----------------------------------


def test_duplicate_effect_resolves_unknown(tmp_path: Path) -> None:
    """Codex round 1, finding 2: effect -> timeout (unknown) ->
    the SAME effect again. The duplicate delivery must still
    resolve the unknown row left by the lost answer — the
    confirmed effect is already in the journal, so the unknown
    cannot stay unresolved forever. Before the fix the duplicate
    branch returned early and never linked the resolution."""
    store = BridgeEffectStore(tmp_path / "effects.db")
    effect = _record(store, request_id="req-1", effect=_effect(seq=7))
    unknown = _record(store, request_id="req-1", outcome="unknown", unknown_reason="timeout")
    duplicate = _record(store, request_id="req-1", effect=_effect(seq=7))
    assert duplicate["duplicate"] is True
    assert duplicate["seq"] == effect["seq"]
    assert duplicate["resolved_unknown_seqs"] == [unknown["seq"]]
    attempt = store.get_record(int(unknown["seq"]))
    assert attempt is not None
    assert attempt["outcome"] == "unknown"  # history is not rewritten
    assert attempt["resolved_by_seq"] == effect["seq"]
    assert attempt["resolved_at"] is not None
    assert store.count() == 2  # no second effect row


def test_prune_with_resolution_reference_keeps_journal_writable(tmp_path: Path) -> None:
    """Codex round 2, finding 4: a resolution link pointing at
    the effect retention is about to prune must not wedge the
    journal. Before the fix the prune DELETE hit the FK on
    resolved_by_seq, the whole insert rolled back, and every later
    insert failed the same way (the recorder then returned
    audit_gap forever). The prune drops the pointer in the same
    transaction; the resolution itself survives in resolved_at,
    so the attempt still reads as resolved."""
    store = BridgeEffectStore(tmp_path / "effects.db", effect_cap=2)
    effect = _record(store, request_id="req-1", effect=_effect(seq=1))
    unknown = _record(store, request_id="req-1", outcome="unknown", unknown_reason="timeout")
    duplicate = _record(store, request_id="req-1", effect=_effect(seq=1))
    assert duplicate["duplicate"] is True
    assert duplicate["resolved_unknown_seqs"] == [unknown["seq"]]
    # The new effect forces a prune of the referenced effect row.
    second = _record(store, request_id="req-2", effect=_effect(seq=2))
    assert store.count() == 2
    assert store.get_record(int(effect["seq"])) is None
    attempt = store.get_record(int(unknown["seq"]))
    assert attempt is not None
    assert attempt["outcome"] == "unknown"  # history is not rewritten
    assert attempt["resolved_at"] is not None  # still resolved...
    assert attempt["resolved_by_seq"] is None  # ...but the pointer left with its target
    # The resolved unknown no longer classifies as unknown, and
    # the pruned daemon seq is not reported as a gap.
    page = store.list_effects(principal_id=_OWNER, outcome="unknown")
    assert [row["seq"] for row in page["records"]] == []
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == []
    # ...and the journal keeps accepting inserts afterwards.
    third = _record(store, request_id="req-3", effect=_effect(seq=3))
    assert store.count() == 2
    assert third["seq"] != second["seq"]


def test_confirmed_effect_resolves_unknown(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    unknown = _record(store, request_id="req-1", outcome="unknown", unknown_reason="timeout")
    assert unknown["outcome"] == "unknown"
    effect = _record(store, request_id="req-1", effect=_effect(seq=7))
    assert effect["resolved_unknown_seqs"] == [unknown["seq"]]
    attempt = store.get_record(int(unknown["seq"]))
    assert attempt is not None
    assert attempt["outcome"] == "unknown"  # history is not rewritten
    assert attempt["resolved_by_seq"] == effect["seq"]
    assert attempt["resolved_at"] is not None
    # The resolved unknown no longer classifies as unknown in a
    # filtered listing.
    page = store.list_effects(principal_id=_OWNER, outcome="unknown")
    assert [row["seq"] for row in page["records"]] == []


def test_no_effect_does_not_resolve_unknown(tmp_path: Path) -> None:
    """A no_effect retry proves nothing about the first attempt
    (patch_mismatch after a successful apply, run_stopped after a
    concurrent STOP), so it must NOT resolve the unknown."""
    store = BridgeEffectStore(tmp_path / "effects.db")
    unknown = _record(store, request_id="req-1", outcome="unknown", unknown_reason="timeout")
    no_effect = _record(store, request_id="req-1", outcome="no_effect", error_code="patch_mismatch")
    assert no_effect["resolved_unknown_seqs"] == []
    attempt = store.get_record(int(unknown["seq"]))
    assert attempt is not None
    assert attempt["resolved_by_seq"] is None


def test_effect_record_immutable_except_resolution(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    stored = _record(store, request_id="req-1", effect=_effect(seq=1))
    with sqlite3.connect(store.path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE effect_records SET outcome = 'unknown' WHERE seq = ?",
                (stored["seq"],),
            )
    # The resolution update itself is the one permitted mutation.
    unknown = _record(store, request_id="req-2", outcome="unknown", unknown_reason="timeout")
    effect = _record(store, request_id="req-2", effect=_effect(seq=2))
    attempt = store.get_record(int(unknown["seq"]))
    assert attempt is not None
    assert attempt["resolved_by_seq"] == effect["seq"]


# --- retention and observation bounds (owner condition 3) ---------------------


def test_retention_prunes_oldest_with_cascade(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db", effect_cap=3)
    first = _record(store, request_id="req-1", effect=_effect(seq=1))
    _attach(store, int(first["seq"]))
    _record(store, request_id="req-2", effect=_effect(seq=2))
    _record(store, request_id="req-3", effect=_effect(seq=3))
    _record(store, request_id="req-4", effect=_effect(seq=4))
    assert store.count() == 3
    assert store.get_record(int(first["seq"])) is None
    # The pruned record's observations are gone in the same
    # transaction (no orphaned observation rows).
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM effect_observations").fetchone()[0] == 0


def test_per_effect_observation_cap_evicts_oldest(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db", per_effect_observation_cap=2)
    effect = _record(store, request_id="req-1", effect=_effect(seq=1))
    seq = int(effect["seq"])
    first = _attach(store, seq, state="one")
    _attach(store, seq, state="two")
    _attach(store, seq, state="three")
    record = store.get_record(seq)
    assert record is not None
    observations = record["observations"]
    assert [obs["observation"]["state"] for obs in observations] == ["two", "three"]
    assert first not in [obs["id"] for obs in observations]


def test_global_observation_cap_evicts_oldest(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db", observation_cap=2)
    first = _record(store, request_id="req-1", effect=_effect(seq=1))
    second = _record(store, request_id="req-2", effect=_effect(seq=2))
    _attach(store, int(first["seq"]), state="one")
    _attach(store, int(second["seq"]), state="two")
    _attach(store, int(second["seq"]), state="three")
    first_record = store.get_record(int(first["seq"]))
    assert first_record is not None
    assert first_record["observations"] == []
    second_record = store.get_record(int(second["seq"]))
    assert second_record is not None
    assert [obs["observation"]["state"] for obs in second_record["observations"]] == [
        "two",
        "three",
    ]


def test_observation_duplicate_is_idempotent(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    effect = _record(store, request_id="req-1", effect=_effect(seq=1))
    seq = int(effect["seq"])
    first = _attach(store, seq)
    second = _attach(store, seq)
    assert first == second
    record = store.get_record(seq)
    assert record is not None
    assert len(record["observations"]) == 1


def test_observation_for_unknown_effect_is_rejected(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    with pytest.raises(ValueError):
        _attach(store, 999)


# --- gap detection (owner condition 7) ----------------------------------------


def _old_time(seconds_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat()


def test_gap_detected_and_closed_by_late_delivery(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    _record(store, request_id="r1", effect=_effect(seq=1), recorded_at=_old_time(300))
    _record(store, request_id="r2", effect=_effect(seq=2), recorded_at=_old_time(300))
    _record(store, request_id="r4", effect=_effect(seq=4), recorded_at=_old_time(300))
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == [
        {"daemon_boot_id": _BOOT, "missing_from": 3, "missing_to": 3, "suspected": True}
    ]
    # A late delivery of seq 3 closes the hole at read time.
    _record(store, request_id="r3", effect=_effect(seq=3), recorded_at=_old_time(300))
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == []


def test_gap_within_grace_window_is_not_suspected(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    _record(store, request_id="r1", effect=_effect(seq=1), recorded_at=_old_time(300))
    _record(store, request_id="r3", effect=_effect(seq=3), recorded_at=_old_time(10))
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == [
        {"daemon_boot_id": _BOOT, "missing_from": 2, "missing_to": 2, "suspected": False}
    ]


def test_retention_floor_is_not_a_gap(tmp_path: Path) -> None:
    """Seqs pruned by retention (below the floor) are not gaps:
    the floor is min retained seq - 1, and only holes above it
    are reported."""
    store = BridgeEffectStore(tmp_path / "effects.db", effect_cap=2)
    _record(store, request_id="r1", effect=_effect(seq=1), recorded_at=_old_time(300))
    _record(store, request_id="r2", effect=_effect(seq=2), recorded_at=_old_time(300))
    _record(store, request_id="r3", effect=_effect(seq=3), recorded_at=_old_time(300))
    # seq 1 was pruned; the floor is 1. No hole is reported for it.
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == []


def test_pruned_middle_seq_is_not_a_gap(tmp_path: Path) -> None:
    """Codex round 1, finding 4: retention prunes by server seq
    (arrival order), but gaps are computed over daemon_seq. When
    effects arrive out of order, pruning can delete the MIDDLE of
    the retained daemon sequence — that is an intentional prune,
    not a lost effect, and must not be reported as a gap."""
    store = BridgeEffectStore(tmp_path / "effects.db", effect_cap=2)
    _record(store, request_id="r2", effect=_effect(seq=2), recorded_at=_old_time(300))
    _record(store, request_id="r1", effect=_effect(seq=1), recorded_at=_old_time(300))
    _record(store, request_id="r3", effect=_effect(seq=3), recorded_at=_old_time(300))
    # Daemon seq 2 arrived first, so it is the oldest server row
    # and retention pruned it; 1 and 3 remain.
    assert store.count() == 2
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == []


def test_partially_pruned_hole_reports_only_real_gap(tmp_path: Path) -> None:
    """A hole that is part pruned, part never-delivered reports
    only the never-delivered part: seq 2 was recorded and pruned,
    seq 3 was never delivered at all."""
    store = BridgeEffectStore(tmp_path / "effects.db", effect_cap=2)
    _record(store, request_id="r2", effect=_effect(seq=2), recorded_at=_old_time(300))
    _record(store, request_id="r1", effect=_effect(seq=1), recorded_at=_old_time(300))
    _record(store, request_id="r4", effect=_effect(seq=4), recorded_at=_old_time(300))
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == [
        {"daemon_boot_id": _BOOT, "missing_from": 3, "missing_to": 3, "suspected": True}
    ]


def test_pruned_ranges_stay_compact_for_in_order_delivery(tmp_path: Path) -> None:
    """In-order delivery prunes a contiguous prefix below the
    retention floor: every pruned seq is below min retained seq,
    so after the floor cleanup (Codex round 3, finding 2) no
    range row survives at all — maximally compact — and gap
    detection still reports no gaps."""
    store = BridgeEffectStore(tmp_path / "effects.db", effect_cap=3)
    for seq in range(1, 7):
        _record(store, request_id=f"r{seq}", effect=_effect(seq=seq))
    assert store.count() == 3
    with sqlite3.connect(store.path) as conn:
        rows = conn.execute(
            "SELECT daemon_boot_id, seq_from, seq_to FROM pruned_daemon_ranges"
        ).fetchall()
    assert rows == []
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == []


def test_pruned_ranges_below_retention_floor_are_dropped(tmp_path: Path) -> None:
    """Codex round 3, finding 2: with sparse daemon seqs the
    pruned-ranges table grows without bound — ranges separated by
    never-delivered seqs never merge, and are dropped only when
    the boot has no retained rows left. The finding's repro:
    effect_cap=2, 20 effects with odd daemon seqs — before the
    fix 18 singleton ranges survive next to 2 effect records.
    Ranges entirely below the retention floor (min retained daemon
    seq) are dead weight: _compute_gaps never searches below the
    floor, so they are dropped at prune time. Gap detection is
    unchanged: only the genuinely missing seq between the two
    retained effects is reported."""
    store = BridgeEffectStore(tmp_path / "effects.db", effect_cap=2)
    for seq in range(1, 40, 2):
        _record(store, request_id=f"r{seq}", effect=_effect(seq=seq), recorded_at=_old_time(300))
    assert store.count() == 2
    with sqlite3.connect(store.path) as conn:
        rows = conn.execute("SELECT seq_from, seq_to FROM pruned_daemon_ranges").fetchall()
    assert rows == []  # red before the fix: 18 singleton ranges
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == [
        {"daemon_boot_id": _BOOT, "missing_from": 38, "missing_to": 38, "suspected": True}
    ]


def test_pruned_range_above_floor_is_kept(tmp_path: Path) -> None:
    """No-regression half of finding 2: a pruned range above the
    retention floor still matters for gap detection. Out-of-order
    delivery pruned daemon seq 30 while 10 and 20 are retained,
    so the hole 21..39 must report every seq except the pruned
    30 — while the range for seq 10, which fell below the new
    floor when 40 arrived, is dropped."""
    store = BridgeEffectStore(tmp_path / "effects.db", effect_cap=2)
    _record(store, request_id="r30", effect=_effect(seq=30), recorded_at=_old_time(300))
    _record(store, request_id="r10", effect=_effect(seq=10), recorded_at=_old_time(300))
    _record(store, request_id="r20", effect=_effect(seq=20), recorded_at=_old_time(300))
    _record(store, request_id="r40", effect=_effect(seq=40), recorded_at=_old_time(300))
    assert store.count() == 2
    with sqlite3.connect(store.path) as conn:
        rows = conn.execute("SELECT seq_from, seq_to FROM pruned_daemon_ranges").fetchall()
    assert [(row[0], row[1]) for row in rows] == [(30, 30)]
    page = store.list_effects(principal_id=_OWNER, include_gaps=True)
    assert page["gaps"] == [
        {"daemon_boot_id": _BOOT, "missing_from": 21, "missing_to": 29, "suspected": True},
        {"daemon_boot_id": _BOOT, "missing_from": 31, "missing_to": 39, "suspected": True},
    ]


# --- bounded persistence (owner condition 4) ----------------------------------


def test_busy_timeout_is_bounded(tmp_path: Path) -> None:
    """A write lock held by another connection fails after the
    bounded timeout + retries — it never waits forever."""
    store = BridgeEffectStore(tmp_path / "effects.db", busy_timeout_ms=100, max_retries=1)
    lock = sqlite3.connect(store.path)
    lock.execute("BEGIN IMMEDIATE")
    lock.execute("INSERT INTO effect_meta (key, value) VALUES ('lock-holder', 1)")
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError):
            _record(store, request_id="req-1", effect=_effect(seq=1))
    finally:
        lock.rollback()
        lock.close()
    elapsed = time.monotonic() - started
    assert elapsed < 5.0  # bounded: 100 ms timeout x 2 attempts + backoff


def test_concurrent_writes_serialize(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")

    def _write(n: int) -> None:
        _record(store, request_id=f"req-{n}", effect=_effect(seq=n))

    threads = [threading.Thread(target=_write, args=(n,)) for n in range(1, 6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert store.count() == 5


# --- listing -------------------------------------------------------------------


def test_list_filters_scoping_and_pagination(tmp_path: Path) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    _record(store, request_id="r1", effect=_effect(seq=1), run_id="run-A")
    _record(store, request_id="r2", effect=_effect(seq=2), run_id="run-B")
    _record(store, request_id="r3", outcome="no_effect", error_code="invalid_params")
    _record(store, request_id="r4", effect=_effect(seq=3), principal_id="someone-else")
    page = store.list_effects(principal_id=_OWNER)
    assert [row["seq"] for row in page["records"]] == [1, 2, 3]
    assert page["next_cursor"] is None
    page = store.list_effects(principal_id=_OWNER, run_id="run-A")
    assert [row["seq"] for row in page["records"]] == [1]
    page = store.list_effects(principal_id=_OWNER, outcome="no_effect")
    assert [row["seq"] for row in page["records"]] == [3]
    page = store.list_effects(principal_id=_OWNER, limit=2)
    assert [row["seq"] for row in page["records"]] == [1, 2]
    assert page["next_cursor"] == 2
    page = store.list_effects(principal_id=_OWNER, cursor=2, limit=2)
    assert [row["seq"] for row in page["records"]] == [3]
    assert page["next_cursor"] is None
    # The other principal sees only their own records.
    page = store.list_effects(principal_id="someone-else")
    assert [row["seq"] for row in page["records"]] == [4]
