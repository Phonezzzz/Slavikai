"""SQLite journal for Owner Host Bridge effects (E03-T08).

One row per bridge call the server made (mutating capabilities
only), plus the daemon effect block when the call returned one.
The journal answers "did this mutation happen, and what did it
change" — the audit trail T11's ToolGateway wiring will feed.

Schema (v3, own ``user_version`` from 1 — a separate file from
``TaskRunStore``/``ui_storage``, so migration-number collisions
are impossible by construction, C-F1). v1 -> v2 adds only the
``pruned_daemon_ranges`` table (Codex round 1, finding 4); no
existing row changes. v2 -> v3 replaces the immutability trigger
so retention may drop a resolution pointer (Codex round 2,
finding 4); ``resolved_at`` — not the pointer — is the durable
resolution marker from v3 on.

- ``effect_records``: one row per recorded call. ``seq`` is the
  server sequence (monotonic, never re-used — assigned from a
  counter in ``effect_meta``, not from ``MAX(seq)``, so pruning
  can never re-issue a seq). ``daemon_boot_id``/``daemon_seq``
  are NULL exactly when the call returned no effect block
  (``CHECK``); ``effect_json`` is non-NULL exactly when
  ``outcome = 'effect'``; ``unknown_reason`` is non-NULL exactly
  when ``outcome = 'unknown'``. The daemon emission identity
  ``(daemon_boot_id, daemon_seq)`` is unique (partial index):
  a duplicate delivery of the same effect is an idempotent
  no-op, never a second row. Rows are immutable (trigger) except
  the resolution link: an ``unknown`` row may once gain
  ``resolved_by_seq``/``resolved_at`` pointing at the confirmed
  effect that resolved it (owner condition 2) — the attempt row
  stays as history, it is never rewritten. From v3, retention
  may null ``resolved_by_seq`` when it deletes the effect the
  pointer names; the resolution itself survives in
  ``resolved_at`` — the pointer is a convenience, the timestamp
  is the marker (Codex round 2, finding 4).
- ``pruned_daemon_ranges``: the daemon seqs retention deleted,
  as merged ``[seq_from, seq_to]`` intervals per boot. Gap
  detection subtracts them: a hole fully covered by pruned
  intervals is an intentional prune, not a lost effect. In
  normal in-order delivery this is one interval per boot; the
  pathological alternating out-of-order case is bounded by the
  number of pruned daemon effects. Rows for boots with no
  retained effect rows are dropped with the prune, as are
  intervals entirely below a boot's retention floor (min
  retained daemon seq — Codex round 3, finding 2): _compute_gaps
  never searches below the floor, so they can never affect a gap
  report. A late delivery that drops the floor again can
  re-expose such a span as suspected — the same accepted
  trade-off as the boot cleanup: with no record left, the span
  is unknown, not pruned.
- ``effect_observations``: delayed observations attached to an
  effect (poll-loop re-check, manual re-check, the E05-T02 binder
  once C-V1 lands). ``UNIQUE (effect_seq, source,
  observation_sha256)`` makes attach idempotent. Bounded
  (owner condition 3): at most ``per_effect_observation_cap``
  observations per effect and ``observation_cap`` in total,
  oldest-first eviction inside the attach transaction; pruning a
  parent record deletes its observations in the same transaction
  (``ON DELETE CASCADE`` plus an explicit delete).

Bounded persistence (owner condition 4): every write runs under
``BEGIN IMMEDIATE`` with an explicit ``busy_timeout`` and bounded
retries on SQLITE_BUSY/locked — no infinite wait, no event-loop
blocking (callers use ``asyncio.to_thread``); after the retries
are exhausted the write raises and the recorder converts it to
``audit_gap`` instead of failing the bridge call.

Retention: at most ``effect_cap`` records (10 000), oldest-first
prune in the same transaction as the exceeding insert.

Gap detection (owner condition 7) is computed here over the
WHOLE effect sequence (all principals, all boots) — never over a
principal-scoped subset — and exposed to the owner only, on
unfiltered reads (see ``server/http/handlers/effects.py``).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

_SCHEMA_VERSION = 3

# Retention bounds (owner conditions 3 and 7).
DEFAULT_EFFECT_CAP = 10_000
DEFAULT_PER_EFFECT_OBSERVATION_CAP = 16
DEFAULT_OBSERVATION_CAP = 100_000
# A hole in the daemon sequence is only *suspected* once the
# record above it is older than this: a late (out-of-order)
# delivery gets a grace window first.
GAP_GRACE_S = 60.0
MAX_GAP_ENTRIES = 100

# Effect rows are immutable except the resolution link on an
# unknown row (owner condition 2) and retention dropping a
# resolution pointer (Codex round 2, finding 4). The two allowed
# transitions, everything else unchanged:
# - resolution: resolved_by_seq/resolved_at go NULL -> set
#   together, exactly once;
# - prune nulling: resolved_by_seq goes set -> NULL when
#   retention deletes the effect it points at; resolved_at is
#   untouched, so the attempt still reads as resolved.
# ``resolved_at`` — not the pointer — is the durable resolution
# marker: it is set exactly once at resolution and never cleared.
_TRIGGER_IMMUTABILITY_V3 = """CREATE TRIGGER immutable_effect_record_update
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
    AND (
        (OLD.resolved_by_seq IS NULL
         AND NEW.resolved_by_seq IS NOT NULL
         AND OLD.resolved_at IS NULL
         AND NEW.resolved_at IS NOT NULL)
        OR
        (OLD.resolved_by_seq IS NOT NULL
         AND NEW.resolved_by_seq IS NULL
         AND OLD.resolved_at IS NOT NULL
         AND NEW.resolved_at IS OLD.resolved_at)
    )
); END;"""

_PRUNED_RANGES_TABLE = """CREATE TABLE IF NOT EXISTS pruned_daemon_ranges (
        daemon_boot_id TEXT NOT NULL,
        seq_from INTEGER NOT NULL,
        seq_to INTEGER NOT NULL,
        PRIMARY KEY (daemon_boot_id, seq_from)
    );"""

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS effect_meta (
        key TEXT PRIMARY KEY,
        value INTEGER NOT NULL
    );""",
    """CREATE TABLE IF NOT EXISTS effect_records (
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
    );""",
    """CREATE UNIQUE INDEX IF NOT EXISTS effect_emission_identity
        ON effect_records (daemon_boot_id, daemon_seq) WHERE daemon_seq IS NOT NULL;""",
    """CREATE TABLE IF NOT EXISTS effect_observations (
        id INTEGER PRIMARY KEY,
        effect_seq INTEGER NOT NULL REFERENCES effect_records(seq) ON DELETE CASCADE,
        attached_at TEXT NOT NULL,
        source TEXT NOT NULL,
        source_ref TEXT,
        observation_json TEXT NOT NULL,
        observation_sha256 TEXT NOT NULL,
        UNIQUE (effect_seq, source, observation_sha256)
    );""",
    _PRUNED_RANGES_TABLE,
    # The immutability trigger (v3 definition lives in
    # _TRIGGER_IMMUTABILITY_V3 above, so the v2 -> v3 migration
    # reuses the same text; fresh databases get it directly).
    _TRIGGER_IMMUTABILITY_V3,
    """CREATE TRIGGER IF NOT EXISTS immutable_effect_observation_update
    BEFORE UPDATE ON effect_observations
    BEGIN SELECT RAISE(ABORT, 'effect observation is immutable'); END;""",
)

# v1 -> v2 (Codex round 1, finding 4): only the pruned-ranges
# table; no existing row changes.
_MIGRATION_1_TO_2 = (_PRUNED_RANGES_TABLE,)

# v2 -> v3 (Codex round 2, finding 4): replace the immutability
# trigger with the v3 definition (retention may drop a
# resolution pointer); no existing row changes, no table rebuild.
_MIGRATION_2_TO_3 = (
    "DROP TRIGGER IF EXISTS immutable_effect_record_update",
    _TRIGGER_IMMUTABILITY_V3,
)

_EXPECTED_COLUMNS = {
    "effect_meta": ["key", "value"],
    "effect_records": [
        "seq",
        "recorded_at",
        "daemon_boot_id",
        "daemon_seq",
        "request_id",
        "capability",
        "principal_id",
        "run_id",
        "params_json",
        "effect_json",
        "outcome",
        "error_code",
        "unknown_reason",
        "resolved_by_seq",
        "resolved_at",
    ],
    "effect_observations": [
        "id",
        "effect_seq",
        "attached_at",
        "source",
        "source_ref",
        "observation_json",
        "observation_sha256",
    ],
    "pruned_daemon_ranges": ["daemon_boot_id", "seq_from", "seq_to"],
}

_T = TypeVar("_T")


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent [from, to] intervals, sorted."""
    merged: list[tuple[int, int]] = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _subtract_intervals(
    low: int, high: int, covered: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Parts of [low, high] not covered by any interval, in order."""
    remaining: list[tuple[int, int]] = [(low, high)]
    for cover_low, cover_high in sorted(covered):
        nxt: list[tuple[int, int]] = []
        for seg_low, seg_high in remaining:
            if cover_high < seg_low or cover_low > seg_high:
                nxt.append((seg_low, seg_high))
                continue
            if seg_low < cover_low:
                nxt.append((seg_low, cover_low - 1))
            if seg_high > cover_high:
                nxt.append((cover_high + 1, seg_high))
        remaining = nxt
        if not remaining:
            break
    return remaining


class BridgeEffectStore:
    """Transactional journal for bridge effects and observations."""

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_ms: int = 2000,
        max_retries: int = 3,
        effect_cap: int = DEFAULT_EFFECT_CAP,
        per_effect_observation_cap: int = DEFAULT_PER_EFFECT_OBSERVATION_CAP,
        observation_cap: int = DEFAULT_OBSERVATION_CAP,
    ) -> None:
        self.path = path
        self._busy_timeout_ms = busy_timeout_ms
        self._max_retries = max_retries
        self._effect_cap = effect_cap
        self._per_effect_observation_cap = per_effect_observation_cap
        self._observation_cap = observation_cap
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == _SCHEMA_VERSION:
                self._validate_schema(conn)
                return
            if version == 1:
                # v1 -> v2 (Codex round 1, finding 4): add the
                # pruned-ranges table; no existing row changes.
                for statement in _MIGRATION_1_TO_2:
                    conn.execute(statement)
                version = 2
            if version == 2:
                # v2 -> v3 (Codex round 2, finding 4): replace the
                # immutability trigger so retention may drop a
                # resolution pointer; no existing row changes, no
                # table rebuild.
                for statement in _MIGRATION_2_TO_3:
                    conn.execute(statement)
                version = 3
            if version == _SCHEMA_VERSION:
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                self._validate_schema(conn)
                return
            occupied = conn.execute(
                "SELECT count(*) FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            if version != 0 or occupied:
                raise RuntimeError("Unsupported bridge effect schema; no automatic migration")
            for statement in _SCHEMA:
                conn.execute(statement)
            conn.execute("INSERT INTO effect_meta (key, value) VALUES ('next_seq', 1)")
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        return conn

    @staticmethod
    def _validate_schema(conn: sqlite3.Connection) -> None:
        for table, columns in _EXPECTED_COLUMNS.items():
            actual = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if actual != columns:
                raise RuntimeError(f"Unsupported bridge effect schema for {table}")
        counter = conn.execute("SELECT value FROM effect_meta WHERE key = 'next_seq'").fetchone()
        if counter is None:
            raise RuntimeError("Unsupported bridge effect schema: missing next_seq")

    def _write(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run ``fn`` under BEGIN IMMEDIATE with bounded retries on
        lock contention (owner condition 4): the busy timeout
        bounds each attempt, ``max_retries`` bounds the attempts,
        and then the error surfaces instead of waiting forever."""
        last_error: sqlite3.OperationalError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with closing(self._connect()) as conn, conn:
                    conn.execute("BEGIN IMMEDIATE")
                    return fn(conn)
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "locked" not in message and "busy" not in message:
                    raise
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _read(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        with closing(self._connect()) as conn:
            return fn(conn)

    # -- recording -----------------------------------------------------------

    def record_call(
        self,
        *,
        capability: str,
        request_id: str,
        principal_id: str | None,
        run_id: str | None,
        params_json: str,
        effect_json: str | None,
        daemon_boot_id: str | None,
        daemon_seq: int | None,
        outcome: str,
        error_code: str | None,
        unknown_reason: str | None,
        recorded_at: str | None = None,
    ) -> dict[str, object]:
        """Record one bridge call. Returns ``{"seq", "outcome",
        "duplicate", "resolved_unknown_seqs"}``.

        Dedupe (owner condition 1): a row with the same daemon
        emission identity ``(daemon_boot_id, daemon_seq)`` already
        recorded returns the existing row (``duplicate: True``) —
        the subject is never a dedupe key, only the emission
        identity is. Resolution (owner condition 2): when the new
        row is a confirmed effect, unresolved ``unknown`` rows with
        the same ``request_id`` gain the resolution link in the
        same transaction; the attempt rows stay as history. A
        duplicate delivery of a confirmed effect resolves them
        too (Codex round 1, finding 2): effect -> timeout -> the
        same effect must not leave the timeout row unknown
        forever. Retention: the oldest rows are pruned (with their
        observations, same transaction) past ``effect_cap``.
        """
        now = recorded_at or datetime.now(UTC).isoformat()

        def _insert(conn: sqlite3.Connection) -> dict[str, object]:
            if daemon_seq is not None:
                existing = conn.execute(
                    "SELECT seq, outcome FROM effect_records "
                    "WHERE daemon_boot_id = ? AND daemon_seq = ?",
                    (daemon_boot_id, daemon_seq),
                ).fetchone()
                if existing is not None:
                    resolved_dup = (
                        self._resolve_unknowns_for_request(conn, request_id, existing["seq"], now)
                        if existing["outcome"] == "effect"
                        else []
                    )
                    return {
                        "seq": existing["seq"],
                        "outcome": existing["outcome"],
                        "duplicate": True,
                        "resolved_unknown_seqs": resolved_dup,
                    }
            seq = conn.execute("SELECT value FROM effect_meta WHERE key = 'next_seq'").fetchone()[
                "value"
            ]
            conn.execute("UPDATE effect_meta SET value = ? WHERE key = 'next_seq'", (seq + 1,))
            conn.execute(
                "INSERT INTO effect_records (seq, recorded_at, daemon_boot_id, daemon_seq,"
                " request_id, capability, principal_id, run_id, params_json, effect_json,"
                " outcome, error_code, unknown_reason)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    seq,
                    now,
                    daemon_boot_id,
                    daemon_seq,
                    request_id,
                    capability,
                    principal_id,
                    run_id,
                    params_json,
                    effect_json,
                    outcome,
                    error_code,
                    unknown_reason,
                ),
            )
            resolved: list[int] = []
            if outcome == "effect":
                # Only a confirmed effect resolves an unknown (owner
                # condition 2). A no_effect retry must NOT resolve:
                # patch_mismatch on retry after a successful apply,
                # or run_stopped after a concurrent STOP, proves
                # nothing about the first attempt.
                resolved = self._resolve_unknowns_for_request(conn, request_id, seq, now)
            over = (
                conn.execute("SELECT count(*) FROM effect_records").fetchone()[0] - self._effect_cap
            )
            if over > 0:
                # The rows about to be deleted, by server seq — the
                # same set the DELETE below removes (Codex round 1,
                # finding 4): their daemon seqs are pruned, not
                # lost, so gap detection must know about them.
                doomed = conn.execute(
                    "SELECT daemon_boot_id, daemon_seq FROM effect_records"
                    " WHERE seq IN (SELECT seq FROM effect_records ORDER BY seq ASC LIMIT ?)"
                    " AND daemon_seq IS NOT NULL",
                    (over,),
                ).fetchall()
                # Codex round 2, finding 4: the rows about to be
                # deleted may be named by resolution links
                # (resolved_by_seq REFERENCES effect_records(seq),
                # NO ACTION). Drop those pointers in the same
                # transaction first — the FK is a backstop, not a
                # wedge: without this the DELETE fails and every
                # later insert rolls back the same way. The
                # resolution itself survives in resolved_at, so the
                # attempts still read as resolved; the trigger
                # permits exactly this set -> NULL transition.
                conn.execute(
                    "UPDATE effect_records SET resolved_by_seq = NULL"
                    " WHERE resolved_by_seq IN ("
                    " SELECT seq FROM effect_records ORDER BY seq ASC LIMIT ?)",
                    (over,),
                )
                conn.execute(
                    "DELETE FROM effect_records WHERE seq IN ("
                    " SELECT seq FROM effect_records ORDER BY seq ASC LIMIT ?)",
                    (over,),
                )
                self._record_pruned_ranges(conn, doomed)
                # Prune data for boots with no retained rows is dead
                # weight: their gaps can never be computed.
                conn.execute(
                    "DELETE FROM pruned_daemon_ranges WHERE daemon_boot_id NOT IN ("
                    " SELECT DISTINCT daemon_boot_id FROM effect_records"
                    " WHERE daemon_seq IS NOT NULL)"
                )
                # Codex round 3, finding 2: ranges entirely below
                # a boot's retention floor (min retained daemon
                # seq) are dead weight too — _compute_gaps never
                # searches below the floor — but they accumulate
                # forever when daemon seqs arrive sparsely (a range
                # per pruned seq, never merged, never dropped while
                # the boot keeps any row). Drop them at prune time,
                # in the same transaction. A boot with no retained
                # rows yields NULL for the floor, so the comparison
                # is NULL and deletes nothing: that case is covered
                # by the boot cleanup above.
                conn.execute(
                    "DELETE FROM pruned_daemon_ranges"
                    " WHERE seq_to < ("
                    " SELECT MIN(daemon_seq) FROM effect_records"
                    " WHERE daemon_boot_id = pruned_daemon_ranges.daemon_boot_id"
                    " AND daemon_seq IS NOT NULL)"
                )
            return {
                "seq": seq,
                "outcome": outcome,
                "duplicate": False,
                "resolved_unknown_seqs": resolved,
            }

        return self._write(_insert)

    @staticmethod
    def _resolve_unknowns_for_request(
        conn: sqlite3.Connection, request_id: str, effect_seq: int, now: str
    ) -> list[int]:
        """Link unresolved unknown rows for this logical request to
        the confirmed effect (owner condition 2). The attempt rows
        stay as history; only the resolution link is set (the
        trigger permits exactly this mutation). Resolved-ness is
        tested on ``resolved_at`` (Codex round 2, finding 4):
        retention may later drop the ``resolved_by_seq`` pointer
        when it deletes the effect, but the timestamp — set
        exactly once at resolution — never leaves."""
        resolved = [
            row["seq"]
            for row in conn.execute(
                "SELECT seq FROM effect_records WHERE request_id = ?"
                " AND outcome = 'unknown' AND resolved_at IS NULL AND seq != ?",
                (request_id, effect_seq),
            ).fetchall()
        ]
        for unknown_seq in resolved:
            conn.execute(
                "UPDATE effect_records SET resolved_by_seq = ?, resolved_at = ? WHERE seq = ?",
                (effect_seq, now, unknown_seq),
            )
        return resolved

    @staticmethod
    def _record_pruned_ranges(conn: sqlite3.Connection, doomed: list[sqlite3.Row]) -> None:
        """Remember the daemon seqs retention just deleted, as
        merged intervals per boot (Codex round 1, finding 4). A
        hole in the retained daemon sequence is a real gap only
        where it is NOT covered by these intervals."""
        by_boot: dict[str, list[int]] = {}
        for row in doomed:
            by_boot.setdefault(row["daemon_boot_id"], []).append(row["daemon_seq"])
        for boot_id, seqs in by_boot.items():
            seqs.sort()
            fresh: list[tuple[int, int]] = []
            start = prev = seqs[0]
            for seq in seqs[1:]:
                if seq <= prev + 1:
                    prev = seq
                else:
                    fresh.append((start, prev))
                    start = prev = seq
            fresh.append((start, prev))
            existing = [
                (row["seq_from"], row["seq_to"])
                for row in conn.execute(
                    "SELECT seq_from, seq_to FROM pruned_daemon_ranges"
                    " WHERE daemon_boot_id = ? ORDER BY seq_from",
                    (boot_id,),
                ).fetchall()
            ]
            merged = _merge_intervals(existing + fresh)
            conn.execute("DELETE FROM pruned_daemon_ranges WHERE daemon_boot_id = ?", (boot_id,))
            conn.executemany(
                "INSERT INTO pruned_daemon_ranges (daemon_boot_id, seq_from, seq_to)"
                " VALUES (?, ?, ?)",
                [(boot_id, low, high) for low, high in merged],
            )

    def has_request_id(self, request_id: str) -> bool:
        """True when any row already exists for this logical request
        (the recorder's retry test)."""
        return self._read(
            lambda conn: int(
                conn.execute(
                    "SELECT count(*) FROM effect_records WHERE request_id = ?",
                    (request_id,),
                ).fetchone()[0]
            )
            > 0
        )

    # -- observations --------------------------------------------------------

    def attach_observation(
        self,
        *,
        effect_seq: int,
        source: str,
        source_ref: str | None,
        observation_json: str,
        observation_sha256: str,
        attached_at: str | None = None,
    ) -> int:
        """Attach an observation to an effect. Idempotent: the same
        ``(effect_seq, source, observation_sha256)`` returns the
        existing id. Bounded (owner condition 3): the per-effect
        and global caps evict oldest-first in the same transaction.
        Raises ``ValueError`` when the effect does not exist (the
        FK would reject it — e.g. pruned between the recorder's
        check and this insert)."""
        now = attached_at or datetime.now(UTC).isoformat()

        def _attach(conn: sqlite3.Connection) -> int:
            if (
                conn.execute(
                    "SELECT count(*) FROM effect_records WHERE seq = ?", (effect_seq,)
                ).fetchone()[0]
                == 0
            ):
                raise ValueError(f"unknown effect_seq {effect_seq}")
            existing = conn.execute(
                "SELECT id FROM effect_observations WHERE effect_seq = ?"
                " AND source = ? AND observation_sha256 = ?",
                (effect_seq, source, observation_sha256),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            # Per-effect cap: keep the newest cap-1, then insert.
            conn.execute(
                "DELETE FROM effect_observations WHERE id IN ("
                " SELECT id FROM effect_observations WHERE effect_seq = ?"
                " ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (effect_seq, max(self._per_effect_observation_cap - 1, 0)),
            )
            # Global cap: same, across all effects.
            conn.execute(
                "DELETE FROM effect_observations WHERE id IN ("
                " SELECT id FROM effect_observations"
                " ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (max(self._observation_cap - 1, 0),),
            )
            try:
                cursor = conn.execute(
                    "INSERT INTO effect_observations (effect_seq, attached_at, source,"
                    " source_ref, observation_json, observation_sha256)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (effect_seq, now, source, source_ref, observation_json, observation_sha256),
                )
            except sqlite3.IntegrityError as exc:
                message = str(exc)
                if "FOREIGN KEY" in message:
                    raise ValueError(f"unknown effect_seq {effect_seq}") from exc
                if "UNIQUE" in message:
                    duplicate = conn.execute(
                        "SELECT id FROM effect_observations WHERE effect_seq = ?"
                        " AND source = ? AND observation_sha256 = ?",
                        (effect_seq, source, observation_sha256),
                    ).fetchone()
                    if duplicate is not None:
                        return int(duplicate["id"])
                raise
            return cursor.lastrowid  # type: ignore[return-value]

        return self._write(_attach)

    # -- reads ---------------------------------------------------------------

    def get_record(self, seq: int) -> dict[str, object] | None:
        """One record with its observations, or None."""
        return self._read(lambda conn: self._get_record(conn, seq))

    def _get_record(self, conn: sqlite3.Connection, seq: int) -> dict[str, object] | None:
        row = conn.execute("SELECT * FROM effect_records WHERE seq = ?", (seq,)).fetchone()
        if row is None:
            return None
        observations = self._observations_for(conn, [seq])
        return self._row_to_record(row, observations.get(seq, []))

    def list_effects(
        self,
        *,
        principal_id: str,
        run_id: str | None = None,
        outcome: str | None = None,
        cursor: int | None = None,
        limit: int = 100,
        include_gaps: bool = False,
        now: datetime | None = None,
    ) -> dict[str, object]:
        """Oldest-first page of the caller's own records.

        ``cursor`` is the last seen server ``seq``; the page holds
        rows with ``seq > cursor``. ``next_cursor`` is the last
        returned seq when more rows follow, else None. An
        ``outcome='unknown'`` filter never returns rows that were
        later resolved (owner condition 2). Gaps are computed over
        the WHOLE effect sequence (owner condition 7) and only
        when ``include_gaps`` is set — the handler enables it for
        the owner on unfiltered reads only.
        """
        return self._read(
            lambda conn: self._list_effects(
                conn,
                principal_id=principal_id,
                run_id=run_id,
                outcome=outcome,
                cursor=cursor,
                limit=limit,
                include_gaps=include_gaps,
                now=now or datetime.now(UTC),
            )
        )

    def _list_effects(
        self,
        conn: sqlite3.Connection,
        *,
        principal_id: str,
        run_id: str | None,
        outcome: str | None,
        cursor: int | None,
        limit: int,
        include_gaps: bool,
        now: datetime,
    ) -> dict[str, object]:
        conditions = ["principal_id = ?"]
        params: list[object] = [principal_id]
        if run_id is not None:
            conditions.append("run_id = ?")
            params.append(run_id)
        if outcome is not None:
            conditions.append("outcome = ?")
            params.append(outcome)
            if outcome == "unknown":
                # A resolved unknown is history, not an unknown:
                # the outcome=unknown listing must not show it
                # (owner condition 2). Resolved-ness is tested on
                # resolved_at (Codex round 2, finding 4): the
                # pointer may leave with its target under
                # retention, the timestamp never does.
                conditions.append("resolved_at IS NULL")
        if cursor is not None:
            conditions.append("seq > ?")
            params.append(cursor)
        rows = conn.execute(
            "SELECT * FROM effect_records WHERE "
            + " AND ".join(conditions)
            + " ORDER BY seq ASC LIMIT ?",
            tuple([*params, limit + 1]),
        ).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        seqs = [row["seq"] for row in rows]
        observations = self._observations_for(conn, seqs)
        records = [self._row_to_record(row, observations.get(row["seq"], [])) for row in rows]
        page: dict[str, object] = {
            "records": records,
            "next_cursor": seqs[-1] if has_more and seqs else None,
        }
        if include_gaps:
            page["gaps"] = self._compute_gaps(conn, now=now)
        return page

    def _observations_for(
        self, conn: sqlite3.Connection, seqs: list[int]
    ) -> dict[int, list[dict[str, object]]]:
        if not seqs:
            return {}
        placeholders = ",".join("?" * len(seqs))
        grouped: dict[int, list[dict[str, object]]] = {}
        for row in conn.execute(
            "SELECT * FROM effect_observations WHERE effect_seq IN (" + placeholders + ")"
            " ORDER BY id ASC",
            tuple(seqs),
        ).fetchall():
            grouped.setdefault(row["effect_seq"], []).append(
                {
                    "id": row["id"],
                    "attached_at": row["attached_at"],
                    "source": row["source"],
                    "source_ref": row["source_ref"],
                    "observation": json.loads(row["observation_json"]),
                    "observation_sha256": row["observation_sha256"],
                }
            )
        return grouped

    def _compute_gaps(self, conn: sqlite3.Connection, *, now: datetime) -> list[dict[str, object]]:
        """Holes in the per-boot daemon sequence, over ALL records.

        The retention floor (min retained daemon_seq - 1 per boot)
        is never a gap: anything below it was pruned, not lost. A
        hole is ``suspected`` only once the record above it is
        older than the grace window — a late (out-of-order)
        delivery above the floor closes the hole at read time.
        Retention prunes by server seq (arrival order), so it can
        also delete the MIDDLE of the retained daemon sequence
        (Codex round 1, finding 4): the pruned intervals recorded
        at prune time are subtracted first, and only the part of
        a hole that was never delivered is reported.
        """
        rows = conn.execute(
            "SELECT daemon_boot_id, daemon_seq, recorded_at FROM effect_records"
            " WHERE daemon_seq IS NOT NULL ORDER BY daemon_boot_id, daemon_seq"
        ).fetchall()
        pruned: dict[str, list[tuple[int, int]]] = {}
        for row in conn.execute(
            "SELECT daemon_boot_id, seq_from, seq_to FROM pruned_daemon_ranges"
        ).fetchall():
            pruned.setdefault(row["daemon_boot_id"], []).append((row["seq_from"], row["seq_to"]))
        by_boot: dict[str, list[tuple[int, str]]] = {}
        for row in rows:
            by_boot.setdefault(row["daemon_boot_id"], []).append(
                (row["daemon_seq"], row["recorded_at"])
            )
        gaps: list[dict[str, object]] = []
        grace_before = now - timedelta(seconds=GAP_GRACE_S)
        for boot_id, seqs in by_boot.items():
            seqs.sort()
            for (lower, _), (upper, upper_at) in zip(seqs, seqs[1:], strict=False):
                if upper <= lower + 1:
                    continue
                try:
                    upper_recorded = datetime.fromisoformat(upper_at)
                except ValueError:
                    upper_recorded = now
                # The floor is lower (the min retained seq) -
                # 1 by construction: every hole found here
                # lies above it, so it is never a prune
                # artifact. A late delivery closes it at
                # read time (the hole disappears).
                for missing_from, missing_to in _subtract_intervals(
                    lower + 1, upper - 1, pruned.get(boot_id, [])
                ):
                    gaps.append(
                        {
                            "daemon_boot_id": boot_id,
                            "missing_from": missing_from,
                            "missing_to": missing_to,
                            "suspected": upper_recorded < grace_before,
                        }
                    )
                    if len(gaps) >= MAX_GAP_ENTRIES:
                        return gaps
        return gaps

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row, observations: list[dict[str, object]]
    ) -> dict[str, object]:
        return {
            "seq": row["seq"],
            "recorded_at": row["recorded_at"],
            "daemon_boot_id": row["daemon_boot_id"],
            "daemon_seq": row["daemon_seq"],
            "request_id": row["request_id"],
            "capability": row["capability"],
            "principal_id": row["principal_id"],
            "run_id": row["run_id"],
            "params": json.loads(row["params_json"]),
            "effect": json.loads(row["effect_json"]) if row["effect_json"] is not None else None,
            "outcome": row["outcome"],
            "error_code": row["error_code"],
            "unknown_reason": row["unknown_reason"],
            "resolved_by_seq": row["resolved_by_seq"],
            "resolved_at": row["resolved_at"],
            "observations": observations,
        }

    def count(self) -> int:
        return self._read(
            lambda conn: int(conn.execute("SELECT count(*) FROM effect_records").fetchone()[0])
        )
