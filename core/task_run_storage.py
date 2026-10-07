from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from shared.task_run import RunScope, RunState, RunTransition, TaskRun, TransitionAuthority


class RunConflictError(ValueError):
    pass


class TaskRunStore:
    """Transactional admission/transition authority; runtime adoption is a separate slice."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 2:
                return
            occupied = conn.execute(
                "SELECT count(*) FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            if version != 0 or occupied:
                raise RuntimeError("Unsupported task/run schema; no automatic migration")
            statements = (
                """CREATE TABLE IF NOT EXISTS task_revisions (
                    task_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision = 1),
                    principal_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_key TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    PRIMARY KEY (task_id, revision),
                    UNIQUE (principal_id, request_key)
                );""",
                """CREATE TABLE IF NOT EXISTS task_runs (
                    task_run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_revision INTEGER NOT NULL,
                    attempt_id TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 0),
                    created_at TEXT NOT NULL,
                    terminal_at TEXT,
                    FOREIGN KEY (task_id, task_revision)
                    REFERENCES task_revisions(task_id, revision)
                );""",
                """CREATE TABLE IF NOT EXISTS run_transitions (
                    task_run_id TEXT NOT NULL REFERENCES task_runs(task_run_id),
                    version INTEGER NOT NULL,
                    prior_state TEXT,
                    state TEXT NOT NULL,
                    authority TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    PRIMARY KEY (task_run_id, version)
                );""",
                """CREATE TRIGGER IF NOT EXISTS immutable_transition_update
                BEFORE UPDATE ON run_transitions
                BEGIN SELECT RAISE(ABORT, 'immutable transition'); END;""",
                """CREATE TRIGGER IF NOT EXISTS immutable_transition_delete
                BEFORE DELETE ON run_transitions
                BEGIN SELECT RAISE(ABORT, 'immutable transition'); END;""",
                """CREATE TRIGGER IF NOT EXISTS immutable_revision_update
                BEFORE UPDATE ON task_revisions
                BEGIN SELECT RAISE(ABORT, 'immutable revision'); END;""",
                """CREATE TRIGGER IF NOT EXISTS immutable_revision_delete
                BEFORE DELETE ON task_revisions
                BEGIN SELECT RAISE(ABORT, 'immutable revision'); END;""",
            )
            for statement in statements:
                conn.execute(statement)
            conn.execute("PRAGMA user_version = 2")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def admit(
        self, scope: RunScope, *, request_key: str, goal: str, mode: str, request_fingerprint: str
    ) -> TaskRun:
        if len(request_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in request_fingerprint
        ):
            raise ValueError("Invalid accepted request fingerprint")
        if not request_key.strip() or len(request_key) > 128:
            raise ValueError("Invalid admission request key")
        if not goal.strip() or len(goal.encode("utf-8")) > 512 * 1024:
            raise ValueError("Invalid accepted goal")
        if mode not in {"ask", "plan", "act", "auto", "desktop"}:
            raise ValueError("Invalid execution mode")
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM task_revisions WHERE principal_id = ? AND request_key = ?",
                (scope.principal_id, request_key),
            ).fetchone()
            if existing is not None:
                if (
                    existing["session_id"],
                    existing["goal"],
                    existing["mode"],
                    existing["request_fingerprint"],
                ) != (
                    scope.session_id,
                    goal,
                    mode,
                    request_fingerprint,
                ):
                    raise RunConflictError("Admission key already belongs to a different request")
                row = conn.execute(
                    "SELECT * FROM task_runs WHERE task_id = ? AND task_revision = ?",
                    (existing["task_id"], existing["revision"]),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Admission has no run")
                return self._record(scope, self._owned_row(conn, scope, row["task_run_id"]))
            task_id, run_id, attempt_id = (str(uuid4()) for _ in range(3))
            now = datetime.now(UTC).isoformat()
            conn.execute(
                "INSERT INTO task_revisions VALUES (?, 1, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    scope.principal_id,
                    scope.session_id,
                    request_key,
                    goal,
                    mode,
                    request_fingerprint,
                ),
            )
            conn.execute(
                "INSERT INTO task_runs VALUES (?, ?, 1, ?, ?, 0, ?, NULL)",
                (run_id, task_id, attempt_id, RunState.ADMITTED, now),
            )
            conn.execute(
                "INSERT INTO run_transitions VALUES (?, 0, NULL, ?, ?, ?, ?)",
                (run_id, RunState.ADMITTED, TransitionAuthority.RUNTIME, "request_accepted", now),
            )
            return self._record(scope, self._owned_row(conn, scope, run_id))

    def transition(
        self,
        scope: RunScope,
        task_run_id: str,
        *,
        expected_version: int,
        state: RunState,
        authority: TransitionAuthority,
        reason_code: str,
    ) -> TaskRun:
        if not isinstance(state, RunState) or not isinstance(authority, TransitionAuthority):
            raise ValueError("Typed transition state/authority required")
        if (
            not reason_code
            or len(reason_code) > 128
            or not all(char.isascii() and (char.isalnum() or char == "_") for char in reason_code)
        ):
            raise ValueError("Invalid transition reason code")
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._record(scope, self._owned_row(conn, scope, task_run_id))
            if current.version != expected_version:
                raise RunConflictError("Stale run version")
            allowed = {
                (RunState.ADMITTED, RunState.RUNNING, TransitionAuthority.RUNTIME),
                (RunState.RUNNING, RunState.RESULT_SUBMITTED, TransitionAuthority.RUNTIME),
                (RunState.RUNNING, RunState.RECOVERY_REQUIRED, TransitionAuthority.RECOVERY),
                (
                    RunState.RESULT_SUBMITTED,
                    RunState.RECOVERY_REQUIRED,
                    TransitionAuthority.RECOVERY,
                ),
            }
            if state == RunState.ABORTED and current.state != RunState.ABORTED:
                valid = authority in {TransitionAuthority.USER, TransitionAuthority.RECOVERY}
            else:
                valid = (current.state, state, authority) in allowed
            if not valid:
                raise RunConflictError("Transition/authority is not permitted")
            now = datetime.now(UTC).isoformat()
            terminal_at = now if state == RunState.ABORTED else None
            conn.execute(
                "UPDATE task_runs SET state = ?, version = version + 1, terminal_at = ? "
                "WHERE task_run_id = ? AND version = ?",
                (state, terminal_at, task_run_id, expected_version),
            )
            conn.execute(
                "INSERT INTO run_transitions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task_run_id,
                    current.version + 1,
                    current.state,
                    state,
                    authority,
                    reason_code,
                    now,
                ),
            )
            return self._record(scope, self._owned_row(conn, scope, task_run_id))

    def find_by_request(self, scope: RunScope, request_key: str) -> TaskRun | None:
        """Read a principal/session-owned admission; absence never authorizes rerun."""
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN")
            row = conn.execute(
                "SELECT r.task_run_id FROM task_runs r JOIN task_revisions t "
                "ON t.task_id = r.task_id AND t.revision = r.task_revision "
                "WHERE t.principal_id = ? AND t.session_id = ? AND t.request_key = ?",
                (scope.principal_id, scope.session_id, request_key),
            ).fetchone()
            if row is None:
                return None
            return self._record(scope, self._owned_row(conn, scope, row["task_run_id"]))

    def get(self, scope: RunScope, task_run_id: str) -> TaskRun:
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN")
            return self._record(scope, self._owned_row(conn, scope, task_run_id))

    def events(self, scope: RunScope, task_run_id: str) -> tuple[RunTransition, ...]:
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN")
            self._owned_row(conn, scope, task_run_id)
            rows = conn.execute(
                "SELECT * FROM run_transitions WHERE task_run_id = ? ORDER BY version",
                (task_run_id,),
            ).fetchall()
            return tuple(
                RunTransition(
                    version=row["version"],
                    prior_state=RunState(row["prior_state"]) if row["prior_state"] else None,
                    state=RunState(row["state"]),
                    authority=TransitionAuthority(row["authority"]),
                    reason_code=row["reason_code"],
                    occurred_at=row["occurred_at"],
                )
                for row in rows
            )

    @staticmethod
    def _owned_row(conn: sqlite3.Connection, scope: RunScope, run_id: str) -> sqlite3.Row:
        row = conn.execute(
            """SELECT r.* FROM task_runs r JOIN task_revisions t
            ON t.task_id = r.task_id AND t.revision = r.task_revision
            WHERE r.task_run_id = ? AND t.principal_id = ? AND t.session_id = ?""",
            (run_id, scope.principal_id, scope.session_id),
        ).fetchone()
        if row is None:
            raise LookupError("Run unavailable in this scope")
        history = conn.execute(
            "SELECT * FROM run_transitions WHERE task_run_id = ? ORDER BY version", (run_id,)
        ).fetchall()
        previous = None
        for version, event in enumerate(history):
            if event["version"] != version or event["prior_state"] != previous:
                raise RuntimeError("Run history is inconsistent")
            previous = event["state"]
        if not history or len(history) != row["version"] + 1 or previous != row["state"]:
            raise RuntimeError("Run state has no matching history")
        if (
            history[0]["state"] != RunState.ADMITTED
            or history[0]["occurred_at"] != row["created_at"]
        ):
            raise RuntimeError("Admission history is inconsistent")
        expected_terminal = history[-1]["occurred_at"] if previous == RunState.ABORTED else None
        if row["terminal_at"] != expected_terminal:
            raise RuntimeError("Terminal history is inconsistent")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _record(scope: RunScope, row: sqlite3.Row) -> TaskRun:
        return TaskRun(
            task_id=row["task_id"],
            task_revision=row["task_revision"],
            task_run_id=row["task_run_id"],
            attempt_id=row["attempt_id"],
            scope=scope,
            state=RunState(row["state"]),
            version=row["version"],
            created_at=row["created_at"],
            terminal_at=row["terminal_at"],
        )
