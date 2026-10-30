from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from shared.task_run import (
    ProducerKind,
    ProducerReport,
    RunScope,
    RunState,
    RunSubmission,
    RunTransition,
    TaskRun,
    TransitionAuthority,
)
from shared.tool_capture import (
    MAX_CAPTURE_BYTES,
    MAX_VIEW_BYTES,
    ProcessObservation,
    StoredToolObservation,
    ToolOperation,
)

_ADMISSION_SCHEMA = (
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


class RunConflictError(ValueError):
    pass


class RunAlreadyAdmittedError(RunConflictError):
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
            if version == 4:
                self._validate_v3_schema(conn, capture=True)
                return
            if version == 3:
                self._validate_v3_schema(conn)
                self._upgrade_capture(conn)
                return
            if version == 2:
                self._validate_v2_schema(conn)
                self._upgrade_submissions(conn)
                self._upgrade_capture(conn)
                return
            occupied = conn.execute(
                "SELECT count(*) FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchone()[0]
            if version != 0 or occupied:
                raise RuntimeError("Unsupported task/run schema; no automatic migration")
            for statement in _ADMISSION_SCHEMA:
                conn.execute(statement)
            self._upgrade_submissions(conn)
            self._upgrade_capture(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _validate_v3_schema(conn: sqlite3.Connection, *, capture: bool = False) -> None:
        columns = {
            "task_revisions": (
                "task_id",
                "revision",
                "principal_id",
                "session_id",
                "request_key",
                "goal",
                "mode",
                "request_fingerprint",
            ),
            "task_runs": (
                "task_run_id",
                "task_id",
                "task_revision",
                "attempt_id",
                "state",
                "version",
                "created_at",
                "terminal_at",
            ),
            "run_transitions": (
                "task_run_id",
                "version",
                "prior_state",
                "state",
                "authority",
                "reason_code",
                "occurred_at",
                "submission_id",
            ),
            "run_submissions": (
                "submission_id",
                "task_run_id",
                "version",
                "payload",
                "content_digest",
                "created_at",
            ),
        }
        if capture:
            columns["tool_operations"] = (
                "operation_id",
                "tool_attempt_id",
                "task_run_id",
                "run_attempt_id",
                "run_version",
                "model_call_id",
                "request_path",
                "created_at",
                "capture_limit",
                "view_limit",
            )
            columns["tool_observations"] = (
                "result_id",
                "operation_id",
                "stdout",
                "stderr",
                "stdout_seen",
                "stderr_seen",
                "stdout_eof",
                "stderr_eof",
                "exit_code",
                "timed_out",
                "capture_error",
                "captured_at",
                "stdout_digest",
                "stderr_digest",
            )
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        triggers = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        }
        required = {
            "immutable_transition_update",
            "immutable_transition_delete",
            "immutable_revision_update",
            "immutable_revision_delete",
            "immutable_submission_update",
            "immutable_submission_delete",
            "submitted_result_binding",
        }
        if capture:
            required |= {
                "immutable_operation_update",
                "immutable_operation_delete",
                "immutable_observation_update",
                "immutable_observation_delete",
            }
        if tables != set(columns) or not required <= triggers:
            raise RuntimeError("Unsupported task/run current schema")
        for table, names in columns.items():
            if tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})")) != names:
                raise RuntimeError("Unsupported task/run current schema")

    @staticmethod
    def _validate_v2_schema(conn: sqlite3.Connection) -> None:
        # Only the exact shipped admission schema is eligible for this one additive upgrade.
        def normalized(sql: str) -> str:
            return " ".join(sql.replace(" IF NOT EXISTS", "").rstrip(";").split())

        actual = {
            normalized(row[0])
            for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
            )
        }
        if actual != {normalized(sql) for sql in _ADMISSION_SCHEMA}:
            raise RuntimeError("Unsupported task/run v2 schema")

    @staticmethod
    def _upgrade_submissions(conn: sqlite3.Connection) -> None:
        conn.execute(
            """CREATE TABLE run_submissions (
                submission_id TEXT PRIMARY KEY NOT NULL,
                task_run_id TEXT NOT NULL UNIQUE REFERENCES task_runs(task_run_id),
                version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (task_run_id, version)
                REFERENCES run_transitions(task_run_id, version) DEFERRABLE INITIALLY DEFERRED
            )"""
        )
        conn.execute(
            "ALTER TABLE run_transitions ADD COLUMN submission_id TEXT "
            "REFERENCES run_submissions(submission_id)"
        )
        for operation in ("UPDATE", "DELETE"):
            conn.execute(
                f"CREATE TRIGGER immutable_submission_{operation.lower()} "
                f"BEFORE {operation} ON run_submissions "
                "BEGIN SELECT RAISE(ABORT, 'immutable submission'); END"
            )
        conn.execute(
            """CREATE TRIGGER submitted_result_binding BEFORE INSERT ON run_transitions
            WHEN (NEW.state = 'result_submitted' AND (
                NEW.submission_id IS NULL OR NOT EXISTS (
                    SELECT 1 FROM run_submissions s
                    WHERE s.submission_id = NEW.submission_id
                    AND s.task_run_id = NEW.task_run_id AND s.version = NEW.version
                    AND s.created_at = NEW.occurred_at
                )
            )) OR (NEW.state != 'result_submitted' AND NEW.submission_id IS NOT NULL)
            BEGIN SELECT RAISE(ABORT, 'submission binding required'); END"""
        )
        conn.execute("PRAGMA user_version = 3")

    @staticmethod
    def _upgrade_capture(conn: sqlite3.Connection) -> None:
        conn.execute(
            """CREATE TABLE tool_operations (
                operation_id TEXT PRIMARY KEY NOT NULL,
                tool_attempt_id TEXT NOT NULL UNIQUE,
                task_run_id TEXT NOT NULL REFERENCES task_runs(task_run_id),
                run_attempt_id TEXT NOT NULL,
                run_version INTEGER NOT NULL,
                model_call_id TEXT NOT NULL,
                request_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                capture_limit INTEGER NOT NULL CHECK (capture_limit = 1048576),
                view_limit INTEGER NOT NULL CHECK (view_limit = 32768),
                FOREIGN KEY (task_run_id, run_version)
                REFERENCES run_transitions(task_run_id, version)
            )"""
        )
        conn.execute(
            """CREATE TABLE tool_observations (
                result_id TEXT PRIMARY KEY NOT NULL,
                operation_id TEXT NOT NULL UNIQUE REFERENCES tool_operations(operation_id),
                stdout BLOB NOT NULL CHECK (typeof(stdout) = 'blob'),
                stderr BLOB NOT NULL CHECK (typeof(stderr) = 'blob'),
                stdout_seen INTEGER NOT NULL CHECK (stdout_seen >= 0),
                stderr_seen INTEGER NOT NULL CHECK (stderr_seen >= 0),
                stdout_eof INTEGER NOT NULL CHECK (stdout_eof IN (0, 1)),
                stderr_eof INTEGER NOT NULL CHECK (stderr_eof IN (0, 1)),
                exit_code INTEGER,
                timed_out INTEGER NOT NULL CHECK (timed_out IN (0, 1)),
                capture_error TEXT,
                captured_at TEXT NOT NULL,
                stdout_digest TEXT NOT NULL,
                stderr_digest TEXT NOT NULL,
                CHECK (length(stdout) = min(stdout_seen, 1048576)),
                CHECK (length(stderr) = min(stderr_seen, 1048576))
            )"""
        )
        for name, table in (("operation", "tool_operations"), ("observation", "tool_observations")):
            for operation in ("UPDATE", "DELETE"):
                conn.execute(
                    f"CREATE TRIGGER immutable_{name}_{operation.lower()} "
                    f"BEFORE {operation} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT, 'immutable {name}'); END"
                )
        conn.execute("PRAGMA user_version = 4")

    def begin_tool_operation(
        self, run: TaskRun, *, model_call_id: str, request_path: str
    ) -> ToolOperation:
        for value, maximum in ((model_call_id, 4096), (request_path, 16384)):
            if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
                raise ValueError("Invalid captured operation input")
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._record(run.scope, self._owned_row(conn, run.scope, run.task_run_id))
            if current != run or current.state != RunState.RUNNING:
                raise RunConflictError("Capture requires the original current running admission")
            mode = conn.execute(
                "SELECT mode FROM task_revisions WHERE task_id=? AND revision=?",
                (run.task_id, run.task_revision),
            ).fetchone()[0]
            if mode != "auto":
                raise RunConflictError("Process capture requires an admitted Auto run")
            identity, attempt = str(uuid4()), str(uuid4())
            now = datetime.now(UTC).isoformat()
            conn.execute(
                "INSERT INTO tool_operations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identity,
                    attempt,
                    run.task_run_id,
                    run.attempt_id,
                    run.version,
                    model_call_id,
                    request_path,
                    now,
                    MAX_CAPTURE_BYTES,
                    MAX_VIEW_BYTES,
                ),
            )
            return self._tool_operation(conn, run.scope, run.task_run_id, identity)

    def get_tool_operation(
        self, scope: RunScope, task_run_id: str, operation_id: str
    ) -> ToolOperation:
        """Intent без observation остаётся unresolved и не разрешает replay."""
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN")
            return self._tool_operation(conn, scope, task_run_id, operation_id)

    def _tool_operation(
        self, conn: sqlite3.Connection, scope: RunScope, run_id: str, operation_id: str
    ) -> ToolOperation:
        row = conn.execute(
            "SELECT o.*, r.task_id, r.task_revision FROM tool_operations o "
            "JOIN task_runs r ON r.task_run_id=o.task_run_id AND r.attempt_id=o.run_attempt_id "
            "JOIN task_revisions t ON t.task_id=r.task_id AND t.revision=r.task_revision "
            "WHERE o.operation_id=? AND o.task_run_id=? AND t.principal_id=? AND t.session_id=?",
            (operation_id, run_id, scope.principal_id, scope.session_id),
        ).fetchone()
        if row is None:
            raise LookupError("Tool capture unavailable in this scope")
        self._owned_row(conn, scope, run_id)
        started = conn.execute(
            "SELECT state FROM run_transitions WHERE task_run_id=? AND version=?",
            (run_id, row["run_version"]),
        ).fetchone()
        if (
            started is None
            or started[0] != RunState.RUNNING
            or row["capture_limit"] != MAX_CAPTURE_BYTES
            or row["view_limit"] != MAX_VIEW_BYTES
        ):
            raise RuntimeError("Tool capture binding/policy is invalid")
        return ToolOperation(
            scope,
            row["task_id"],
            row["task_revision"],
            run_id,
            row["run_attempt_id"],
            row["run_version"],
            row["operation_id"],
            row["tool_attempt_id"],
            row["model_call_id"],
            row["request_path"],
            row["created_at"],
        )

    def record_tool_observation(
        self, operation: ToolOperation, observation: ProcessObservation
    ) -> str:
        if not isinstance(observation, ProcessObservation):
            raise ValueError("Typed raw process observation required")
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            stored = self._tool_operation(
                conn, operation.scope, operation.task_run_id, operation.operation_id
            )
            if stored != operation:
                raise RunConflictError("Tool observation does not match its original intent")
            identity = str(uuid4())
            conn.execute(
                "INSERT INTO tool_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identity,
                    operation.operation_id,
                    sqlite3.Binary(observation.stdout),
                    sqlite3.Binary(observation.stderr),
                    observation.stdout_seen,
                    observation.stderr_seen,
                    int(observation.stdout_eof),
                    int(observation.stderr_eof),
                    observation.exit_code,
                    int(observation.timed_out),
                    observation.capture_error,
                    datetime.now(UTC).isoformat(),
                    hashlib.sha256(observation.stdout).hexdigest(),
                    hashlib.sha256(observation.stderr).hexdigest(),
                ),
            )
            return identity

    def get_tool_observation(
        self, scope: RunScope, task_run_id: str, operation_id: str, tool_attempt_id: str
    ) -> StoredToolObservation:
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN")
            operation = self._tool_operation(conn, scope, task_run_id, operation_id)
            if operation.tool_attempt_id != tool_attempt_id:
                raise LookupError("Tool capture unavailable in this scope")
            row = conn.execute(
                "SELECT * FROM tool_observations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if row is None:
                raise LookupError("Tool capture unavailable in this scope")
            observation = ProcessObservation(
                row["stdout"],
                row["stderr"],
                row["stdout_seen"],
                row["stderr_seen"],
                bool(row["stdout_eof"]),
                bool(row["stderr_eof"]),
                row["exit_code"],
                bool(row["timed_out"]),
                row["capture_error"],
            )
            if (
                hashlib.sha256(observation.stdout).hexdigest() != row["stdout_digest"]
                or hashlib.sha256(observation.stderr).hexdigest() != row["stderr_digest"]
            ):
                raise RuntimeError("Stored raw process digest is invalid")
            return StoredToolObservation(
                operation,
                row["result_id"],
                observation,
                row["stdout_digest"],
                row["stderr_digest"],
                row["captured_at"],
            )

    def submit_result(
        self,
        scope: RunScope,
        task_run_id: str,
        *,
        expected_version: int,
        report: ProducerReport,
        reason_code: str,
    ) -> TaskRun:
        if not isinstance(report, ProducerReport):
            raise ValueError("Typed producer report required")
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            run = self._record(scope, self._owned_row(conn, scope, task_run_id))
            if run.version != expected_version or run.state != RunState.RUNNING:
                raise RunConflictError("Submission requires the current running version")
            identity, now = str(uuid4()), datetime.now(UTC).isoformat()
            payload = json.dumps(
                {
                    "schema_version": 1,
                    "submission_id": identity,
                    "principal_id": scope.principal_id,
                    "session_id": scope.session_id,
                    "task_id": run.task_id,
                    "task_revision": run.task_revision,
                    "task_run_id": run.task_run_id,
                    "attempt_id": run.attempt_id,
                    "version": run.version + 1,
                    "created_at": now,
                    "producer_kind": report.kind.value,
                    "text": report.text,
                    "failure_code": report.failure_code,
                    "failure_message": report.failure_message,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            encoded = payload.encode("utf-8")
            if len(encoded) > 1024 * 1024:
                raise ValueError("Producer report exceeds submission storage limit")
            conn.execute(
                "INSERT INTO run_submissions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    identity,
                    run.task_run_id,
                    run.version + 1,
                    payload,
                    hashlib.sha256(encoded).hexdigest(),
                    now,
                ),
            )
            return self._transition(
                conn,
                scope,
                task_run_id,
                expected_version=expected_version,
                state=RunState.RESULT_SUBMITTED,
                authority=TransitionAuthority.RUNTIME,
                reason_code=reason_code,
                now=now,
                submission_id=identity,
            )

    def get_submission(self, scope: RunScope, task_run_id: str) -> RunSubmission:
        """Read the scoped producer report; absence never authorizes execution/replay."""
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN")
            row = conn.execute(
                "SELECT s.* FROM run_submissions s JOIN task_runs r USING(task_run_id) "
                "JOIN task_revisions t ON t.task_id=r.task_id AND t.revision=r.task_revision "
                "WHERE s.task_run_id=? AND t.principal_id=? AND t.session_id=?",
                (task_run_id, scope.principal_id, scope.session_id),
            ).fetchone()
            if row is None:
                raise LookupError("Submission unavailable in this scope")
            run = self._record(scope, self._owned_row(conn, scope, task_run_id))
            event = conn.execute(
                "SELECT * FROM run_transitions WHERE task_run_id=? AND version=?",
                (task_run_id, row["version"]),
            ).fetchone()
            digest = hashlib.sha256(row["payload"].encode("utf-8")).hexdigest()
            data = json.loads(row["payload"])
            expected = {
                "schema_version": 1,
                "submission_id": row["submission_id"],
                "principal_id": scope.principal_id,
                "session_id": scope.session_id,
                "task_id": run.task_id,
                "task_revision": run.task_revision,
                "task_run_id": run.task_run_id,
                "attempt_id": run.attempt_id,
                "version": row["version"],
                "created_at": row["created_at"],
            }
            if (
                digest != row["content_digest"]
                or not isinstance(data, dict)
                or set(data)
                != set(expected) | {"producer_kind", "text", "failure_code", "failure_message"}
                or any(
                    data.get(key) != value or type(data.get(key)) is not type(value)
                    for key, value in expected.items()
                )
                or event is None
                or event["state"] != RunState.RESULT_SUBMITTED
                or event["submission_id"] != row["submission_id"]
                or event["occurred_at"] != row["created_at"]
            ):
                raise RuntimeError("Stored submission binding/integrity is invalid")
            report = ProducerReport(
                ProducerKind(data["producer_kind"]),
                data["text"],
                data["failure_code"],
                data["failure_message"],
            )
            return RunSubmission(
                row["submission_id"],
                run.task_id,
                run.task_revision,
                run.task_run_id,
                run.attempt_id,
                scope,
                row["version"],
                row["created_at"],
                report,
                digest,
            )

    def admit(
        self,
        scope: RunScope,
        *,
        request_key: str,
        goal: str,
        mode: str,
        request_fingerprint: str,
        require_new: bool = False,
    ) -> TaskRun:
        if len(request_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in request_fingerprint
        ):
            raise ValueError("Invalid accepted request fingerprint")
        if not request_key.strip() or len(request_key) > 128:
            raise ValueError("Invalid admission request key")
        if not goal.strip() or len(goal.encode("utf-8")) > 1024 * 1024:
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
                record = self._record(scope, self._owned_row(conn, scope, row["task_run_id"]))
                if require_new:
                    raise RunAlreadyAdmittedError(
                        "Admission already exists; context recovery required"
                    )
                return record
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
                "INSERT INTO run_transitions "
                "(task_run_id, version, prior_state, state, authority, reason_code, occurred_at) "
                "VALUES (?, 0, NULL, ?, ?, ?, ?)",
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
        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._transition(
                conn,
                scope,
                task_run_id,
                expected_version=expected_version,
                state=state,
                authority=authority,
                reason_code=reason_code,
                now=datetime.now(UTC).isoformat(),
            )

    def _transition(
        self,
        conn: sqlite3.Connection,
        scope: RunScope,
        task_run_id: str,
        *,
        expected_version: int,
        state: RunState,
        authority: TransitionAuthority,
        reason_code: str,
        now: str,
        submission_id: str | None = None,
    ) -> TaskRun:
        if not isinstance(state, RunState) or not isinstance(authority, TransitionAuthority):
            raise ValueError("Typed transition state/authority required")
        if (
            not reason_code
            or len(reason_code) > 128
            or not all(char.isascii() and (char.isalnum() or char == "_") for char in reason_code)
        ):
            raise ValueError("Invalid transition reason code")
        current = self._record(scope, self._owned_row(conn, scope, task_run_id))
        if current.version != expected_version:
            raise RunConflictError("Stale run version")
        allowed = {
            (RunState.ADMITTED, RunState.RUNNING, TransitionAuthority.RUNTIME),
            (RunState.RUNNING, RunState.RECOVERY_REQUIRED, TransitionAuthority.RECOVERY),
            (
                RunState.RESULT_SUBMITTED,
                RunState.RECOVERY_REQUIRED,
                TransitionAuthority.RECOVERY,
            ),
        }
        if submission_id is not None:
            allowed.add((RunState.RUNNING, RunState.RESULT_SUBMITTED, TransitionAuthority.RUNTIME))
        if state == RunState.ABORTED and current.state != RunState.ABORTED:
            valid = authority in {TransitionAuthority.USER, TransitionAuthority.RECOVERY}
        else:
            valid = (current.state, state, authority) in allowed
        if not valid:
            raise RunConflictError("Transition/authority is not permitted")
        terminal_at = now if state == RunState.ABORTED else None
        conn.execute(
            "UPDATE task_runs SET state = ?, version = version + 1, terminal_at = ? "
            "WHERE task_run_id = ? AND version = ?",
            (state, terminal_at, task_run_id, expected_version),
        )
        conn.execute(
            "INSERT INTO run_transitions "
            "(task_run_id, version, prior_state, state, authority, reason_code, "
            "occurred_at, submission_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_run_id,
                current.version + 1,
                current.state,
                state,
                authority,
                reason_code,
                now,
                submission_id,
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
                    submission_id=row["submission_id"],
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
