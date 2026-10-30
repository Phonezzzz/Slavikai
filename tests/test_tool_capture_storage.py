from __future__ import annotations

import hashlib

import pytest

from core.task_run_storage import TaskRunStore
from shared.task_run import RunScope, RunState, TransitionAuthority


def test_raw_observation_reopens_with_original_intent_and_rejects_foreign_scope(tmp_path):
    from shared.tool_capture import ProcessObservation

    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("synthetic-alice", "session")
    admitted = store.admit(
        scope, request_key="key", goal="synthetic run", mode="auto", request_fingerprint="a" * 64
    )
    run = store.transition(
        scope,
        admitted.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="started",
    )
    operation = store.begin_tool_operation(run, model_call_id="model-label", request_path="emit.py")
    observation = ProcessObservation(b" \xff\r\n", b"err\x00\n", 4, 5, True, True, 23, False)
    result_id = store.record_tool_observation(operation, observation)
    reopened = TaskRunStore(store.path)
    stored = reopened.get_tool_observation(
        scope, run.task_run_id, operation.operation_id, operation.tool_attempt_id
    )
    assert stored.result_id == result_id
    assert stored.operation == operation
    assert stored.operation.run_attempt_id == run.attempt_id
    assert stored.observation == observation
    assert stored.stdout_digest == hashlib.sha256(b" \xff\r\n").hexdigest()
    assert stored.stderr_digest == hashlib.sha256(b"err\x00\n").hexdigest()
    assert reopened.get(scope, run.task_run_id) == run
    with pytest.raises(LookupError):
        reopened.get_tool_observation(
            RunScope("synthetic-bob", "session"),
            run.task_run_id,
            operation.operation_id,
            operation.tool_attempt_id,
        )
    with pytest.raises(LookupError):
        reopened.get_tool_observation(
            scope, run.task_run_id, operation.operation_id, "foreign-attempt"
        )


def test_additive_v3_upgrade_preserves_original_history_and_has_no_fake_tool_capture(tmp_path):
    import sqlite3
    from pathlib import Path

    path = tmp_path / "v3.db"
    with sqlite3.connect(path) as conn:
        conn.executescript((Path(__file__).parent / "fixtures/task_run_v3.sql").read_text())
        originals = {
            table: conn.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("task_revisions", "task_runs", "run_transitions", "run_submissions")
        }
    store = TaskRunStore(path)
    scope = RunScope("alice", "session")
    run_id = "a17b0a38-0af1-475e-a0a1-83f225cbcec0"
    assert store.get_submission(scope, run_id).report.text == "producer-v3"
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        for table, rows in originals.items():
            assert conn.execute(f"SELECT * FROM {table}").fetchall() == rows
        assert conn.execute("SELECT count(*) FROM tool_operations").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM tool_observations").fetchone()[0] == 0


def test_duplicate_model_label_does_not_reuse_intent_and_stale_run_cannot_admit(tmp_path):
    from core.task_run_storage import RunConflictError

    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("synthetic-alice", "session")
    admitted = store.admit(
        scope, request_key="key", goal="synthetic", mode="auto", request_fingerprint="b" * 64
    )
    run = store.transition(
        scope,
        admitted.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="started",
    )
    first = store.begin_tool_operation(run, model_call_id="same-label", request_path="emit.py")
    second = store.begin_tool_operation(run, model_call_id="same-label", request_path="emit.py")
    assert first.operation_id != second.operation_id
    assert first.tool_attempt_id != second.tool_attempt_id
    with pytest.raises(RunConflictError):
        store.begin_tool_operation(admitted, model_call_id="same-label", request_path="emit.py")


@pytest.mark.parametrize("table", ["tool_operations", "tool_observations"])
def test_stored_intent_and_raw_observation_are_immutable(tmp_path, table):
    import sqlite3

    from shared.tool_capture import ProcessObservation

    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("synthetic-alice", "session")
    admitted = store.admit(
        scope, request_key="key", goal="synthetic", mode="auto", request_fingerprint="c" * 64
    )
    run = store.transition(
        scope,
        admitted.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="started",
    )
    operation = store.begin_tool_operation(run, model_call_id="label", request_path="emit.py")
    store.record_tool_observation(
        operation, ProcessObservation(b"out", b"err", 3, 3, True, True, 0, False)
    )
    for query in (f"DELETE FROM {table}", f"UPDATE {table} SET operation_id='changed'"):
        with sqlite3.connect(store.path) as conn, pytest.raises(sqlite3.IntegrityError):
            conn.execute(query)
    assert (
        store.get_tool_observation(
            scope, run.task_run_id, operation.operation_id, operation.tool_attempt_id
        ).observation.stdout
        == b"out"
    )


def test_late_observation_cannot_resurrect_aborted_run_or_authorize_next_intent(tmp_path):
    from shared.tool_capture import ProcessObservation

    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("synthetic-alice", "session")
    admitted = store.admit(
        scope, request_key="key", goal="synthetic", mode="auto", request_fingerprint="d" * 64
    )
    run = store.transition(
        scope,
        admitted.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="started",
    )
    operation = store.begin_tool_operation(run, model_call_id="label", request_path="emit.py")
    aborted = store.transition(
        scope,
        run.task_run_id,
        expected_version=1,
        state=RunState.ABORTED,
        authority=TransitionAuthority.USER,
        reason_code="abort",
    )
    observation = ProcessObservation(b"late", b"", 4, 0, True, True, 0, False)
    store.record_tool_observation(operation, observation)
    assert (
        store.get_tool_observation(
            scope, run.task_run_id, operation.operation_id, operation.tool_attempt_id
        ).observation
        == observation
    )
    assert store.get(scope, run.task_run_id) == aborted
    with pytest.raises(ValueError):
        store.begin_tool_operation(run, model_call_id="new-label", request_path="emit.py")


def test_capture_upgrade_failure_preserves_v3_schema_and_original_report(tmp_path, monkeypatch):
    import sqlite3
    from pathlib import Path

    path = tmp_path / "v3.db"
    with sqlite3.connect(path) as conn:
        conn.executescript((Path(__file__).parent / "fixtures/task_run_v3.sql").read_text())
        report = conn.execute("SELECT * FROM run_submissions").fetchall()
    upgrade = TaskRunStore._upgrade_capture

    def fail(conn):
        upgrade(conn)
        raise OSError("synthetic upgrade fault")

    monkeypatch.setattr(TaskRunStore, "_upgrade_capture", staticmethod(fail))
    with pytest.raises(OSError, match="synthetic upgrade fault"):
        TaskRunStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        assert conn.execute("SELECT * FROM run_submissions").fetchall() == report
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE name IN ('tool_operations','tool_observations')"
            ).fetchall()
            == []
        )


def test_raw_payload_digest_is_verified_when_reading_committed_evidence(tmp_path):
    import sqlite3

    from shared.tool_capture import ProcessObservation

    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("synthetic-alice", "session")
    admitted = store.admit(
        scope, request_key="key", goal="synthetic", mode="auto", request_fingerprint="e" * 64
    )
    run = store.transition(
        scope,
        admitted.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="started",
    )
    operation = store.begin_tool_operation(run, model_call_id="label", request_path="emit.py")
    store.record_tool_observation(
        operation, ProcessObservation(b"original", b"", 8, 0, True, True, 0, False)
    )
    with sqlite3.connect(store.path) as conn:
        conn.execute("DROP TRIGGER immutable_observation_update")
        conn.execute("UPDATE tool_observations SET stdout=?", (sqlite3.Binary(b"tampered"),))
    with pytest.raises(RuntimeError, match="digest"):
        store.get_tool_observation(
            scope, run.task_run_id, operation.operation_id, operation.tool_attempt_id
        )
