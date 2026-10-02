from __future__ import annotations

import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor

import pytest

from core.task_run_storage import RunConflictError, TaskRunStore
from shared.task_run import RunScope, RunState, TransitionAuthority


def test_admission_is_durable_scoped_and_idempotent(tmp_path) -> None:
    path = tmp_path / "runs.db"
    scope = RunScope(principal_id="alice", session_id="session-a")
    store = TaskRunStore(path)
    run = store.admit(scope, request_key="request-1", goal="inspect project", mode="ask")

    assert run.state == RunState.ADMITTED
    assert run.version == 0
    assert run.task_revision == 1
    assert len({run.task_id, run.task_run_id, run.attempt_id}) == 3
    reopened = TaskRunStore(path)
    assert reopened.get(scope, run.task_run_id) == run
    assert reopened.admit(scope, request_key="request-1", goal="inspect project", mode="ask") == run
    events = reopened.events(scope, run.task_run_id)
    assert len(events) == 1
    assert events[0].version == 0
    assert events[0].state == RunState.ADMITTED


def test_transition_is_versioned_and_durable_without_inferred_completion(tmp_path) -> None:
    path = tmp_path / "runs.db"
    store = TaskRunStore(path)
    scope = RunScope("alice", "session-a")
    run = store.admit(scope, request_key="1", goal="inspect", mode="auto")
    running = store.transition(
        scope,
        run.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="execution_started",
    )
    submitted = store.transition(
        scope,
        run.task_run_id,
        expected_version=running.version,
        state=RunState.RESULT_SUBMITTED,
        authority=TransitionAuthority.RUNTIME,
        reason_code="result_submitted",
    )
    assert submitted.terminal_at is None
    reopened = TaskRunStore(path)
    assert reopened.get(scope, run.task_run_id) == submitted
    assert [event.version for event in reopened.events(scope, run.task_run_id)] == [0, 1, 2]
    assert reopened.events(scope, run.task_run_id)[-1].prior_state == RunState.RUNNING


def test_materialized_state_without_matching_history_is_rejected(tmp_path) -> None:
    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("alice", "session-a")
    run = store.admit(scope, request_key="1", goal="inspect", mode="ask")
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE task_runs SET state = 'running' WHERE task_run_id = ?", (run.task_run_id,)
        )
    with pytest.raises(RuntimeError, match="history"):
        store.get(scope, run.task_run_id)
    with pytest.raises(RuntimeError, match="history"):
        store.admit(scope, request_key="1", goal="inspect", mode="ask")


@pytest.mark.parametrize("scope", [RunScope("bob", "session-a"), RunScope("alice", "session-b")])
def test_foreign_scope_cannot_read_history_or_transition(tmp_path, scope) -> None:
    store = TaskRunStore(tmp_path / "runs.db")
    owner = RunScope("alice", "session-a")
    run = store.admit(owner, request_key="1", goal="inspect", mode="ask")
    with pytest.raises(LookupError):
        store.get(scope, run.task_run_id)
    with pytest.raises(LookupError):
        store.events(scope, run.task_run_id)
    with pytest.raises(LookupError):
        store.transition(
            scope,
            run.task_run_id,
            expected_version=0,
            state=RunState.RUNNING,
            authority=TransitionAuthority.RUNTIME,
            reason_code="execution_started",
        )
    assert store.get(owner, run.task_run_id) == run
    with pytest.raises(LookupError, match="Run unavailable in this scope"):
        store.get(scope, "unknown-run")


@pytest.mark.parametrize(
    "field,value", [("goal", "changed"), ("mode", "auto"), ("session", "other")]
)
def test_admission_key_cannot_be_reused_for_another_request(tmp_path, field, value) -> None:
    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("alice", "session-a")
    run = store.admit(scope, request_key="1", goal="inspect", mode="ask")
    args = {"goal": "inspect", "mode": "ask", "session": "session-a"}
    args[field] = value
    with pytest.raises(RunConflictError):
        store.admit(
            RunScope("alice", args["session"]),
            request_key="1",
            goal=args["goal"],
            mode=args["mode"],
        )
    assert store.get(scope, run.task_run_id) == run
    other = store.admit(RunScope("bob", "session-a"), request_key="1", goal="inspect", mode="ask")
    assert other.task_id != run.task_id


def _concurrent_admit(args):
    path, scope, barrier = args
    store = TaskRunStore(path)
    barrier.wait()
    return store.admit(scope, request_key="1", goal="inspect", mode="ask")


def _concurrent_start(args):
    path, scope, run_id, barrier = args
    store = TaskRunStore(path)
    barrier.wait()
    try:
        return store.transition(
            scope,
            run_id,
            expected_version=0,
            state=RunState.RUNNING,
            authority=TransitionAuthority.RUNTIME,
            reason_code="execution_started",
        )
    except RunConflictError:
        return None


def test_concurrent_admission_creates_exactly_one_identity(tmp_path) -> None:
    path = tmp_path / "runs.db"
    store = TaskRunStore(path)
    scope = RunScope("alice", "session-a")
    with multiprocessing.Manager() as manager:
        barrier = manager.Barrier(4)
        with ProcessPoolExecutor(max_workers=4) as pool:
            runs = list(pool.map(_concurrent_admit, [(path, scope, barrier)] * 4))
    assert len({run.task_run_id for run in runs}) == 1
    assert len(store.events(scope, runs[0].task_run_id)) == 1


def test_concurrent_transition_fences_stale_writer(tmp_path) -> None:
    path = tmp_path / "runs.db"
    store = TaskRunStore(path)
    scope = RunScope("alice", "session-a")
    run = store.admit(scope, request_key="1", goal="inspect", mode="auto")
    with multiprocessing.Manager() as manager:
        barrier = manager.Barrier(2)
        with ProcessPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(_concurrent_start, [(path, scope, run.task_run_id, barrier)] * 2)
            )
    assert sum(result is not None for result in results) == 1
    assert store.get(scope, run.task_run_id).version == 1
    assert len(store.events(scope, run.task_run_id)) == 2


def test_terminal_abort_cannot_be_resurrected(tmp_path) -> None:
    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("alice", "session-a")
    run = store.admit(scope, request_key="1", goal="inspect", mode="auto")
    aborted = store.transition(
        scope,
        run.task_run_id,
        expected_version=0,
        state=RunState.ABORTED,
        authority=TransitionAuthority.USER,
        reason_code="user_abort",
    )
    assert aborted.terminal_at is not None
    for version in (0, aborted.version):
        with pytest.raises(RunConflictError):
            store.transition(
                scope,
                run.task_run_id,
                expected_version=version,
                state=RunState.RUNNING,
                authority=TransitionAuthority.RUNTIME,
                reason_code="execution_started",
            )
    assert store.get(scope, run.task_run_id) == aborted
    assert store.admit(scope, request_key="1", goal="inspect", mode="auto") == aborted


def test_recovery_requires_explicit_authority_and_does_not_resume(tmp_path) -> None:
    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("alice", "session-a")
    run = store.admit(scope, request_key="1", goal="inspect", mode="auto")
    run = store.transition(
        scope,
        run.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="execution_started",
    )
    with pytest.raises(RunConflictError):
        store.transition(
            scope,
            run.task_run_id,
            expected_version=1,
            state=RunState.RECOVERY_REQUIRED,
            authority=TransitionAuthority.RUNTIME,
            reason_code="worker_lost",
        )
    recovered = store.transition(
        scope,
        run.task_run_id,
        expected_version=1,
        state=RunState.RECOVERY_REQUIRED,
        authority=TransitionAuthority.RECOVERY,
        reason_code="worker_lost",
    )
    assert recovered.terminal_at is None
    assert TaskRunStore(store.path).get(scope, run.task_run_id) == recovered
    with pytest.raises(RunConflictError):
        store.transition(
            scope,
            run.task_run_id,
            expected_version=2,
            state=RunState.RUNNING,
            authority=TransitionAuthority.RUNTIME,
            reason_code="execution_started",
        )


@pytest.mark.parametrize("during_admission", [True, False])
def test_history_write_failure_rolls_back_materialized_state(tmp_path, during_admission) -> None:
    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("alice", "session-a")
    run = (
        None
        if during_admission
        else store.admit(scope, request_key="1", goal="inspect", mode="ask")
    )
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "CREATE TRIGGER fail_history BEFORE INSERT ON run_transitions "
            "BEGIN SELECT RAISE(ABORT, 'disk failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="disk failure"):
        if run is None:
            store.admit(scope, request_key="1", goal="inspect", mode="ask")
        else:
            store.transition(
                scope,
                run.task_run_id,
                expected_version=0,
                state=RunState.RUNNING,
                authority=TransitionAuthority.RUNTIME,
                reason_code="execution_started",
            )
    if run is not None:
        assert store.get(scope, run.task_run_id) == run
        assert len(store.events(scope, run.task_run_id)) == 1
    else:
        with sqlite3.connect(store.path) as conn:
            for table in ("task_revisions", "task_runs", "run_transitions"):
                assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.parametrize("table", ["task_revisions", "run_transitions"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_revision_and_history_are_immutable(tmp_path, table, operation) -> None:
    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("alice", "session-a")
    run = store.admit(scope, request_key="1", goal="inspect", mode="ask")
    statement = (
        f"DELETE FROM {table}"
        if operation == "DELETE"
        else f"UPDATE {table} SET task_run_id=task_run_id"
        if table == "run_transitions"
        else f"UPDATE {table} SET goal=goal"
    )
    with (
        sqlite3.connect(store.path) as conn,
        pytest.raises(sqlite3.IntegrityError, match="immutable"),
    ):
        conn.execute(statement)
    assert store.get(scope, run.task_run_id) == run


def test_store_permissions_and_symlink_rejection(tmp_path) -> None:
    path = tmp_path / "private" / "runs.db"
    TaskRunStore(path)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    link = tmp_path / "link.db"
    link.symlink_to(path)
    with pytest.raises(OSError):
        TaskRunStore(link)


def test_unrelated_or_unknown_database_is_not_adopted(tmp_path) -> None:
    path = tmp_path / "other.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE user_data(value TEXT)")
        conn.execute("INSERT INTO user_data VALUES ('keep')")
    with pytest.raises(RuntimeError, match="schema"):
        TaskRunStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM user_data").fetchone()[0] == "keep"
        conn.execute("PRAGMA user_version = 99")
    with pytest.raises(RuntimeError, match="schema"):
        TaskRunStore(path)
