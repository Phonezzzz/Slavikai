from __future__ import annotations

import hashlib
import json
import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from core.task_run_storage import RunConflictError, TaskRunStore
from shared.task_run import ProducerKind, ProducerReport, RunScope, RunState, TransitionAuthority


def test_submission_survives_reopen_bound_to_original_run(tmp_path) -> None:
    from shared.task_run import ProducerKind, ProducerReport

    path = tmp_path / "runs.db"
    store = TaskRunStore(path)
    scope = RunScope("alice", "session")
    run = store.admit(
        scope, request_key="one", goal="inspect", mode="ask", request_fingerprint="a" * 64
    )
    running = store.transition(
        scope,
        run.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="started",
    )
    report = ProducerReport(ProducerKind.MODEL, "Ответ\n\x00🦊", "stream_error", "partial")
    submitted = store.submit_result(
        scope,
        run.task_run_id,
        expected_version=running.version,
        report=report,
        reason_code="result_submitted",
    )
    reopened = TaskRunStore(path)
    saved = reopened.get_submission(scope, run.task_run_id)
    assert saved.report == report
    assert saved.scope == scope
    assert (saved.task_id, saved.task_revision, saved.task_run_id, saved.attempt_id) == (
        run.task_id,
        run.task_revision,
        run.task_run_id,
        run.attempt_id,
    )
    assert saved.version == submitted.version == 2
    assert saved.submission_id == reopened.events(scope, run.task_run_id)[-1].submission_id
    assert saved.created_at == reopened.events(scope, run.task_run_id)[-1].occurred_at
    assert submitted.state == RunState.RESULT_SUBMITTED
    assert submitted.terminal_at is None
    with sqlite3.connect(path) as conn:
        payload, digest = conn.execute(
            "SELECT payload,content_digest FROM run_submissions"
        ).fetchone()
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == digest == saved.content_digest
    assert json.loads(payload)["text"] == "Ответ\n\x00🦊"


def running_store(tmp_path):
    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("alice", "session")
    run = store.admit(
        scope, request_key="one", goal="inspect", mode="auto", request_fingerprint="a" * 64
    )
    run = store.transition(
        scope,
        run.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="started",
    )
    return store, scope, run


def test_generic_transition_cannot_submit_without_report(tmp_path) -> None:
    store, scope, run = running_store(tmp_path)
    with pytest.raises(RunConflictError):
        store.transition(
            scope,
            run.task_run_id,
            expected_version=1,
            state=RunState.RESULT_SUBMITTED,
            authority=TransitionAuthority.RUNTIME,
            reason_code="submitted",
        )
    assert store.get(scope, run.task_run_id) == run


@pytest.mark.parametrize("fault", ["payload", "history", "state"])
def test_submission_write_fault_is_atomic(tmp_path, fault) -> None:
    store, scope, run = running_store(tmp_path)
    table, op = {
        "payload": ("run_submissions", "INSERT"),
        "history": ("run_transitions", "INSERT"),
        "state": ("task_runs", "UPDATE"),
    }[fault]
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            f"CREATE TRIGGER fault BEFORE {op} ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'storage fault'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="storage fault"):
        store.submit_result(
            scope,
            run.task_run_id,
            expected_version=1,
            report=ProducerReport(ProducerKind.AUTO, "report"),
            reason_code="submitted",
        )
    reopened = TaskRunStore(store.path)
    assert reopened.get(scope, run.task_run_id) == run
    assert len(reopened.events(scope, run.task_run_id)) == 2
    with pytest.raises(LookupError, match="Submission unavailable"):
        reopened.get_submission(scope, run.task_run_id)
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM run_submissions").fetchone()[0] == 0


@pytest.mark.parametrize("foreign", [RunScope("bob", "session"), RunScope("alice", "other")])
def test_submission_scope_is_required_for_read_and_write(tmp_path, foreign) -> None:
    store, scope, run = running_store(tmp_path)
    report = ProducerReport(ProducerKind.AUTO, "private")
    with pytest.raises(LookupError):
        store.submit_result(
            foreign, run.task_run_id, expected_version=1, report=report, reason_code="submitted"
        )
    store.submit_result(
        scope, run.task_run_id, expected_version=1, report=report, reason_code="submitted"
    )
    for who, identity in [(foreign, run.task_run_id), (scope, "missing")]:
        with pytest.raises(LookupError, match="^Submission unavailable in this scope$"):
            store.get_submission(who, identity)
    assert store.get_submission(scope, run.task_run_id).report.text == "private"


@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_submission_payload_is_immutable(tmp_path, operation) -> None:
    store, scope, run = running_store(tmp_path)
    store.submit_result(
        scope,
        run.task_run_id,
        expected_version=1,
        report=ProducerReport(ProducerKind.AUTO, "private"),
        reason_code="submitted",
    )
    with (
        sqlite3.connect(store.path) as conn,
        pytest.raises(sqlite3.IntegrityError, match="immutable"),
    ):
        conn.execute(
            "DELETE FROM run_submissions"
            if operation == "DELETE"
            else "UPDATE run_submissions SET payload='{}'"
        )
    assert store.get_submission(scope, run.task_run_id).report.text == "private"


def concurrent_submit(args):
    path, scope, run_id, barrier, text = args
    writer = TaskRunStore(path)
    barrier.wait(timeout=10)
    try:
        writer.submit_result(
            scope,
            run_id,
            expected_version=1,
            report=ProducerReport(ProducerKind.AUTO, text),
            reason_code="submitted",
        )
        return text
    except RunConflictError:
        return None


def test_racing_submitters_commit_one_report_without_orphan(tmp_path) -> None:
    store, scope, run = running_store(tmp_path)
    with multiprocessing.Manager() as manager:
        barrier = manager.Barrier(2)
        with ProcessPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    concurrent_submit,
                    [
                        (store.path, scope, run.task_run_id, barrier, text)
                        for text in ["one", "two"]
                    ],
                )
            )
    assert len([x for x in results if x is not None]) == 1
    assert store.get_submission(scope, run.task_run_id).report.text in results
    assert len(store.events(scope, run.task_run_id)) == 3
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM run_submissions").fetchone()[0] == 1


@pytest.mark.parametrize("text", ["x" * (1024 * 1024), "🦊" * (256 * 1024), "\ud800"])
def test_report_is_not_truncated_or_replaced_on_invalid_storage_input(tmp_path, text) -> None:
    store, scope, run = running_store(tmp_path)
    with pytest.raises(ValueError):
        store.submit_result(
            scope,
            run.task_run_id,
            expected_version=1,
            report=ProducerReport(ProducerKind.AUTO, text),
            reason_code="submitted",
        )
    assert store.get(scope, run.task_run_id) == run
    with pytest.raises(LookupError):
        store.get_submission(scope, run.task_run_id)


def v2_fixture(path):
    with sqlite3.connect(path) as conn:
        conn.executescript((Path(__file__).parent / "fixtures/task_run_v2.sql").read_text())
        conn.execute(
            "INSERT INTO task_revisions VALUES ('task',1,'alice','session','key','goal','ask',?)",
            ("a" * 64,),
        )
        conn.execute(
            "INSERT INTO task_runs VALUES "
            "('run','task',1,'attempt','result_submitted',2,'time',NULL)"
        )
        for version, prior, state in [
            (0, None, "admitted"),
            (1, "admitted", "running"),
            (2, "running", "result_submitted"),
        ]:
            conn.execute(
                "INSERT INTO run_transitions VALUES ('run',?,?,?,?,?, 'time')",
                (version, prior, state, "runtime", "historical"),
            )


def test_v2_upgrade_preserves_history_without_fabricating_report(tmp_path) -> None:
    path = tmp_path / "old.db"
    v2_fixture(path)
    store = TaskRunStore(path)
    scope = RunScope("alice", "session")
    run = store.get(scope, "run")
    assert (run.task_id, run.attempt_id, run.version, run.state) == (
        "task",
        "attempt",
        2,
        RunState.RESULT_SUBMITTED,
    )
    assert [e.submission_id for e in store.events(scope, "run")] == [None] * 3
    with pytest.raises(LookupError, match="Submission unavailable"):
        store.get_submission(scope, "run")
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        assert conn.execute("SELECT count(*) FROM run_submissions").fetchone()[0] == 0
    # Old running/admitted records are not grandfathered into future payload-free submissions.
    with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError, match="binding"):
        conn.execute(
            "INSERT INTO run_transitions "
            "(task_run_id,version,state,authority,reason_code,occurred_at) "
            "VALUES ('run',3,'result_submitted','runtime','fake','later')"
        )


def test_upgrade_ddl_fault_rolls_back_version_and_schema(tmp_path, monkeypatch) -> None:
    path = tmp_path / "old.db"
    v2_fixture(path)
    upgrade = TaskRunStore._upgrade_submissions

    def fail(conn):
        upgrade(conn)
        raise OSError("upgrade fault")

    monkeypatch.setattr(TaskRunStore, "_upgrade_submissions", staticmethod(fail))
    with pytest.raises(OSError, match="upgrade fault"):
        TaskRunStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert len(conn.execute("PRAGMA table_info(run_transitions)").fetchall()) == 7
        assert conn.execute("SELECT count(*) FROM run_transitions").fetchone()[0] == 3
        assert (
            conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name='run_submissions'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("version", [2, 3])
def test_unrelated_versioned_database_is_not_adopted(tmp_path, version) -> None:
    path = tmp_path / "other.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE user_data(value TEXT); INSERT INTO user_data VALUES('keep'); "
            f"PRAGMA user_version={version};"
        )
    with pytest.raises(RuntimeError, match="schema"):
        TaskRunStore(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM user_data").fetchone()[0] == "keep"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == version


def test_stored_report_tampering_is_not_silently_accepted(tmp_path) -> None:
    store, scope, run = running_store(tmp_path)
    store.submit_result(
        scope,
        run.task_run_id,
        expected_version=1,
        report=ProducerReport(ProducerKind.AUTO, "original"),
        reason_code="submitted",
    )
    with sqlite3.connect(store.path) as conn:
        conn.execute("DROP TRIGGER immutable_submission_update")
        conn.execute("UPDATE run_submissions SET payload=replace(payload,'original','tampered')")
    with pytest.raises(RuntimeError, match="integrity"):
        store.get_submission(scope, run.task_run_id)


@pytest.mark.parametrize(
    "status", ["cancelled", "failed_worker", "failed_verifier", "failed_internal"]
)
def test_auto_producer_report_is_not_acceptance_and_omits_runtime_details(tmp_path, status) -> None:
    from core.agent_response import AgentResponse, ResponseFailure
    from core.auto_runtime import AutoRunOutcome
    from core.mwv.models import StopReasonCode
    from shared.auto_models import AutoRunStatus

    store, scope, run = running_store(tmp_path)
    outcome = AutoRunOutcome(
        "internal-runtime-text",
        AutoRunStatus(status),
        StopReasonCode.BUDGET_EXHAUSTED,
        None,
        ["private-next-step"],
    )
    report = AgentResponse(
        "public report", outcome, ResponseFailure("error", "partial")
    ).producer_report()
    submitted = store.submit_result(
        scope, run.task_run_id, expected_version=1, report=report, reason_code="submitted"
    )
    assert submitted.state == RunState.RESULT_SUBMITTED and submitted.terminal_at is None
    saved = store.get_submission(scope, run.task_run_id)
    assert saved.report == ProducerReport(ProducerKind.AUTO, "public report", "error", "partial")
    with sqlite3.connect(store.path) as conn:
        payload = conn.execute("SELECT payload FROM run_submissions").fetchone()[0]
    assert "internal-runtime-text" not in payload and "private-next-step" not in payload


@pytest.mark.parametrize("result", [None, object(), "model", {"type": "auto"}])
def test_untyped_producer_is_not_inferred_from_prose(result) -> None:
    from core.agent_response import AgentResponse

    with pytest.raises(ValueError, match="supported typed runtime"):
        AgentResponse("completed", result).producer_report()
