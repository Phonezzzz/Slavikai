"""Bridge effect recorder (E03-T08, rev. 2 §1 + owner conditions).

Classification is tested with canned bridge answers (no live
daemon): the recorder's job is to turn an answer into the
journal's outcome trichotomy and persist it. The live-daemon
answers are covered by test_host_bridge_audit.py.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from core.bridge_effect_recorder import (
    OBSERVATION_SOURCE_BINDER,
    OBSERVATION_SOURCE_MANUAL,
    OBSERVATION_SOURCE_POLL_LOOP,
    BridgeEffectRecorder,
)
from core.bridge_effect_store import BridgeEffectStore

_OWNER = "owner"
_BOOT = "b" * 32


@pytest.fixture()
def recorder(tmp_path: Path) -> BridgeEffectRecorder:
    return BridgeEffectRecorder(BridgeEffectStore(tmp_path / "effects.db"))


def _effect(*, seq: int = 1, kind: str = "fs.write", **fields: object) -> dict[str, object]:
    effect: dict[str, object] = {
        "seq": seq,
        "boot_id": _BOOT,
        "kind": kind,
        "at": "2026-11-25T00:00:00+00:00",
        "principal_id": _OWNER,
        "run_id": None,
    }
    effect.update(fields)
    return effect


def _write_result(
    *, effect: dict[str, object] | None = None, **fields: object
) -> dict[str, object]:
    result: dict[str, object] = {
        "root_id": "work",
        "path": "a.txt",
        "size": 1,
        "bytes_written": 1,
        "committed": True,
    }
    if effect is not None:
        result["effect"] = effect
    result.update(fields)
    return result


# --- success classification --------------------------------------------------


def test_effect_success_records_effect(recorder: BridgeEffectRecorder) -> None:
    outcome = recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt", "content": "secret-bytes"},
        result=_write_result(effect=_effect(seq=1, root_id="work", path="a.txt", size=1)),
    )
    assert outcome.outcome == "effect"
    assert outcome.seq == 1
    assert outcome.audit_gap is False
    assert outcome.duplicate is False


def test_metadata_projection_excludes_content_and_args(recorder: BridgeEffectRecorder) -> None:
    """Owner condition 6 (structural): the params projection is a
    fixed whitelist — content, patch text and args can never reach
    the journal."""
    recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={
            "root_id": "work",
            "path": "a.txt",
            "content": "TOP-SECRET-CONTENT",
            "mode": "create",
            "write_id": "w1",
            "commit": True,
        },
        result=_write_result(effect=_effect(seq=1)),
    )
    recorder.record_success(
        capability="host.process.run",
        request_id="req-2",
        principal_id=_OWNER,
        run_id="run-1",
        params={"profile_id": "sleep", "args": ["--api-key", "SECRET"], "timeout_s": 5},
        result={
            "process_id": "p1",
            "profile_id": "sleep",
            "state": "running",
            "started_at": "2026-11-25T00:00:00+00:00",
            "run_id": "run-1",
            "effect": _effect(seq=2, kind="process.run", process_id="p1", profile_id="sleep"),
        },
    )
    store = recorder._store
    first = store.get_record(1)
    second = store.get_record(2)
    assert first is not None and second is not None
    assert first["params"] == {
        "root_id": "work",
        "path": "a.txt",
        "mode": "create",
        "write_id": "w1",
        "commit": True,
    }
    assert "content" not in first["params"]
    assert second["params"] == {"profile_id": "sleep", "timeout_s": 5}
    assert "args" not in second["params"]
    assert json.dumps(first["params"]).find("TOP-SECRET") == -1
    assert json.dumps(second["params"]).find("SECRET") == -1


def test_missing_effect_block_on_commit_is_unknown(recorder: BridgeEffectRecorder) -> None:
    outcome = recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt"},
        result=_write_result(),
    )
    assert outcome.outcome == "unknown"
    record = recorder._store.get_record(int(outcome.seq or 0))
    assert record is not None
    assert record["unknown_reason"] == "missing_effect_block"


def test_chunk_with_parents_effect_is_effect(recorder: BridgeEffectRecorder) -> None:
    """Codex round 2, finding 2 (recorder pin): a chunk answer
    that carries an fs.write effect for parent directories it
    created is a confirmed mutation — the effect branch accepts
    it, whatever the committed flag says. A chunk without an
    effect stays no_effect (test_chunk_abort_and_dry_run_are_no_effect)."""
    outcome = recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "new-parent/file.txt", "create_parents": True},
        result={
            "root_id": "work",
            "path": "new-parent/file.txt",
            "size": 1,
            "bytes_written": 1,
            "committed": False,
            "write_id": "w1",
            "effect": _effect(
                seq=1, committed=False, parents_created=["new-parent"], size=1, bytes_written=1
            ),
        },
    )
    assert outcome.outcome == "effect"


def test_chunk_abort_and_dry_run_are_no_effect(recorder: BridgeEffectRecorder) -> None:
    chunk = recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt"},
        result={
            "root_id": "work",
            "path": "a.txt",
            "size": 0,
            "bytes_written": 1,
            "write_id": "w1",
        },
    )
    assert chunk.outcome == "no_effect"
    abort = recorder.record_success(
        capability="host.fs.write",
        request_id="req-2",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt", "write_id": "w1", "abort": True},
        result={"root_id": "work", "path": "a.txt", "size": 0, "bytes_written": 0, "aborted": True},
    )
    assert abort.outcome == "no_effect"
    dry = recorder.record_success(
        capability="host.fs.patch",
        request_id="req-3",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt", "dry_run": True},
        result={"root_id": "work", "path": "a.txt", "applied": False, "dry_run": True, "size": 1},
    )
    assert dry.outcome == "no_effect"


def test_move_delete_and_stop_without_effect_are_unknown(recorder: BridgeEffectRecorder) -> None:
    for capability, result in (
        (
            "host.fs.move",
            {
                "root_id": "work",
                "path": "a.txt",
                "dest_root_id": "work",
                "dest_path": "b.txt",
                "moved": True,
            },
        ),
        (
            "host.fs.delete",
            {"root_id": "work", "path": "a.txt", "trash_name": "a.txt", "recoverable": True},
        ),
        ("host.process.stop", {"run_id": "run-1", "processes": [], "all_final": True}),
    ):
        outcome = recorder.record_success(
            capability=capability,
            request_id=f"req-{capability}",
            principal_id=_OWNER,
            run_id=None,
            params={"root_id": "work", "path": "a.txt"},
            result=result,
        )
        assert outcome.outcome == "unknown", capability


def test_kill_final_state_without_effect_is_no_effect(recorder: BridgeEffectRecorder) -> None:
    for state in ("exited", "killed", "timeout", "spawn_failed"):
        outcome = recorder.record_success(
            capability="host.process.kill",
            request_id=f"req-kill-{state}",
            principal_id=_OWNER,
            run_id=None,
            params={"process_id": "p1"},
            result={"process_id": "p1", "state": state, "exit_code": 0, "signal": None},
        )
        assert outcome.outcome == "no_effect", state


def test_kill_non_final_without_effect_is_unknown(recorder: BridgeEffectRecorder) -> None:
    outcome = recorder.record_success(
        capability="host.process.kill",
        request_id="req-kill",
        principal_id=_OWNER,
        run_id=None,
        params={"process_id": "p1"},
        result={"process_id": "p1", "state": "killing", "exit_code": None, "signal": None},
    )
    assert outcome.outcome == "unknown"


def test_run_without_effect_is_unknown(recorder: BridgeEffectRecorder) -> None:
    """A run result without an effect is unknown even for a
    spawn_failed state: the spawn_destroyed path means the command
    may have executed, so no-effect cannot be proven."""
    outcome = recorder.record_success(
        capability="host.process.run",
        request_id="req-run",
        principal_id=_OWNER,
        run_id="run-1",
        params={"profile_id": "sleep"},
        result={
            "process_id": "p1",
            "profile_id": "sleep",
            "state": "spawn_failed",
            "started_at": "",
            "run_id": "run-1",
        },
    )
    assert outcome.outcome == "unknown"


def test_malformed_effect_is_treated_as_missing(recorder: BridgeEffectRecorder) -> None:
    outcome = recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt"},
        result=_write_result(effect={"seq": 0, "boot_id": _BOOT, "kind": "fs.write"}),
    )
    assert outcome.outcome == "unknown"


# --- error classification ------------------------------------------------------


@pytest.mark.parametrize(
    "code,is_retry,expected",
    [
        ("invalid_params", False, "no_effect"),
        ("invalid_params", True, "no_effect"),
        ("run_stopped", False, "no_effect"),
        ("patch_mismatch", False, "no_effect"),
        ("already_exists", False, "no_effect"),
        ("trash_unavailable", False, "no_effect"),
        ("process_unavailable", False, "no_effect"),
        ("effect_build_failed", False, "no_effect"),
        ("unknown_profile", False, "no_effect"),
        ("invalid_args", False, "no_effect"),
        ("binding_mismatch", False, "no_effect"),
        ("too_many_processes", False, "no_effect"),
        ("is_a_directory", False, "no_effect"),
        ("path_outside_root", False, "no_effect"),
        ("write_incomplete", False, "no_effect"),
        ("invalid_path", False, "no_effect"),
        ("unknown_root", False, "no_effect"),
        ("root_read_only", False, "no_effect"),
        ("process_not_found", False, "no_effect"),
        ("process_not_found", True, "unknown"),
        ("request_timeout", False, "unknown"),
        ("connection_closed", False, "unknown"),
        ("internal_error", False, "unknown"),
        ("spawn_failed", False, "unknown"),
        ("io_error", False, "unknown"),
        ("some_future_code", False, "unknown"),
    ],
)
def test_error_classification_table(
    recorder: BridgeEffectRecorder, code: str, is_retry: bool, expected: str
) -> None:
    if is_retry:
        # A prior row with the same request_id makes this a retry.
        recorder.record_transport_failure(
            capability="host.process.kill",
            request_id="req-1",
            principal_id=_OWNER,
            run_id=None,
            params={"process_id": "p1"},
            reason="timeout",
        )
    outcome = recorder.record_error(
        capability="host.process.kill",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"process_id": "p1"},
        code=code,
        message=f"{code} happened",
    )
    assert outcome.outcome == expected, code


def test_unknown_write_id_retry_is_unknown(recorder: BridgeEffectRecorder) -> None:
    """The daemon raises invalid_params with the message
    'unknown write_id' (no distinct wire code exists — verified in
    host_bridge/fs.py). First call: no_effect. Retry: unknown."""
    first = recorder.record_error(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt", "write_id": "w-nope"},
        code="invalid_params",
        message="unknown write_id",
    )
    assert first.outcome == "no_effect"
    retry = recorder.record_error(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt", "write_id": "w-nope"},
        code="invalid_params",
        message="unknown write_id",
    )
    assert retry.outcome == "unknown"
    record = recorder._store.get_record(int(retry.seq or 0))
    assert record is not None
    assert record["unknown_reason"] == "unknown_write_id_on_retry"


# --- unknown resolution (owner condition 2) -----------------------------------


def test_retry_with_confirmed_effect_resolves_unknown(recorder: BridgeEffectRecorder) -> None:
    unknown = recorder.record_transport_failure(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt"},
        reason="timeout",
    )
    assert unknown.outcome == "unknown"
    effect = recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt"},
        result=_write_result(effect=_effect(seq=5, root_id="work", path="a.txt")),
    )
    assert effect.outcome == "effect"
    assert effect.resolved_unknown_seqs == [unknown.seq]
    attempt = recorder._store.get_record(int(unknown.seq or 0))
    assert attempt is not None
    assert attempt["outcome"] == "unknown"
    assert attempt["resolved_by_seq"] == effect.seq


def test_duplicate_effect_delivery_is_idempotent(recorder: BridgeEffectRecorder) -> None:
    first = recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt"},
        result=_write_result(effect=_effect(seq=1)),
    )
    second = recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt"},
        result=_write_result(effect=_effect(seq=1)),
    )
    assert first.seq == second.seq == 1
    assert second.duplicate is True
    assert recorder._store.count() == 1


# --- observations ---------------------------------------------------------------


def _seed_effect(recorder: BridgeEffectRecorder, *, seq: int = 1, kind: str = "fs.write") -> int:
    outcome = recorder.record_success(
        capability=f"host.{kind}",
        request_id=f"req-{seq}",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt"}
        if kind.startswith("fs.")
        else {"process_id": "p1"},
        result=_write_result(effect=_effect(seq=seq, kind=kind, root_id="work", path="a.txt"))
        if kind.startswith("fs.")
        else {
            "process_id": "p1",
            "profile_id": "sleep",
            "state": "running",
            "started_at": "2026-11-25T00:00:00+00:00",
            "run_id": None,
            "effect": _effect(seq=seq, kind=kind, process_id="p1"),
        },
    )
    assert outcome.seq is not None
    return outcome.seq


def test_observation_attach_poll_loop(recorder: BridgeEffectRecorder) -> None:
    seq = _seed_effect(recorder)
    observation_id = recorder.attach_observation(
        effect_seq=seq,
        source=OBSERVATION_SOURCE_POLL_LOOP,
        source_ref="poll:1234",
        payload={
            "subject": "a.txt",
            "observed_at": "2026-11-25T00:01:00+00:00",
            "state": "exists",
        },
    )
    assert observation_id == 1
    record = recorder._store.get_record(seq)
    assert record is not None
    assert len(record["observations"]) == 1
    assert record["observations"][0]["source"] == "poll-loop"
    # Idempotent: the same payload again returns the same id.
    again = recorder.attach_observation(
        effect_seq=seq,
        source=OBSERVATION_SOURCE_POLL_LOOP,
        source_ref="poll:1234",
        payload={
            "subject": "a.txt",
            "observed_at": "2026-11-25T00:01:00+00:00",
            "state": "exists",
        },
    )
    assert again == observation_id


def test_observation_subject_mismatch_is_rejected(recorder: BridgeEffectRecorder) -> None:
    seq = _seed_effect(recorder)
    with pytest.raises(ValueError):
        recorder.attach_observation(
            effect_seq=seq,
            source=OBSERVATION_SOURCE_MANUAL,
            source_ref=None,
            payload={
                "subject": "other.txt",
                "observed_at": "2026-11-25T00:01:00+00:00",
                "state": "exists",
            },
        )


def test_observation_for_stop_accepts_a_listed_process(recorder: BridgeEffectRecorder) -> None:
    outcome = recorder.record_success(
        capability="host.process.stop",
        request_id="req-stop",
        principal_id=_OWNER,
        run_id="run-1",
        params={"run_id": "run-1"},
        result={
            "run_id": "run-1",
            "processes": [{"process_id": "p9", "state": "killed", "exit_code": None, "signal": 15}],
            "all_final": True,
            "effect": _effect(
                seq=1,
                kind="process.stop",
                run_id="run-1",
                processes=[
                    {"process_id": "p9", "state": "killed", "exit_code": None, "signal": 15}
                ],
                all_final=True,
                mark_set=True,
            ),
        },
    )
    assert outcome.seq is not None
    # The run itself and any process the stop listed are valid subjects.
    recorder.attach_observation(
        effect_seq=outcome.seq,
        source=OBSERVATION_SOURCE_POLL_LOOP,
        source_ref=None,
        payload={
            "subject": "run-1",
            "observed_at": "2026-11-25T00:01:00+00:00",
            "state": "stopped",
        },
    )
    recorder.attach_observation(
        effect_seq=outcome.seq,
        source=OBSERVATION_SOURCE_POLL_LOOP,
        source_ref=None,
        payload={"subject": "p9", "observed_at": "2026-11-25T00:01:00+00:00", "state": "killed"},
    )
    with pytest.raises(ValueError):
        recorder.attach_observation(
            effect_seq=outcome.seq,
            source=OBSERVATION_SOURCE_POLL_LOOP,
            source_ref=None,
            payload={
                "subject": "p-other",
                "observed_at": "2026-11-25T00:01:00+00:00",
                "state": "running",
            },
        )


def test_observation_binder_source_is_a_tag_only_seam(recorder: BridgeEffectRecorder) -> None:
    """C-V1: the generic mechanism accepts the binder source tag,
    but there is NO binder field mapping in T08 — the mapping
    arrives with the E05-T02 binder. The test pins the seam: the
    tag attaches through the generic path, and no helper converts
    a binder outcome into an observation."""
    seq = _seed_effect(recorder)
    observation_id = recorder.attach_observation(
        effect_seq=seq,
        source=OBSERVATION_SOURCE_BINDER,
        source_ref="binder:rev-abc",
        payload={
            "subject": "a.txt",
            "observed_at": "2026-11-25T00:01:00+00:00",
            "state": "exists",
        },
    )
    assert observation_id == 1
    # No binder-mapping helper exists on the recorder.
    assert not hasattr(recorder, "attach_binder_observation")
    assert not hasattr(recorder, "binder_to_observation")


def test_observation_payload_must_be_metadata_only(recorder: BridgeEffectRecorder) -> None:
    seq = _seed_effect(recorder)
    with pytest.raises(ValueError):
        recorder.attach_observation(
            effect_seq=seq,
            source=OBSERVATION_SOURCE_MANUAL,
            source_ref=None,
            payload={
                "subject": "a.txt",
                "observed_at": "2026-11-25T00:01:00+00:00",
                "state": "exists",
                "content": "file bytes must never be an observation field",
            },
        )
    with pytest.raises(ValueError):
        recorder.attach_observation(
            effect_seq=seq,
            source=OBSERVATION_SOURCE_MANUAL,
            source_ref=None,
            payload={
                "subject": "a.txt",
                "observed_at": "2026-11-25T00:01:00+00:00",
                "state": "exists",
                "detail": "x" * 1025,
            },
        )


# --- audit gap (owner condition 4) ----------------------------------------------


def test_persist_failure_becomes_audit_gap_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BridgeEffectStore(tmp_path / "effects.db")
    recorder = BridgeEffectRecorder(store)

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected store failure")

    monkeypatch.setattr(store, "record_call", _boom)
    outcome = recorder.record_success(
        capability="host.fs.write",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt"},
        result=_write_result(effect=_effect(seq=1)),
    )
    assert outcome.outcome == "effect"
    assert outcome.seq is None
    assert outcome.audit_gap is True


def test_error_retry_lookup_failure_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex round 1, finding 3: when the journal cannot be read
    (locked, schema unavailable), the retry lookup must not raise
    out of record_error and replace the typed bridge error. The
    history is unverifiable, so the call is classified as a retry
    (fail-closed: process_not_found is unknown, not no_effect),
    and the row is still persisted through _persist."""
    store = BridgeEffectStore(tmp_path / "effects.db")
    recorder = BridgeEffectRecorder(store)

    def _boom(request_id: str) -> bool:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "has_request_id", _boom)
    outcome = recorder.record_error(
        capability="host.process.kill",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"process_id": "p1"},
        code="process_not_found",
        message="process_not_found happened",
    )
    assert outcome.outcome == "unknown"
    assert outcome.seq is not None
    assert outcome.audit_gap is False
    record = store.get_record(int(outcome.seq or 0))
    assert record is not None
    assert record["unknown_reason"] == "process_not_found_on_retry"


def test_error_retry_lookup_and_persist_failure_keeps_bridge_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex round 1, finding 3, journal fully down: both the
    retry lookup and the persist fail. record_error still returns
    a fail-closed outcome with audit_gap=True instead of raising
    — the caller (BridgeClient) then raises RecordedBridgeError
    with the ORIGINAL bridge code, never the SQLite error."""
    store = BridgeEffectStore(tmp_path / "effects.db")
    recorder = BridgeEffectRecorder(store)

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected store failure")

    monkeypatch.setattr(store, "has_request_id", _boom)
    monkeypatch.setattr(store, "record_call", _boom)
    outcome = recorder.record_error(
        capability="host.process.kill",
        request_id="req-1",
        principal_id=_OWNER,
        run_id=None,
        params={"process_id": "p1"},
        code="process_not_found",
        message="process_not_found happened",
    )
    assert outcome.outcome == "unknown"
    assert outcome.seq is None
    assert outcome.audit_gap is True


# --- unknown resolution binding / long-path params (Codex round 4) -------------


def test_cross_principal_effect_does_not_resolve_unknown(recorder: BridgeEffectRecorder) -> None:
    """Codex round 4, finding 1, end to end: Alice times out on
    host.fs.write/run-a; Bob's confirmed host.process.stop/run-b
    reuses the same request_id. Bob's effect must not resolve
    Alice's unknown — it proves nothing about her operation, and
    her row must stay visible under outcome=unknown."""
    unknown = recorder.record_transport_failure(
        capability="host.fs.write",
        request_id="shared-id",
        principal_id="alice",
        run_id="run-a",
        params={"root_id": "work", "path": "a.txt"},
        reason="timeout",
    )
    assert unknown.outcome == "unknown"
    effect = recorder.record_success(
        capability="host.process.stop",
        request_id="shared-id",
        principal_id="bob",
        run_id="run-b",
        params={"run_id": "run-b"},
        result={
            "run_id": "run-b",
            "processes": [],
            "mark_set": True,
            "effect": _effect(
                seq=9, kind="process.stop", principal_id="bob", run_id="run-b", processes=[]
            ),
        },
    )
    assert effect.outcome == "effect"
    assert effect.resolved_unknown_seqs == []
    attempt = recorder._store.get_record(int(unknown.seq or 0))
    assert attempt is not None
    assert attempt["outcome"] == "unknown"
    assert attempt["resolved_by_seq"] is None
    assert attempt["resolved_at"] is None


def test_long_path_survives_sanitize_as_preview(recorder: BridgeEffectRecorder) -> None:
    """Codex round 4, finding 4: a path longer than 256 bytes is
    turned into a preview dict by sanitize_record, and the
    projection filter then dropped the field entirely — on a
    timeout path (no effect block) the journal lost the only
    pointer to the file whose state must be checked before a
    retry. The field must survive as a bounded string preview."""
    long_path = "/".join(["nested"] * 40 + ["file.txt"])
    assert len(long_path.encode("utf-8")) > 256
    outcome = recorder.record_transport_failure(
        capability="host.fs.write",
        request_id="req-long-path",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": long_path, "commit": True},
        reason="timeout",
    )
    assert outcome.outcome == "unknown"
    record = recorder._store.get_record(int(outcome.seq or 0))
    assert record is not None
    params = record["params"]
    assert isinstance(params, dict)
    assert "path" in params  # red before the fix: the field is dropped
    kept = params["path"]
    assert isinstance(kept, str)
    assert kept.startswith("nested/nested/nested")
    assert kept.endswith("…[truncated]")


def test_long_dest_path_survives_sanitize_as_preview(recorder: BridgeEffectRecorder) -> None:
    """Same shape as finding 4 for host.fs.move: a long dest_path
    must not vanish from the journal on a timeout either."""
    long_dest = "/".join(["nested"] * 40 + ["moved.txt"])
    assert len(long_dest.encode("utf-8")) > 256
    outcome = recorder.record_transport_failure(
        capability="host.fs.move",
        request_id="req-long-dest",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "a.txt", "dest_path": long_dest},
        reason="timeout",
    )
    assert outcome.outcome == "unknown"
    record = recorder._store.get_record(int(outcome.seq or 0))
    assert record is not None
    params = record["params"]
    assert isinstance(params, dict)
    assert "dest_path" in params
    kept = params["dest_path"]
    assert isinstance(kept, str)
    assert kept.startswith("nested/nested/nested")
    assert kept.endswith("…[truncated]")
