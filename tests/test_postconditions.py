"""Typed postconditions (E05-T01): model, checkers, readers, combine, gate.

Groups (proposal §3):
1  model (frozen, JSON round-trip, revision, STALE)
2  file_exists (real desktop reader)
3  file_content_matches (real desktop reader)
4  process_running (fake + real reader)
5  process_exit (fake bridge records)
6  service_state (fake systemd runner)
7  browser_state (model-level; the E02-T10 regression suite stays in
   test_browser_postconditions.py and must pass unmodified)
8  custom_command (fake canonical executor)
9  combination (typed + canonical + browser channels)
10 gate (DONE only on a confirmed postcondition)
12 anti-fake (verified=True proves nothing; forged before fails)

Group 11 (plan path) lives in test_plan_execute.py next to the
negative test it complements.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path

import psutil
import pytest

from core.desktop_security import DesktopPathSecurity
from core.mwv.models import VerificationResult, VerificationStatus
from core.mwv.verifier_runtime import CANONICAL_CHECK_COMMAND, VerifierRuntime
from core.postconditions import (
    BrowserStateParams,
    CheckOutcome,
    CustomCommandParams,
    FileContentMatchesParams,
    FileExistsParams,
    FileInfo,
    Postcondition,
    PostconditionKind,
    ProcessExitParams,
    ProcessRunningParams,
    ServiceStateParams,
    browser_expectation_digest,
    check_name,
    evaluate_postcondition,
)
from core.run_outcome import (
    VerificationOutcome,
    combine_verification_results,
    map_evaluation_outcome,
    outcome_from_loop_result,
    verification_from_evaluations,
)
from core.tool_loop import AgentToolLoopResult
from shared.models import JSONValue, ToolResult
from tools.postcondition_readers import BridgePostconditionReader, DesktopPostconditionReader


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop_approvals.json",
    )


def _bound(kind: PostconditionKind, params: object) -> Postcondition:
    return Postcondition(kind=kind, params=params, source="test").with_revision()  # type: ignore[arg-type]


def _result(status: VerificationStatus) -> VerificationResult:
    return VerificationResult(
        status=status,
        command=["make", "check"],
        exit_code=0 if status is VerificationStatus.PASSED else 1,
        stdout="",
        stderr="",
        duration_seconds=0.0,
    )


class FakeReader:
    """Recording reader with canned answers (never touches the host)."""

    def __init__(
        self,
        *,
        files: dict[str, FileInfo] | None = None,
        contents: dict[str, bytes | None] | None = None,
        processes: dict[str, dict[str, JSONValue] | None] | None = None,
        os_processes: dict[int, tuple[bool, float | None]] | None = None,
        services: dict[tuple[str, str], dict[str, JSONValue] | None] | None = None,
        canonical: VerificationResult | None = None,
        read_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self.files = files or {}
        self.contents = contents or {}
        self.processes = processes or {}
        self.os_processes = os_processes or {}
        self.services = services or {}
        self._canonical = canonical
        self.read_errors = read_errors or {}

    def describe_file(self, path: str) -> FileInfo:
        self.calls.append(("describe_file", path))
        return self.files.get(path, FileInfo(exists=False, is_regular_file=False))

    def read_file(self, path: str) -> bytes | None:
        self.calls.append(("read_file", path))
        error = self.read_errors.get(path)
        if error is not None:
            raise error
        return self.contents.get(path)

    def process_status(self, process_id: str) -> dict[str, JSONValue] | None:
        self.calls.append(("process_status", process_id))
        return self.processes.get(process_id)

    def os_process_running(self, pid: int) -> tuple[bool, float | None]:
        self.calls.append(("os_process_running", pid))
        return self.os_processes.get(pid, (False, None))

    def service_status(self, unit: str, scope: str) -> dict[str, JSONValue] | None:
        self.calls.append(("service_status", (unit, scope)))
        return self.services.get((unit, scope))

    def canonical_check(self) -> VerificationResult:
        self.calls.append(("canonical_check", None))
        if self._canonical is None:
            return _result(VerificationStatus.ERROR)
        return self._canonical


# ---------------------------------------------------------------------------
# 1. Model.
# ---------------------------------------------------------------------------


def test_postcondition_is_frozen() -> None:
    pc = _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="/tmp/x.txt"))
    with pytest.raises(FrozenInstanceError):
        pc.kind = PostconditionKind.PROCESS_EXIT  # type: ignore[misc]


def test_postcondition_json_round_trip_all_kinds() -> None:
    cases = [
        PostconditionKind.FILE_EXISTS,
        PostconditionKind.FILE_CONTENT_MATCHES,
        PostconditionKind.PROCESS_RUNNING,
        PostconditionKind.PROCESS_EXIT,
        PostconditionKind.SERVICE_STATE,
        PostconditionKind.BROWSER_STATE,
        PostconditionKind.CUSTOM_COMMAND,
    ]
    params: dict[PostconditionKind, object] = {
        PostconditionKind.FILE_EXISTS: FileExistsParams(path="/tmp/x.txt"),
        PostconditionKind.FILE_CONTENT_MATCHES: FileContentMatchesParams(
            path="/tmp/x.txt", expected="hi"
        ),
        PostconditionKind.PROCESS_RUNNING: ProcessRunningParams(pid=123, create_time=1000.0),
        PostconditionKind.PROCESS_EXIT: ProcessExitParams(
            process_id="proc-1", expected_exit_code=0
        ),
        PostconditionKind.SERVICE_STATE: ServiceStateParams(
            unit="foo.service", scope="user", expected_active_state="active"
        ),
        PostconditionKind.BROWSER_STATE: BrowserStateParams(
            entity_id="e1", kind="text", after="Done"
        ),
        PostconditionKind.CUSTOM_COMMAND: CustomCommandParams(argv=tuple(CANONICAL_CHECK_COMMAND)),
    }
    for kind in cases:
        pc = Postcondition(
            kind=kind,
            params=params[kind],
            source="test",
            task_id="t1",
            step_id="s1",  # type: ignore[arg-type]
        ).with_revision()
        assert Postcondition.from_json(pc.to_json()) == pc
        # A second, registry-form process_running check round-trips too.
    registry = Postcondition(
        kind=PostconditionKind.PROCESS_RUNNING,
        params=ProcessRunningParams(process_id="proc-9"),
        source="test",
    ).with_revision()
    assert Postcondition.from_json(registry.to_json()) == registry


def test_browser_state_empty_text_round_trips() -> None:
    # Round-1 finding: an empty text is a valid expectation
    # (appearance of text from nothing, or clearing to ""); from_json
    # must not reject it.
    for params in (
        BrowserStateParams(
            entity_id="label", kind="text", after="Saved", before="", has_before=True
        ),
        BrowserStateParams(entity_id="label", kind="text", after=""),
    ):
        pc = Postcondition(
            kind=PostconditionKind.BROWSER_STATE, params=params, source="test"
        ).with_revision()
        assert Postcondition.from_json(pc.to_json()) == pc


def test_revision_is_stable_across_key_order() -> None:
    first = Postcondition.from_json(
        {
            "kind": "file_exists",
            "params": {"path": "/tmp/x.txt"},
            "source": "test",
            "criterion_revision": "ignored-by-revision",
        }
    )
    second = Postcondition.from_json(
        {
            "criterion_revision": "ignored-by-revision",
            "source": "test",
            "params": {"path": "/tmp/x.txt"},
            "kind": "file_exists",
        }
    )
    assert first.revision() == second.revision()
    assert first.revision() != "ignored-by-revision"


def test_postcondition_from_json_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        Postcondition.from_json({"kind": "nope", "params": {}})
    with pytest.raises(ValueError):
        Postcondition.from_json({"kind": "file_exists", "params": {"path": "/x", "extra": 1}})
    with pytest.raises(ValueError):
        Postcondition.from_json({"kind": "file_exists", "params": {}})
    with pytest.raises(ValueError):
        # kind/params mismatch: file_exists with process_exit params.
        Postcondition.from_json(
            {
                "kind": "file_exists",
                "params": {"process_id": "p", "expected_exit_code": 0},
            }
        )
    with pytest.raises(ValueError):
        Postcondition.from_json({"kind": "file_exists"})
    with pytest.raises(ValueError):
        Postcondition(
            kind=PostconditionKind.FILE_EXISTS,
            params=ProcessExitParams(process_id="p", expected_exit_code=0),  # type: ignore[arg-type]
        )


def test_stale_revision_never_reads() -> None:
    reader = FakeReader(files={"/tmp/x.txt": FileInfo(exists=True, is_regular_file=True)})
    pc = Postcondition(
        kind=PostconditionKind.FILE_EXISTS,
        params=FileExistsParams(path="/tmp/x.txt"),
        source="test",
        criterion_revision="forged",
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.STALE
    assert reader.calls == []
    unbound = Postcondition(
        kind=PostconditionKind.FILE_EXISTS,
        params=FileExistsParams(path="/tmp/x.txt"),
        source="test",
    )
    assert evaluate_postcondition(unbound, reader, now=1.0).outcome is CheckOutcome.STALE
    assert reader.calls == []


def test_process_running_bare_pid_is_construction_error() -> None:
    with pytest.raises(ValueError):
        ProcessRunningParams(pid=123)
    with pytest.raises(ValueError):
        ProcessRunningParams(pid=123, process_id="proc-1")
    with pytest.raises(ValueError):
        ProcessRunningParams(pid=123, create_time=1.0, process_id="proc-1")


# ---------------------------------------------------------------------------
# 2. file_exists (real desktop reader).
# ---------------------------------------------------------------------------


def test_file_exists_regular_file_passes(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("done", encoding="utf-8")
    reader = DesktopPostconditionReader(_security(tmp_path))
    pc = _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path=str(target)))
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.PASSED
    assert evaluation.evidence["reason"] == "ok"


def test_file_exists_directory_and_missing_fail(tmp_path: Path) -> None:
    reader = DesktopPostconditionReader(_security(tmp_path))
    directory = _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path=str(tmp_path)))
    assert evaluate_postcondition(directory, reader, now=1.0).outcome is CheckOutcome.FAILED
    missing = _bound(
        PostconditionKind.FILE_EXISTS, FileExistsParams(path=str(tmp_path / "nope.txt"))
    )
    assert evaluate_postcondition(missing, reader, now=1.0).outcome is CheckOutcome.FAILED


def test_file_exists_symlink_inside_root_passes(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    reader = DesktopPostconditionReader(_security(tmp_path))
    pc = _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path=str(link)))
    assert evaluate_postcondition(pc, reader, now=1.0).outcome is CheckOutcome.PASSED


def test_file_exists_symlink_escaping_root_is_unknown(tmp_path: Path) -> None:
    escape = tmp_path / "escape"
    escape.symlink_to(Path("/proc/version"))
    reader = DesktopPostconditionReader(_security(tmp_path))
    pc = _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path=str(escape)))
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.INCONCLUSIVE
    assert evaluation.evidence["reason"] == "path_denied"


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
def test_file_content_unreadable_is_unknown(tmp_path: Path) -> None:
    # Existence does not require readability: file_exists passes on a
    # chmod-000 file (stat needs no read permission), but reading its
    # content is INCONCLUSIVE (read_error), not a silent pass/fail.
    target = tmp_path / "secret.txt"
    target.write_text("x", encoding="utf-8")
    target.chmod(0o000)
    try:
        reader = DesktopPostconditionReader(_security(tmp_path))
        exists = _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path=str(target)))
        assert evaluate_postcondition(exists, reader, now=1.0).outcome is CheckOutcome.PASSED
        content = _bound(
            PostconditionKind.FILE_CONTENT_MATCHES,
            FileContentMatchesParams(path=str(target), expected="x"),
        )
        evaluation = evaluate_postcondition(content, reader, now=1.0)
        assert evaluation.outcome is CheckOutcome.INCONCLUSIVE
        assert evaluation.evidence["reason"] == "read_error"
    finally:
        target.chmod(0o644)


# ---------------------------------------------------------------------------
# 3. file_content_matches (real desktop reader).
# ---------------------------------------------------------------------------


def test_file_content_substring_pass_and_mismatch_fail(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("hello world\n", encoding="utf-8")
    reader = DesktopPostconditionReader(_security(tmp_path))
    passing = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(target), expected="world"),
    )
    assert evaluate_postcondition(passing, reader, now=1.0).outcome is CheckOutcome.PASSED
    failing = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(target), expected="goodbye"),
    )
    evaluation = evaluate_postcondition(failing, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "content_mismatch"


def test_file_content_exact_match_is_byte_exact(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("exact\n", encoding="utf-8")
    reader = DesktopPostconditionReader(_security(tmp_path))
    exact = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(target), expected="exact\n", match="exact"),
    )
    assert evaluate_postcondition(exact, reader, now=1.0).outcome is CheckOutcome.PASSED
    trailing = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(target), expected="exact", match="exact"),
    )
    assert evaluate_postcondition(trailing, reader, now=1.0).outcome is CheckOutcome.FAILED


def test_file_content_substring_normalizes_crlf(tmp_path: Path) -> None:
    # Round-1 finding: the legacy file_contains read with
    # Path.read_text() (universal newlines), so CRLF text matched an
    # LF expectation. Substring keeps that normalization.
    target = tmp_path / "win.txt"
    target.write_bytes(b"first\r\nsecond\r\n")
    reader = DesktopPostconditionReader(_security(tmp_path))
    passing = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(target), expected="first\nsecond"),
    )
    assert evaluate_postcondition(passing, reader, now=1.0).outcome is CheckOutcome.PASSED


def test_file_content_exact_keeps_crlf(tmp_path: Path) -> None:
    # Exact match keeps byte-exact semantics: no normalization.
    target = tmp_path / "win.txt"
    target.write_bytes(b"first\r\nsecond\r\n")
    reader = DesktopPostconditionReader(_security(tmp_path))
    lf_exact = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(target), expected="first\nsecond", match="exact"),
    )
    assert evaluate_postcondition(lf_exact, reader, now=1.0).outcome is CheckOutcome.FAILED
    crlf_exact = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(target), expected="first\r\nsecond\r\n", match="exact"),
    )
    assert evaluate_postcondition(crlf_exact, reader, now=1.0).outcome is CheckOutcome.PASSED


def test_file_content_missing_is_unknown_not_fail(tmp_path: Path) -> None:
    reader = DesktopPostconditionReader(_security(tmp_path))
    pc = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(tmp_path / "nope.txt"), expected="x"),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.INCONCLUSIVE
    assert evaluation.evidence["reason"] == "file_missing"


def test_file_content_over_cap_is_unknown(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_bytes(b"x" * 20)
    reader = DesktopPostconditionReader(_security(tmp_path), max_read_bytes=10)
    pc = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(target), expected="x"),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.INCONCLUSIVE
    assert evaluation.evidence["reason"] == "file_too_large"


def test_file_content_empty_substring_is_construction_error() -> None:
    with pytest.raises(ValueError):
        FileContentMatchesParams(path="/tmp/x.txt", expected="")
    # Exact match with an empty expectation is meaningful (empty file).
    FileContentMatchesParams(path="/tmp/x.txt", expected="", match="exact")


# ---------------------------------------------------------------------------
# 4. process_running (fake + real reader).
# ---------------------------------------------------------------------------


def test_process_running_registry_record_running_passes() -> None:
    reader = FakeReader(processes={"proc-1": {"state": "running", "exit_code": None}})
    pc = _bound(PostconditionKind.PROCESS_RUNNING, ProcessRunningParams(process_id="proc-1"))
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.PASSED
    assert evaluation.checker == "bridge_poll"


def test_process_running_registry_final_fails() -> None:
    for state in ("exited", "killed", "timeout", "spawn_failed"):
        reader = FakeReader(processes={"proc-1": {"state": state, "exit_code": 0}})
        pc = _bound(PostconditionKind.PROCESS_RUNNING, ProcessRunningParams(process_id="proc-1"))
        evaluation = evaluate_postcondition(pc, reader, now=1.0)
        assert evaluation.outcome is CheckOutcome.FAILED
        assert evaluation.evidence["reason"] == "process_final"


def test_process_running_registry_transitional_is_unknown() -> None:
    for state in ("launching", "killing"):
        reader = FakeReader(processes={"proc-1": {"state": state, "exit_code": None}})
        pc = _bound(PostconditionKind.PROCESS_RUNNING, ProcessRunningParams(process_id="proc-1"))
        assert evaluate_postcondition(pc, reader, now=1.0).outcome is CheckOutcome.INCONCLUSIVE


def test_process_running_unknown_process_id_is_unknown() -> None:
    reader = FakeReader(processes={})
    pc = _bound(PostconditionKind.PROCESS_RUNNING, ProcessRunningParams(process_id="proc-missing"))
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.INCONCLUSIVE
    assert evaluation.evidence["reason"] == "no_record"


def test_process_running_pid_alive_matches_create_time(tmp_path: Path) -> None:
    reader = DesktopPostconditionReader(_security(tmp_path))
    pid = os.getpid()
    create_time = psutil.Process(pid).create_time()
    pc = _bound(
        PostconditionKind.PROCESS_RUNNING,
        ProcessRunningParams(pid=pid, create_time=create_time),
    )
    assert evaluate_postcondition(pc, reader, now=1.0).outcome is CheckOutcome.PASSED
    # Within the 10 ms tolerance a float round-trip still passes.
    close = _bound(
        PostconditionKind.PROCESS_RUNNING,
        ProcessRunningParams(pid=pid, create_time=create_time + 0.005),
    )
    assert evaluate_postcondition(close, reader, now=1.0).outcome is CheckOutcome.PASSED
    stale = _bound(
        PostconditionKind.PROCESS_RUNNING,
        ProcessRunningParams(pid=pid, create_time=create_time + 60.0),
    )
    evaluation = evaluate_postcondition(stale, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "stale_pid"


def test_process_running_pid_dead_fails() -> None:
    reader = FakeReader(os_processes={4242: (False, None)})
    pc = _bound(
        PostconditionKind.PROCESS_RUNNING,
        ProcessRunningParams(pid=4242, create_time=1000.0),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "pid_dead"


@pytest.mark.skipif(os.name != "posix", reason="fork-based zombie test needs POSIX")
def test_process_running_zombie_is_not_running(tmp_path: Path) -> None:
    reader = DesktopPostconditionReader(_security(tmp_path))
    pid = os.fork()
    if pid == 0:  # noqa: PLR2004
        os._exit(0)
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
        alive, _ = reader.os_process_running(pid)
        assert alive is False
    finally:
        os.waitpid(pid, 0)


# ---------------------------------------------------------------------------
# 5. process_exit (fake bridge records).
# ---------------------------------------------------------------------------


def test_process_exit_final_matching_code_passes() -> None:
    reader = FakeReader(processes={"proc-1": {"state": "exited", "exit_code": 0}})
    pc = _bound(
        PostconditionKind.PROCESS_EXIT,
        ProcessExitParams(process_id="proc-1", expected_exit_code=0),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.PASSED
    assert evaluation.checker == "bridge_poll"


def test_process_exit_wrong_code_fails() -> None:
    reader = FakeReader(processes={"proc-1": {"state": "exited", "exit_code": 3}})
    pc = _bound(
        PostconditionKind.PROCESS_EXIT,
        ProcessExitParams(process_id="proc-1", expected_exit_code=0),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "exit_code_mismatch"


def test_process_exit_running_and_missing_are_unknown() -> None:
    reader = FakeReader(processes={"proc-1": {"state": "running", "exit_code": None}})
    pc = _bound(
        PostconditionKind.PROCESS_EXIT,
        ProcessExitParams(process_id="proc-1", expected_exit_code=0),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.INCONCLUSIVE
    assert evaluation.evidence["reason"] == "not_final"
    missing = _bound(
        PostconditionKind.PROCESS_EXIT,
        ProcessExitParams(process_id="proc-nope", expected_exit_code=0),
    )
    assert evaluate_postcondition(missing, reader, now=1.0).outcome is CheckOutcome.INCONCLUSIVE


def test_process_exit_missing_expected_code_is_construction_error() -> None:
    with pytest.raises(TypeError):
        ProcessExitParams(process_id="proc-1")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 6. service_state (fake systemd runner).
# ---------------------------------------------------------------------------


def _systemd_output(*, active_state: str, load_state: str = "loaded") -> str:
    return (
        "Id=foo.service\n"
        f"LoadState={load_state}\n"
        f"ActiveState={active_state}\n"
        "SubState=running\n"
        "UnitFileState=enabled\n"
        "MainPID=123\n"
        "Result=success\n"
    )


def _systemd_runner(outputs: dict[str, str], *, failures: set[str] | None = None) -> Callable:
    def run(argv, *, timeout, env=None, input_text=None):
        del env, input_text
        unit = argv[-1]
        if failures and unit in failures:
            return subprocess.CompletedProcess(
                args=list(argv), returncode=1, stdout="", stderr="dbus boom"
            )
        return subprocess.CompletedProcess(
            args=list(argv),
            returncode=0,
            stdout=outputs.get(unit, "LoadState=not-found\n"),
            stderr="",
        )

    return run


def test_service_state_active_passes(tmp_path: Path) -> None:
    runner = _systemd_runner({"foo.service": _systemd_output(active_state="active")})
    reader = DesktopPostconditionReader(_security(tmp_path), systemd_runner=runner)
    pc = _bound(
        PostconditionKind.SERVICE_STATE,
        ServiceStateParams(unit="foo.service", scope="user", expected_active_state="active"),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.PASSED
    assert evaluation.checker == "systemd_status"


def test_service_state_mismatch_fails(tmp_path: Path) -> None:
    runner = _systemd_runner({"foo.service": _systemd_output(active_state="inactive")})
    reader = DesktopPostconditionReader(_security(tmp_path), systemd_runner=runner)
    pc = _bound(
        PostconditionKind.SERVICE_STATE,
        ServiceStateParams(unit="foo.service", scope="user", expected_active_state="active"),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "state_mismatch"


def test_service_state_not_found_fails(tmp_path: Path) -> None:
    runner = _systemd_runner({})
    reader = DesktopPostconditionReader(_security(tmp_path), systemd_runner=runner)
    pc = _bound(
        PostconditionKind.SERVICE_STATE,
        ServiceStateParams(unit="nope.service", scope="system", expected_active_state="active"),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "unit_not_found"


def test_service_state_reader_error_is_unknown(tmp_path: Path) -> None:
    runner = _systemd_runner({}, failures={"foo.service"})
    reader = DesktopPostconditionReader(_security(tmp_path), systemd_runner=runner)
    pc = _bound(
        PostconditionKind.SERVICE_STATE,
        ServiceStateParams(unit="foo.service", scope="user", expected_active_state="active"),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.INCONCLUSIVE
    assert evaluation.evidence["reason"] == "reader_error"


def test_service_state_systemd_timeout_is_unknown(tmp_path: Path) -> None:
    # Round-1 finding: a hung systemctl raises TimeoutExpired from
    # the runner; the reader converts it to OSError, so the check is
    # INCONCLUSIVE instead of crashing verify_postconditions.
    def runner(argv, *, timeout, env=None, input_text=None):
        del argv, timeout, env, input_text
        raise subprocess.TimeoutExpired(["systemctl", "show"], 20)

    reader = DesktopPostconditionReader(_security(tmp_path), systemd_runner=runner)
    pc = _bound(
        PostconditionKind.SERVICE_STATE,
        ServiceStateParams(unit="foo.service", scope="user", expected_active_state="active"),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.INCONCLUSIVE
    assert evaluation.evidence["reason"] == "reader_error"


def test_service_state_missing_scope_is_construction_error() -> None:
    with pytest.raises(ValueError):
        ServiceStateParams(unit="foo.service", scope="host", expected_active_state="active")
    with pytest.raises(ValueError):
        ServiceStateParams(unit="foo.service", scope="user", expected_active_state="sleeping")


# ---------------------------------------------------------------------------
# 7. browser_state (model-level; E02-T10 regression suite unchanged).
# ---------------------------------------------------------------------------


def _text_facts(
    *,
    after_ok: bool = True,
    before_changed: bool = True,
    before_ok: bool = True,
    after: str = "Published",
    before: str | None = None,
    has_before: bool = False,
):
    # Facts carry the digest of the expectation they were computed
    # against, exactly as the tool stamps them (see
    # browser_expectation_digest): the checker requires it to equal
    # the postcondition's revision.
    return [
        {
            "entity_id": "publish-btn",
            "kind": "text",
            "after_matches_after": after_ok,
            "before_matches_after": not before_changed,
            "before_matches_expected_before": before_ok,
            "expectation_sha256": browser_expectation_digest(
                BrowserStateParams(
                    entity_id="publish-btn",
                    kind="text",
                    after=after,
                    before=before,
                    has_before=has_before,
                ).to_json()
            ),
        }
    ]


def test_browser_state_text_confirms_proven_change() -> None:
    params = BrowserStateParams(entity_id="publish-btn", kind="text", after="Published")
    evaluation = evaluate_postcondition(
        _bound(PostconditionKind.BROWSER_STATE, params), None, browser_facts=_text_facts(), now=1.0
    )
    assert evaluation.outcome is CheckOutcome.PASSED


def test_browser_state_no_facts_fails() -> None:
    params = BrowserStateParams(entity_id="publish-btn", kind="text", after="Published")
    evaluation = evaluate_postcondition(
        _bound(PostconditionKind.BROWSER_STATE, params), None, browser_facts=[], now=1.0
    )
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "browser_postcondition_unconfirmed"


def test_browser_state_changed_after_capture_is_stale() -> None:
    original = BrowserStateParams(entity_id="publish-btn", kind="text", after="Published")
    bound = Postcondition(
        kind=PostconditionKind.BROWSER_STATE, params=original, source="test"
    ).with_revision()
    tampered = Postcondition(
        kind=PostconditionKind.BROWSER_STATE,
        params=BrowserStateParams(entity_id="publish-btn", kind="text", after="Draft"),
        source="test",
        criterion_revision=bound.criterion_revision,
    )
    evaluation = evaluate_postcondition(tampered, None, browser_facts=_text_facts(), now=1.0)
    assert evaluation.outcome is CheckOutcome.STALE


def test_browser_state_facts_for_other_expectation_do_not_confirm() -> None:
    # Round-1 finding: facts captured for after="Saved" must not
    # confirm a postcondition declaring after="Deleted", even though
    # every comparison boolean is favourable.
    saved = _text_facts(after="Saved")
    params = BrowserStateParams(entity_id="publish-btn", kind="text", after="Deleted")
    evaluation = evaluate_postcondition(
        _bound(PostconditionKind.BROWSER_STATE, params), None, browser_facts=saved, now=1.0
    )
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "browser_postcondition_unconfirmed"


def test_browser_state_facts_without_binding_do_not_confirm() -> None:
    # Facts without the expectation digest fail closed: unbound facts
    # prove nothing.
    facts = _text_facts()
    del facts[0]["expectation_sha256"]
    params = BrowserStateParams(entity_id="publish-btn", kind="text", after="Published")
    evaluation = evaluate_postcondition(
        _bound(PostconditionKind.BROWSER_STATE, params), None, browser_facts=facts, now=1.0
    )
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "browser_postcondition_unconfirmed"


def test_browser_state_facts_for_other_before_do_not_confirm() -> None:
    # The digest covers before too: facts bound to before="Pending"
    # must not confirm a postcondition declaring before="Draft", even
    # when the before boolean claims a match.
    facts = _text_facts(after="Published", before="Pending", has_before=True)
    params = BrowserStateParams(
        entity_id="publish-btn", kind="text", after="Published", before="Draft", has_before=True
    )
    evaluation = evaluate_postcondition(
        _bound(PostconditionKind.BROWSER_STATE, params), None, browser_facts=facts, now=1.0
    )
    assert evaluation.outcome is CheckOutcome.FAILED
    assert evaluation.evidence["reason"] == "browser_postcondition_unconfirmed"


def test_browser_state_count_requires_exactly_one() -> None:
    params = BrowserStateParams(entity_id="item", kind="count", after=1)
    facts = [{"entity_id": "item", "kind": "count", "before_count": 0, "after_count": 1}]
    evaluation = evaluate_postcondition(
        _bound(PostconditionKind.BROWSER_STATE, params), None, browser_facts=facts, now=1.0
    )
    assert evaluation.outcome is CheckOutcome.PASSED
    two = BrowserStateParams(entity_id="item", kind="count", after=2)
    assert (
        evaluate_postcondition(
            _bound(PostconditionKind.BROWSER_STATE, two), None, browser_facts=facts, now=1.0
        ).outcome
        is CheckOutcome.FAILED
    )


# ---------------------------------------------------------------------------
# 8. custom_command (fake canonical executor).
# ---------------------------------------------------------------------------


def test_custom_command_canonical_delegates_to_executor() -> None:
    for status, expected in (
        (VerificationStatus.PASSED, CheckOutcome.PASSED),
        (VerificationStatus.FAILED, CheckOutcome.FAILED),
        (VerificationStatus.ERROR, CheckOutcome.INCONCLUSIVE),
    ):
        reader = FakeReader(canonical=_result(status))
        pc = _bound(
            PostconditionKind.CUSTOM_COMMAND,
            CustomCommandParams(argv=tuple(CANONICAL_CHECK_COMMAND)),
        )
        evaluation = evaluate_postcondition(pc, reader, now=1.0)
        assert evaluation.outcome is expected
        assert ("canonical_check", None) in reader.calls


def test_custom_command_other_argv_never_spawns() -> None:
    calls: list = []

    def canonical() -> VerificationResult:
        calls.append("spawned")
        return _result(VerificationStatus.PASSED)

    reader = FakeReader(canonical=_result(VerificationStatus.PASSED))
    reader.canonical_check = canonical  # type: ignore[method-assign]
    pc = _bound(
        PostconditionKind.CUSTOM_COMMAND,
        CustomCommandParams(argv=("make", "test")),
    )
    evaluation = evaluate_postcondition(pc, reader, now=1.0)
    assert evaluation.outcome is CheckOutcome.INCONCLUSIVE
    assert evaluation.evidence["reason"] == "non_canonical_command"
    assert calls == []


def test_custom_command_empty_argv_is_construction_error() -> None:
    with pytest.raises(ValueError):
        CustomCommandParams(argv=())


# ---------------------------------------------------------------------------
# 9. Combination.
# ---------------------------------------------------------------------------


def _all_passing_checks(tmp_path: Path) -> list[Postcondition]:
    target = tmp_path / "result.txt"
    target.write_text("hello", encoding="utf-8")
    return [
        _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path=str(target))),
        _bound(
            PostconditionKind.FILE_CONTENT_MATCHES,
            FileContentMatchesParams(path=str(target), expected="hell"),
        ),
        _bound(PostconditionKind.PROCESS_RUNNING, ProcessRunningParams(process_id="proc-1")),
        _bound(
            PostconditionKind.PROCESS_EXIT,
            ProcessExitParams(process_id="proc-2", expected_exit_code=0),
        ),
        _bound(
            PostconditionKind.SERVICE_STATE,
            ServiceStateParams(unit="foo.service", scope="user", expected_active_state="active"),
        ),
        _bound(
            PostconditionKind.BROWSER_STATE,
            BrowserStateParams(entity_id="publish-btn", kind="text", after="Published"),
        ),
        _bound(
            PostconditionKind.CUSTOM_COMMAND,
            CustomCommandParams(argv=tuple(CANONICAL_CHECK_COMMAND)),
        ),
    ]


def _passing_reader(tmp_path: Path) -> FakeReader:
    return FakeReader(
        processes={
            "proc-1": {"state": "running", "exit_code": None},
            "proc-2": {"state": "exited", "exit_code": 0},
        },
        services={("foo.service", "user"): {"active_state": "active"}},
        canonical=_result(VerificationStatus.PASSED),
        files={
            str(tmp_path / "result.txt"): FileInfo(exists=True, is_regular_file=True),
        },
        contents={str(tmp_path / "result.txt"): b"hello"},
    )


def test_verify_postconditions_all_passed(tmp_path: Path) -> None:
    checks = _all_passing_checks(tmp_path)
    verification = VerifierRuntime().verify_postconditions(
        checks, _passing_reader(tmp_path), browser_facts=_text_facts()
    )
    assert verification is not None
    assert verification.outcome is VerificationOutcome.PASSED
    assert len(verification.checks) == 7
    assert {check.name.split(":")[0] for check in verification.checks} == {
        kind.value for kind in PostconditionKind
    }
    assert all(check.passed for check in verification.checks)


def test_verify_postconditions_one_failed_is_failed(tmp_path: Path) -> None:
    checks = _all_passing_checks(tmp_path)
    checks[0] = _bound(
        PostconditionKind.FILE_EXISTS, FileExistsParams(path=str(tmp_path / "missing.txt"))
    )
    verification = VerifierRuntime().verify_postconditions(
        checks, _passing_reader(tmp_path), browser_facts=_text_facts()
    )
    assert verification is not None
    assert verification.outcome is VerificationOutcome.FAILED


def test_verify_postconditions_one_unknown_is_unknown(tmp_path: Path) -> None:
    checks = _all_passing_checks(tmp_path)
    checks[1] = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(tmp_path / "missing.txt"), expected="x"),
    )
    verification = VerifierRuntime().verify_postconditions(
        checks, _passing_reader(tmp_path), browser_facts=_text_facts()
    )
    assert verification is not None
    assert verification.outcome is VerificationOutcome.UNKNOWN


def test_verify_postconditions_empty_is_none() -> None:
    assert VerifierRuntime().verify_postconditions([], FakeReader()) is None


def test_combine_typed_failed_with_browser_passed_is_failed() -> None:
    typed = verification_from_evaluations(
        [
            evaluate_postcondition(
                _bound(
                    PostconditionKind.FILE_EXISTS,
                    FileExistsParams(path="/tmp/missing.txt"),
                ),
                FakeReader(),
                now=1.0,
            )
        ]
    )
    combined = combine_verification_results(None, _result(VerificationStatus.PASSED), typed=typed)
    assert combined is not None
    assert combined.outcome is VerificationOutcome.FAILED


def test_combine_typed_passed_with_browser_unconfirmed_is_failed() -> None:
    typed = verification_from_evaluations(
        [
            evaluate_postcondition(
                _bound(
                    PostconditionKind.FILE_EXISTS,
                    FileExistsParams(path="/tmp/x.txt"),
                ),
                FakeReader(files={"/tmp/x.txt": FileInfo(exists=True, is_regular_file=True)}),
                now=1.0,
            )
        ]
    )
    combined = combine_verification_results(None, _result(VerificationStatus.FAILED), typed=typed)
    assert combined is not None
    assert combined.outcome is VerificationOutcome.FAILED


def test_check_name_format() -> None:
    pc = _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="/workspace/result.txt"))
    assert check_name(pc) == "file_exists:/workspace/result.txt"
    assert map_evaluation_outcome(CheckOutcome.STALE) is VerificationOutcome.UNKNOWN
    assert map_evaluation_outcome(CheckOutcome.BLOCKED) is VerificationOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# 10. Gate.
# ---------------------------------------------------------------------------


def _clean_loop() -> AgentToolLoopResult:
    return AgentToolLoopResult(text="done", messages=[], tool_calls=[])


def test_gate_state_changing_with_passed_checks_is_full_done(tmp_path: Path) -> None:
    verification = VerifierRuntime().verify_postconditions(
        _all_passing_checks(tmp_path), _passing_reader(tmp_path), browser_facts=_text_facts()
    )
    outcome = outcome_from_loop_result(
        _clean_loop(), state_changing=True, verification=verification
    )
    assert outcome.is_full_done is True


def test_gate_one_unknown_check_is_unverified(tmp_path: Path) -> None:
    checks = _all_passing_checks(tmp_path)
    checks[1] = _bound(
        PostconditionKind.FILE_CONTENT_MATCHES,
        FileContentMatchesParams(path=str(tmp_path / "missing.txt"), expected="x"),
    )
    verification = VerifierRuntime().verify_postconditions(
        checks, _passing_reader(tmp_path), browser_facts=_text_facts()
    )
    outcome = outcome_from_loop_result(
        _clean_loop(), state_changing=True, verification=verification
    )
    assert outcome.is_full_done is False
    assert outcome.verification.outcome is VerificationOutcome.UNKNOWN


def test_gate_tool_ok_alone_is_unverified() -> None:
    outcome = outcome_from_loop_result(_clean_loop(), state_changing=True, verification=None)
    assert outcome.is_full_done is False
    assert outcome.verification.outcome is VerificationOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# 12. Anti-fake.
# ---------------------------------------------------------------------------


def test_verified_true_tool_result_is_ignored_by_typed_channel() -> None:
    # A ToolResult claiming verified=True carries no witness kind: the
    # typed channel never looks at it, and the browser_state check
    # still fails for absence of facts.
    result = ToolResult.success({"verified": True})
    assert result.data.get("verified") is True
    from core.postconditions import collect_browser_state_facts

    assert collect_browser_state_facts([result]) == []
    pc = _bound(
        PostconditionKind.BROWSER_STATE,
        BrowserStateParams(entity_id="publish-btn", kind="text", after="Published"),
    )
    evaluation = evaluate_postcondition(pc, None, browser_facts=[], now=1.0)
    assert evaluation.outcome is CheckOutcome.FAILED


def test_forged_before_fails() -> None:
    params = BrowserStateParams(
        entity_id="publish-btn", kind="text", after="Published", before="Draft", has_before=True
    )
    forged = _text_facts(before_ok=False, before="Draft", has_before=True)
    evaluation = evaluate_postcondition(
        _bound(PostconditionKind.BROWSER_STATE, params), None, browser_facts=forged, now=1.0
    )
    assert evaluation.outcome is CheckOutcome.FAILED


def test_no_proven_change_fails() -> None:
    params = BrowserStateParams(entity_id="publish-btn", kind="text", after="Published")
    no_change = _text_facts(before_changed=False)
    evaluation = evaluate_postcondition(
        _bound(PostconditionKind.BROWSER_STATE, params), None, browser_facts=no_change, now=1.0
    )
    assert evaluation.outcome is CheckOutcome.FAILED


def test_bridge_reader_delegates_files_to_desktop(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("hello", encoding="utf-8")
    desktop = DesktopPostconditionReader(_security(tmp_path))
    bridge = BridgePostconditionReader(
        lambda process_id: (
            {"state": "running", "exit_code": None} if process_id == "proc-1" else None
        ),
        desktop,
    )
    pc = _bound(PostconditionKind.PROCESS_RUNNING, ProcessRunningParams(process_id="proc-1"))
    assert evaluate_postcondition(pc, bridge, now=1.0).outcome is CheckOutcome.PASSED
    file_pc = _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path=str(target)))
    assert evaluate_postcondition(file_pc, bridge, now=1.0).outcome is CheckOutcome.PASSED
    assert bridge.process_status("proc-missing") is None
