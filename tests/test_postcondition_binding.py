"""E05-T02: effect-to-postcondition binding (schema §1–§3).

The binder derives typed checks in trusted code from
``(tool name, args, result)`` — never from model text, never from a
tool's ``verified`` flag. Covered here:

* §1.3 — the derivation table, one test per row (plus its edges:
  dry_run, the read cap, empty content, dead launch, missing scope);
* §2.2 — precedence: expected wins per subject, the superseded
  suggestion is recorded, never silently dropped;
* §2.4 — the step hook: bind + evaluate through the typed checker,
  ``None`` when the step has no checks;
* Q12 — hash stability for plans/packets without the new field;
* §2.1/§2.3 — normalize / compile / serialize / deserialize
  plumbing for ``expected_postconditions``.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from core.postcondition_binding import (
    bind_run_postconditions,
    bind_step_postconditions,
    make_step_verification_hook,
    suggest_postconditions,
)
from core.postconditions import (
    BrowserStateParams,
    FileAbsentParams,
    FileContentMatchesParams,
    FileExistsParams,
    FileInfo,
    Postcondition,
    PostconditionKind,
    ProcessRunningParams,
    ServiceStateParams,
    check_name,
)
from core.run_outcome import VerificationOutcome
from core.tool_loop import AgentToolLoopResult, ExecutedToolCall
from llm.types import ToolCall
from server.http.common.workflow_runtime import (
    compile_plan_to_task_packet,
    deserialize_task_packet_payload,
    serialize_task_packet_payload,
)
from server.http.common.workflow_state import normalize_plan_step, plan_hash_payload
from shared.models import ToolResult
from tests.test_postconditions import FakeReader
from tools.desktop_tools import MAX_DESKTOP_READ_BYTES

# Q12 — pinned on the pre-T02 head (script
# ~/workspace/slavikai-e05-t02/pin_hashes.py). A plan WITHOUT
# expected_postconditions must hash exactly as before the field
# existed; including the key with None/[] would rewrite every
# already-approved plan's hash. NOTE: only the PLAN hash pins to an
# absolute value — the packet hash cannot (compile generates a random
# task_id per call, and the packet hash covers it), so packet
# stability is proven structurally instead: the hashed payload of a
# packet without the field is byte-identical to the pre-T02 inline
# shape (see test_packet_hash_payload_without_expected_matches_pre_t02_shape).
PLAN_HASH_NO_EXPECTED = "6a863f888b49399db81e7c69809f335a5c6e78780102fa18401e7db9f515202a"


def _capability(name: str) -> str | None:
    """Fake registry: every table tool is write/exec, one is read."""
    return {
        "workspace_write": "write",
        "workspace_create": "write",
        "workspace_patch": "write",
        "workspace_delete": "write",
        "workspace_rename": "write",
        "workspace_move": "write",
        "desktop_file_write": "write",
        "desktop_file_transfer": "write",
        "desktop_file_delete": "write",
        "desktop_process": "exec",
        "desktop_systemd": "exec",
        "desktop_browser": "write",
        "desktop_shell": "exec",
        "workspace_run": "exec",
        "desktop_file_read": "read",
    }.get(name)


def _call(
    name: str,
    args: dict,
    data: dict | None = None,
    *,
    ok: bool = True,
    meta: dict | None = None,
) -> ExecutedToolCall:
    return ExecutedToolCall(
        call=ToolCall(id=f"c-{name}", name=name, arguments=args),
        result=ToolResult(ok=ok, data=data or {}, meta=meta),
    )


def _loop(*calls: ExecutedToolCall) -> AgentToolLoopResult:
    return AgentToolLoopResult(text="done", messages=[], tool_calls=list(calls))


def _reader() -> FakeReader:
    return FakeReader(
        files={
            "/tmp/a.txt": FileInfo(exists=True, is_regular_file=True),
            "/tmp/dir": FileInfo(exists=True, is_regular_file=False),
        },
        contents={"/tmp/a.txt": b"written"},
        os_processes={4242: (True, 1710000000.0)},
        services={("app.service", "user"): {"active_state": "active"}},
    )


def _bound(kind: PostconditionKind, params: object) -> Postcondition:
    return Postcondition(kind=kind, params=params, source="test").with_revision()  # type: ignore[arg-type]


def _kinds(checks: list[Postcondition]) -> list[str]:
    return [pc.kind.value for pc in checks]


# ---------------------------------------------------------------------------
# §1.3 — derivation table.
# ---------------------------------------------------------------------------


def test_suggest_workspace_write() -> None:
    checks = suggest_postconditions(
        tool_name="workspace_write",
        args={"path": "/tmp/a.txt", "content": "written"},
        result=ToolResult.success(
            {"output": "ok", "path": "/tmp/a.txt"},
        ),
        get_capability=_capability,
        reader=_reader(),
    )
    assert _kinds(checks) == ["file_exists", "file_content_matches"]
    assert checks[0].params == FileExistsParams(path="/tmp/a.txt")
    assert checks[1].params == FileContentMatchesParams(
        path="/tmp/a.txt", expected="written", match="exact"
    )
    assert checks[0].source == "suggested:workspace_write"
    assert all(pc.revision() for pc in checks)


def test_suggest_workspace_write_over_cap_only_file_exists() -> None:
    # Content over MAX_DESKTOP_READ_BYTES cannot be verified exactly
    # (the checker would call it INCONCLUSIVE) — so no content check
    # is suggested at all (Q13).
    big = "x" * (MAX_DESKTOP_READ_BYTES + 1)
    checks = suggest_postconditions(
        tool_name="workspace_write",
        args={"path": "/tmp/a.txt", "content": big},
        result=ToolResult.success(
            {"output": "ok", "path": "/tmp/a.txt"},
        ),
        get_capability=_capability,
        reader=_reader(),
    )
    assert _kinds(checks) == ["file_exists"]


def test_suggest_workspace_write_empty_content() -> None:
    # Empty content is a real bar with match="exact" (an empty file);
    # substring would be vacuous and is rejected by the params.
    checks = suggest_postconditions(
        tool_name="workspace_write",
        args={"path": "/tmp/a.txt", "content": ""},
        result=ToolResult.success(
            {"output": "ok", "path": "/tmp/a.txt"},
        ),
        get_capability=_capability,
        reader=_reader(),
    )
    assert _kinds(checks) == ["file_exists", "file_content_matches"]
    assert checks[1].params == FileContentMatchesParams(
        path="/tmp/a.txt", expected="", match="exact"
    )


def test_suggest_workspace_patch_and_dry_run() -> None:
    checks = suggest_postconditions(
        tool_name="workspace_patch",
        args={"path": "/tmp/a.txt", "patch": "@@ -1 +1 @@\n-x\n+y\n", "dry_run": False},
        result=ToolResult.success(
            {"output": "ok", "path": "/tmp/a.txt", "content": "written", "dry_run": False},
        ),
        get_capability=_capability,
        reader=_reader(),
    )
    assert _kinds(checks) == ["file_exists", "file_content_matches"]
    assert checks[1].params == FileContentMatchesParams(
        path="/tmp/a.txt", expected="written", match="exact"
    )
    # A dry run changes nothing on disk: no check at all.
    dry = suggest_postconditions(
        tool_name="workspace_patch",
        args={"path": "/tmp/a.txt", "patch": "@@ -1 +1 @@\n-x\n+y\n", "dry_run": True},
        result=ToolResult.success(
            {"output": "ok", "path": "/tmp/a.txt", "content": "written", "dry_run": True},
        ),
        get_capability=_capability,
        reader=_reader(),
    )
    assert dry == []


def test_suggest_workspace_rename_and_move() -> None:
    rename = suggest_postconditions(
        tool_name="workspace_rename",
        args={"old_path": "/tmp/old.txt", "new_path": "/tmp/a.txt"},
        result=ToolResult.success(
            {"output": "ok", "old_path": "/tmp/old.txt", "new_path": "/tmp/a.txt"},
        ),
        get_capability=_capability,
        reader=_reader(),
    )
    assert _kinds(rename) == ["file_exists", "file_absent"]
    assert rename[0].params == FileExistsParams(path="/tmp/a.txt")
    assert rename[1].params == FileAbsentParams(path="/tmp/old.txt")
    move = suggest_postconditions(
        tool_name="workspace_move",
        args={"from_path": "/tmp/old.txt", "to_path": "/tmp/a.txt"},
        result=ToolResult.success(
            {"output": "ok", "from_path": "/tmp/old.txt", "to_path": "/tmp/a.txt"},
        ),
        get_capability=_capability,
        reader=_reader(),
    )
    assert _kinds(move) == ["file_exists", "file_absent"]
    assert move[1].params == FileAbsentParams(path="/tmp/old.txt")


def test_suggest_transfer_to_directory_keeps_only_absent() -> None:
    # file_exists is regular-file-only: suggesting it for a directory
    # destination would FAIL a good move. The binder stats the dest
    # first and suppresses the check it knows would fail (Q13).
    reader = _reader()
    move = suggest_postconditions(
        tool_name="desktop_file_transfer",
        args={"operation": "move", "source": "/tmp/old.txt", "destination": "/tmp/dir"},
        result=ToolResult.success(
            {"output": "ok", "source": "/tmp/old.txt", "destination": "/tmp/dir"},
        ),
        get_capability=_capability,
        reader=reader,
    )
    assert _kinds(move) == ["file_absent"]
    copy = suggest_postconditions(
        tool_name="desktop_file_transfer",
        args={"operation": "copy", "source": "/tmp/old.txt", "destination": "/tmp/dir"},
        result=ToolResult.success(
            {"output": "ok", "source": "/tmp/old.txt", "destination": "/tmp/dir"},
        ),
        get_capability=_capability,
        reader=reader,
    )
    assert copy == []


def test_suggest_without_reader_suggests_nothing() -> None:
    # No reader can evaluate any suggestion (every suggested check
    # reads real state): suggesting one would only ever evaluate
    # BLOCKED and veto a run the caller never asked to verify. The
    # UI desktop path runs without a reader — a suggested file_absent
    # there turned a verified delete into UNVERIFIED (regression
    # caught by
    # tests/ui_api/test_desktop_mode.py::test_desktop_chat_approval_resumes_same_pipeline).
    # Expected conditions are unaffected: the caller asked for those
    # explicitly (T01 evaluates them BLOCKED with no reader).
    for name, args, data in (
        (
            "workspace_write",
            {"path": "/tmp/a.txt", "content": "written"},
            {"output": "ok", "path": "/tmp/a.txt"},
        ),
        (
            "desktop_file_transfer",
            {"operation": "move", "source": "/tmp/old.txt", "destination": "/tmp/a.txt"},
            {"output": "ok", "source": "/tmp/old.txt", "destination": "/tmp/a.txt"},
        ),
        (
            "workspace_delete",
            {"path": "/tmp/old.txt", "recursive": False},
            {"output": "ok", "path": "/tmp/old.txt"},
        ),
        (
            "desktop_process",
            {"operation": "launch"},
            {"pid": 4242, "create_time": 1710000000.0},
        ),
        (
            "desktop_systemd",
            {"operation": "start", "unit": "app.service", "scope": "user"},
            {"output": "ok"},
        ),
    ):
        assert (
            suggest_postconditions(
                tool_name=name,
                args=args,
                result=ToolResult.success(data),
                get_capability=_capability,
                reader=None,
            )
            == []
        )


def test_suggest_delete() -> None:
    checks = suggest_postconditions(
        tool_name="workspace_delete",
        args={"path": "/tmp/old.txt", "recursive": False},
        result=ToolResult.success(
            {"output": "ok", "path": "/tmp/old.txt"},
        ),
        get_capability=_capability,
        reader=_reader(),
    )
    assert _kinds(checks) == ["file_absent"]
    assert checks[0].params == FileAbsentParams(path="/tmp/old.txt")


def test_suggest_launch_alive_dead_bad_pid_no_reader() -> None:
    reader = _reader()
    alive = suggest_postconditions(
        tool_name="desktop_process",
        args={"operation": "launch"},
        result=ToolResult.success(
            {"pid": 4242, "create_time": 1710000000.0, "verified": True},
        ),
        get_capability=_capability,
        reader=reader,
    )
    # create_time is re-read from the live process at bind time — the
    # result's value is never trusted.
    assert _kinds(alive) == ["process_running"]
    assert alive[0].params == ProcessRunningParams(pid=4242, create_time=1710000000.0)
    dead_reader = FakeReader(os_processes={4242: (False, None)})
    assert (
        suggest_postconditions(
            tool_name="desktop_process",
            args={"operation": "launch"},
            result=ToolResult.success(
                {"pid": 4242, "create_time": 1710000000.0},
            ),
            get_capability=_capability,
            reader=dead_reader,
        )
        == []
    )
    assert (
        suggest_postconditions(
            tool_name="desktop_process",
            args={"operation": "launch"},
            result=ToolResult.success(
                {"pid": "not-a-pid", "create_time": 1.0},
            ),
            get_capability=_capability,
            reader=reader,
        )
        == []
    )
    assert (
        suggest_postconditions(
            tool_name="desktop_process",
            args={"operation": "launch"},
            result=ToolResult.success(
                {"pid": 4242, "create_time": 1710000000.0},
            ),
            get_capability=_capability,
            reader=None,
        )
        == []
    )


def test_suggest_systemd_start_stop_restart() -> None:
    for operation, expected_state in (
        ("start", "active"),
        ("restart", "active"),
        ("stop", "inactive"),
    ):
        checks = suggest_postconditions(
            tool_name="desktop_systemd",
            args={"operation": operation, "unit": "app.service", "scope": "user"},
            result=ToolResult.success(
                {"output": "ok"},
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        assert _kinds(checks) == ["service_state"]
        assert checks[0].params == ServiceStateParams(
            unit="app.service", scope="user", expected_active_state=expected_state
        )


def test_suggest_systemd_missing_scope_invalid_unit_enable() -> None:
    # Scope is required explicitly — the tool's default ("system") is
    # never assumed: suggesting a check for the wrong scope would
    # verify the wrong unit.
    assert (
        suggest_postconditions(
            tool_name="desktop_systemd",
            args={"operation": "start", "unit": "app.service"},
            result=ToolResult.success(
                {"output": "ok"},
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        == []
    )
    assert (
        suggest_postconditions(
            tool_name="desktop_systemd",
            args={"operation": "start", "unit": "app service!", "scope": "user"},
            result=ToolResult.success(
                {"output": "ok"},
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        == []
    )
    assert (
        suggest_postconditions(
            tool_name="desktop_systemd",
            args={"operation": "enable", "unit": "app.service", "scope": "user"},
            result=ToolResult.success(
                {"output": "ok"},
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        == []
    )
    assert (
        suggest_postconditions(
            tool_name="desktop_systemd",
            args={"operation": "disable", "unit": "app.service", "scope": "user"},
            result=ToolResult.success(
                {"output": "ok"},
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        == []
    )


def test_suggest_no_suggestion_tools() -> None:
    # Q5/Q6: shell, workspace_run, extract and GUI tools have no
    # suggestion in v1 — known gaps, recorded, not silent passes.
    for name, args in (
        ("desktop_shell", {"command": "ls"}),
        ("workspace_run", {"command": "ls"}),
    ):
        assert (
            suggest_postconditions(
                tool_name=name,
                args=args,
                result=ToolResult.success(
                    {"output": "ok"},
                ),
                get_capability=_capability,
                reader=_reader(),
            )
            == []
        )


def test_suggest_read_tool_refused() -> None:
    # A read-capability tool never suggests a check, even if its name
    # looks like a writer: suggestions follow capability, not name.
    assert (
        suggest_postconditions(
            tool_name="desktop_file_read",
            args={"path": "/tmp/a.txt"},
            result=ToolResult.success(
                {"output": "written", "path": "/tmp/a.txt"},
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        == []
    )


def test_suggest_failed_call_and_unknown_tool() -> None:
    assert (
        suggest_postconditions(
            tool_name="workspace_write",
            args={"path": "/tmp/a.txt", "content": "written"},
            result=ToolResult(
                ok=False,
                error="denied",
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        == []
    )
    assert (
        suggest_postconditions(
            tool_name="some_future_tool",
            args={"path": "/tmp/a.txt"},
            result=ToolResult.success(
                {"output": "ok"},
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        == []
    )


def test_suggest_ignores_model_text_and_verified_flag() -> None:
    # The suggestion comes only from (name, args, result) via the
    # table. A result whose text mentions another file, and a
    # verified:true flag on a tool with no table row, suggest nothing
    # for that other file.
    checks = suggest_postconditions(
        tool_name="workspace_write",
        args={"path": "/tmp/a.txt", "content": "written"},
        result=ToolResult.success(
            {"output": "wrote /tmp/evil.txt"},
        ),
        get_capability=_capability,
        reader=_reader(),
    )
    assert all(getattr(pc.params, "path", None) == "/tmp/a.txt" for pc in checks)
    assert (
        suggest_postconditions(
            tool_name="desktop_browser",
            args={"operation": "navigate", "url": "https://example.com"},
            result=ToolResult.success(
                {"verified": True},
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        == []
    )
    # A shell result whose output *looks* like a check description
    # still suggests nothing: model/tool text is never parsed.
    assert (
        suggest_postconditions(
            tool_name="desktop_shell",
            args={"command": "echo file_exists:/tmp/a.txt"},
            result=ToolResult.success(
                {"output": "file_exists:/tmp/a.txt"},
            ),
            get_capability=_capability,
            reader=_reader(),
        )
        == []
    )


# ---------------------------------------------------------------------------
# §2.2 — precedence.
# ---------------------------------------------------------------------------


def test_expected_wins_per_subject_superseded_recorded() -> None:
    step = {
        "expected_postconditions": [
            _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="/tmp/a.txt")).to_json()
        ]
    }
    loop = _loop(
        _call(
            "workspace_write",
            {"path": "/tmp/a.txt", "content": "written"},
            {"output": "ok", "path": "/tmp/a.txt"},
        )
    )
    bound = bind_step_postconditions(
        step=step, loop_result=loop, get_capability=_capability, reader=_reader()
    )
    # The expected file_exists covers the suggested one (same subject);
    # the suggested file_content_matches is a different kind, so it
    # survives — expected wins per subject, not per file.
    assert _kinds(bound) == ["file_exists", "file_content_matches"]
    assert bound[0].source == "test"
    assert bound[1].source == "suggested:workspace_write"


def test_expected_file_absent_supersedes_rename_suggestion() -> None:
    step = {
        "expected_postconditions": [
            _bound(PostconditionKind.FILE_ABSENT, FileAbsentParams(path="/tmp/old.txt")).to_json()
        ]
    }
    loop = _loop(
        _call(
            "workspace_rename",
            {"old_path": "/tmp/old.txt", "new_path": "/tmp/a.txt"},
            {"output": "ok", "old_path": "/tmp/old.txt", "new_path": "/tmp/a.txt"},
        )
    )
    bound = bind_step_postconditions(
        step=step, loop_result=loop, get_capability=_capability, reader=_reader()
    )
    assert _kinds(bound) == ["file_absent", "file_exists"]
    assert bound[0].source == "test"
    assert bound[1].source == "suggested:workspace_rename"


def test_browser_declaration_becomes_expected() -> None:
    # Q8: a click with a browser_state declaration converts to an
    # expected browser_state condition automatically.
    step: dict = {}
    loop = _loop(
        _call(
            "desktop_browser",
            {
                "operation": "click",
                "browser_state": {
                    "entity_id": "publish-btn",
                    "expect": {"kind": "text", "after": "Published"},
                },
            },
            {"output": "ok", "verified": True},
        )
    )
    bound = bind_step_postconditions(
        step=step, loop_result=loop, get_capability=_capability, reader=_reader()
    )
    assert _kinds(bound) == ["browser_state"]
    assert bound[0].params == BrowserStateParams(
        entity_id="publish-btn", kind="text", after="Published"
    )
    assert bound[0].source == "expected:browser_state"


def test_browser_click_without_declaration_suggests_nothing() -> None:
    loop = _loop(
        _call(
            "desktop_browser",
            {"operation": "click"},
            {"output": "ok", "verified": True},
        )
    )
    assert (
        bind_step_postconditions(
            step={}, loop_result=loop, get_capability=_capability, reader=_reader()
        )
        == []
    )


def test_bind_run_postconditions_expected_plus_suggested() -> None:
    expected = [_bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="/tmp/a.txt"))]
    loop = _loop(
        _call(
            "workspace_rename",
            {"old_path": "/tmp/old.txt", "new_path": "/tmp/b.txt"},
            {"output": "ok", "old_path": "/tmp/old.txt", "new_path": "/tmp/b.txt"},
        )
    )
    bound = bind_run_postconditions(
        expected=expected, loop_result=loop, get_capability=_capability, reader=_reader()
    )
    assert _kinds(bound) == ["file_exists", "file_exists", "file_absent"]
    assert bound[0].source == "test"
    assert bound[1].params == FileExistsParams(path="/tmp/b.txt")


# ---------------------------------------------------------------------------
# §2.4 — the step hook.
# ---------------------------------------------------------------------------


def test_hook_evaluates_expected_and_returns_none_when_empty() -> None:
    hook = make_step_verification_hook(reader=_reader(), get_capability=_capability)
    step = {
        "expected_postconditions": [
            _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="/tmp/a.txt")).to_json()
        ]
    }
    verification = hook(step, _loop())
    assert verification is not None
    assert verification.outcome is VerificationOutcome.PASSED
    assert hook.last_binding_notes == ()
    assert hook({}, _loop()) is None


def test_hook_evaluates_suggested_write() -> None:
    hook = make_step_verification_hook(reader=_reader(), get_capability=_capability)
    loop = _loop(
        _call(
            "workspace_write",
            {"path": "/tmp/a.txt", "content": "written"},
            {"output": "ok", "path": "/tmp/a.txt"},
        )
    )
    verification = hook({}, loop)
    assert verification is not None
    assert verification.outcome is VerificationOutcome.PASSED
    assert {check.name.split(":")[0] for check in verification.checks} == {
        "file_exists",
        "file_content_matches",
    }


def test_hook_toctou_deleted_between_effect_and_check_fails() -> None:
    # The write happened, but the file is gone by check time: the
    # suggestion evaluates FAILED — an honest failure, not a pass
    # from the tool's ok flag.
    hook = make_step_verification_hook(reader=FakeReader(), get_capability=_capability)
    loop = _loop(
        _call(
            "workspace_write",
            {"path": "/tmp/a.txt", "content": "written"},
            {"output": "ok", "path": "/tmp/a.txt"},
        )
    )
    verification = hook({}, loop)
    assert verification is not None
    assert verification.outcome is VerificationOutcome.FAILED


def test_hook_records_superseded_note() -> None:
    hook = make_step_verification_hook(reader=_reader(), get_capability=_capability)
    step = {
        "expected_postconditions": [
            _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="/tmp/a.txt")).to_json()
        ]
    }
    loop = _loop(
        _call(
            "workspace_write",
            {"path": "/tmp/a.txt", "content": "written"},
            {"output": "ok", "path": "/tmp/a.txt"},
        )
    )
    verification = hook(step, loop)
    assert verification is not None
    assert verification.outcome is VerificationOutcome.PASSED
    assert len(hook.last_binding_notes) == 1
    assert hook.last_binding_notes[0].startswith(
        "suggestion superseded by expected: file_exists:/tmp/a.txt"
    )


# ---------------------------------------------------------------------------
# Q12 — hash stability.
# ---------------------------------------------------------------------------

_CANONICAL_PLAN = {
    "plan_id": "plan-1",
    "goal": "Тестовый план",
    "status": "approved",
    "plan_revision": 1,
    "steps": [
        {
            "step_id": "step-1",
            "title": "Шаг step-1",
            "description": "Проверить",
            "allowed_tool_kinds": ["workspace_write"],
            "inputs": {
                "operation": "workspace_write",
                "tool_args": {"path": "x.txt", "content": "written"},
            },
            "expected_outputs": [],
            "acceptance_checks": [],
            "status": "todo",
            "evidence": None,
        }
    ],
}


def test_plan_hash_without_expected_is_stable() -> None:
    assert plan_hash_payload(dict(_CANONICAL_PLAN)) == PLAN_HASH_NO_EXPECTED


def test_packet_hash_payload_without_expected_matches_pre_t02_shape() -> None:
    # The packet hash cannot pin to an absolute value: compile
    # generates a random task_id per call and the hash covers it.
    # Instead prove the hashed payload of a packet WITHOUT the field
    # is byte-identical to the pre-T02 inline shape: rebuild that
    # shape by hand (the old task_packet_hash_payload body) and hash
    # it — the function's result must equal it exactly.
    from core.mwv.models import task_packet_hash_payload

    packet = compile_plan_to_task_packet(
        plan=dict(_CANONICAL_PLAN),
        session_id="s1",
        trace_id="t1",
        workspace_root="/tmp",
    )
    assert packet.steps[0].expected_postconditions == []
    pre_t02_payload = {
        "task_id": packet.task_id,
        "packet_revision": packet.packet_revision,
        "session_id": packet.session_id,
        "trace_id": packet.trace_id,
        "goal": packet.goal,
        "messages": [
            {"role": message.role, "content": message.content} for message in packet.messages
        ],
        "steps": [
            {
                "step_id": step.step_id,
                "title": step.title,
                "description": step.description,
                "allowed_tool_kinds": list(step.allowed_tool_kinds),
                "inputs": dict(step.inputs),
                "expected_outputs": list(step.expected_outputs),
                "acceptance_checks": list(step.acceptance_checks),
            }
            for step in packet.steps
        ],
        "constraints": list(packet.constraints),
        "policy": dict(packet.policy),
        "scope": dict(packet.scope),
        "budgets": dict(packet.budgets),
        "approvals": dict(packet.approvals),
        "verifier": dict(packet.verifier),
        "context": dict(packet.context),
    }
    expected_hash = hashlib.sha256(
        json.dumps(
            pre_t02_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert task_packet_hash_payload(packet) == expected_hash
    assert packet.packet_hash == expected_hash


def test_packet_hash_changes_when_expected_added() -> None:
    # Same packet, same task_id: with the field present the hash must
    # differ from the no-field payload — the field binds to the hash.
    from core.mwv.models import task_packet_hash_payload

    plan_with = dict(_CANONICAL_PLAN)
    step_with = dict(_CANONICAL_PLAN["steps"][0])
    step_with["expected_postconditions"] = [
        _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="x.txt")).to_json()
    ]
    plan_with["steps"] = [step_with]
    packet = compile_plan_to_task_packet(
        plan=plan_with, session_id="s1", trace_id="t1", workspace_root="/tmp"
    )
    assert len(packet.steps[0].expected_postconditions) == 1
    no_field_payload_hash = hashlib.sha256(
        json.dumps(
            {
                "task_id": packet.task_id,
                "packet_revision": packet.packet_revision,
                "session_id": packet.session_id,
                "trace_id": packet.trace_id,
                "goal": packet.goal,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in packet.messages
                ],
                "steps": [
                    {
                        "step_id": step.step_id,
                        "title": step.title,
                        "description": step.description,
                        "allowed_tool_kinds": list(step.allowed_tool_kinds),
                        "inputs": dict(step.inputs),
                        "expected_outputs": list(step.expected_outputs),
                        "acceptance_checks": list(step.acceptance_checks),
                    }
                    for step in packet.steps
                ],
                "constraints": list(packet.constraints),
                "policy": dict(packet.policy),
                "scope": dict(packet.scope),
                "budgets": dict(packet.budgets),
                "approvals": dict(packet.approvals),
                "verifier": dict(packet.verifier),
                "context": dict(packet.context),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert task_packet_hash_payload(packet) != no_field_payload_hash


def test_plan_hash_changes_when_expected_added() -> None:
    plan_with = dict(_CANONICAL_PLAN)
    step_with = dict(_CANONICAL_PLAN["steps"][0])
    step_with["expected_postconditions"] = [
        _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="x.txt")).to_json()
    ]
    plan_with["steps"] = [step_with]
    assert plan_hash_payload(plan_with) != PLAN_HASH_NO_EXPECTED
    packet = compile_plan_to_task_packet(
        plan=plan_with, session_id="s1", trace_id="t1", workspace_root="/tmp"
    )
    assert len(packet.steps[0].expected_postconditions) == 1


# ---------------------------------------------------------------------------
# §2.1/§2.3 — normalize / compile / serialize plumbing.
# ---------------------------------------------------------------------------

_STATUSES = {"todo", "in_progress", "done", "failed", "skipped"}


def _plan_step(*, expected: object = None, include_field: bool = True) -> dict:
    step = {
        "step_id": "step-1",
        "title": "Шаг step-1",
        "description": "Проверить",
        "allowed_tool_kinds": ["workspace_write"],
        "inputs": {"operation": "workspace_write", "tool_args": {"path": "x.txt"}},
        "expected_outputs": [],
        "acceptance_checks": [],
        "status": "todo",
        "evidence": None,
    }
    if include_field:
        step["expected_postconditions"] = expected
    return step


def test_normalize_plan_step_keeps_expected_canonical() -> None:
    normalized = normalize_plan_step(
        _plan_step(
            expected=[
                _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="x.txt")).to_json()
            ]
        ),
        plan_step_statuses=_STATUSES,
    )
    assert normalized is not None
    assert normalized["expected_postconditions"] == [
        _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="x.txt")).to_json()
    ]


def test_normalize_plan_step_without_field_has_no_key() -> None:
    normalized = normalize_plan_step(_plan_step(include_field=False), plan_step_statuses=_STATUSES)
    assert normalized is not None
    assert "expected_postconditions" not in normalized


def test_normalize_plan_step_invalid_expected_drops_step() -> None:
    assert (
        normalize_plan_step(
            _plan_step(expected=[{"kind": "not_a_kind", "params": {}}]),
            plan_step_statuses=_STATUSES,
        )
        is None
    )
    assert (
        normalize_plan_step(_plan_step(expected="not-a-list"), plan_step_statuses=_STATUSES) is None
    )


def test_compile_serialize_deserialize_round_trip() -> None:
    plan = dict(_CANONICAL_PLAN)
    step = dict(_CANONICAL_PLAN["steps"][0])
    step["expected_postconditions"] = [
        _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="x.txt")).to_json()
    ]
    plan["steps"] = [step]
    packet = compile_plan_to_task_packet(
        plan=plan, session_id="s1", trace_id="t1", workspace_root="/tmp"
    )
    payload = serialize_task_packet_payload(packet)
    assert payload["steps"][0]["expected_postconditions"] == [
        _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="x.txt")).to_json()
    ]
    restored = deserialize_task_packet_payload(payload)
    assert restored.steps[0].expected_postconditions == packet.steps[0].expected_postconditions
    assert restored.packet_hash == packet.packet_hash


def test_serialize_without_expected_has_no_key() -> None:
    packet = compile_plan_to_task_packet(
        plan=dict(_CANONICAL_PLAN),
        session_id="s1",
        trace_id="t1",
        workspace_root="/tmp",
    )
    payload = serialize_task_packet_payload(packet)
    assert "expected_postconditions" not in payload["steps"][0]
    assert deserialize_task_packet_payload(payload).steps[0].expected_postconditions == []


def test_compile_invalid_expected_is_fail_closed() -> None:
    plan = dict(_CANONICAL_PLAN)
    step = dict(_CANONICAL_PLAN["steps"][0])
    step["expected_postconditions"] = [{"kind": "not_a_kind", "params": {}}]
    plan["steps"] = [step]
    with pytest.raises(ValueError, match="expected_postconditions"):
        compile_plan_to_task_packet(
            plan=plan, session_id="s1", trace_id="t1", workspace_root="/tmp"
        )


def test_deserialize_invalid_expected_is_fail_closed() -> None:
    packet = compile_plan_to_task_packet(
        plan=dict(_CANONICAL_PLAN),
        session_id="s1",
        trace_id="t1",
        workspace_root="/tmp",
    )
    payload = serialize_task_packet_payload(packet)
    payload["steps"][0]["expected_postconditions"] = [{"kind": "not_a_kind", "params": {}}]
    with pytest.raises(ValueError, match="expected_postconditions"):
        deserialize_task_packet_payload(payload)


def test_check_names_cover_binding_subjects() -> None:
    # The note format the executor appends to the step result text.
    pc = _bound(PostconditionKind.FILE_EXISTS, FileExistsParams(path="/tmp/a.txt"))
    assert check_name(pc) == "file_exists:/tmp/a.txt"
