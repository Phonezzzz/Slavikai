"""Regression tests: Computer activity/result UX separation from visible Chat.

Закрепляет invariants PR-17:
- role="tool" не попадает в visible chat messages (UIHub отклоняет его на уровне
  _build_message / _normalize_message_payload — `_MESSAGE_ROLES` = {user, assistant, system}).
- computer_events хранятся отдельно от chat messages в _SessionState.computer_events.
- session snapshot содержит computer_events как отдельный ключ.
- Drain pipeline: agent.drain_computer_events() → hub.append_computer_events() → snapshot.
- Error events передают ok=False без изменений.
- Visible chat content (assistant prose) — не raw tool JSON.
- ComputerActivityLog и UIHub не пересекаются в хранении.

Нет Docker, нет FakeContainerRunner, нет lane="computer", нет /ui/api/computer/send.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from core.agent_computer import build_computer_changes_review_decision, execute_local_commit
from core.computer_activity_log import ComputerActivityLog, build_computer_activity_summary
from core.computer_backend import LocalComputerBackend
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig, ToolCall, ToolSpec
from server.ui_hub import MAX_COMPUTER_EVENTS, UIHub
from shared.models import LLMMessage, ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────────────────────


class _ScriptedBrain(Brain):
    def __init__(self, script: list[LLMResult]) -> None:
        self._script = list(script)

    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        if not self._script:
            return LLMResult(text="(empty)")
        return self._script.pop(0)


def _make_registry(tool_name: str, result: ToolResult) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool_name, lambda req: result, enabled=True, capability="read")
    return registry


def _make_hub() -> UIHub:
    return UIHub()


# ── UIHub: role="tool" guard ──────────────────────────────────────────────────


def test_build_message_rejects_tool_role() -> None:
    """create_message raises ValueError for role='tool' — never enters storage."""
    hub = _make_hub()
    with pytest.raises(ValueError, match="unsupported message role"):
        hub.create_message(role="tool", content="raw result json", lane="chat")


def test_build_message_rejects_computer_role() -> None:
    """create_message raises ValueError for role='computer' — lane must not exist."""
    hub = _make_hub()
    with pytest.raises(ValueError, match="unsupported message role"):
        hub.create_message(role="computer", content="x", lane="chat")


def test_append_message_rejects_raw_tool_dict() -> None:
    """hub.append_message() rejects raw dict with role='tool' at normalisation."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        raw_tool_msg: dict[str, object] = {
            "message_id": "m-tool",
            "role": "tool",
            "content": '{"ok": true}',
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        with pytest.raises(ValueError):
            await hub.append_message(session_id, raw_tool_msg, lane="chat")  # type: ignore[arg-type]

    asyncio.run(run())


def test_visible_chat_messages_never_contain_tool_role() -> None:
    """After a sequence of user+assistant messages, no role='tool' in messages."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        user_msg = hub.create_message(role="user", content="hi", lane="chat")
        assistant_msg = hub.create_message(role="assistant", content="hello", lane="chat")
        await hub.append_message(session_id, user_msg, lane="chat")
        await hub.append_message(session_id, assistant_msg, lane="chat")
        messages = await hub.get_messages(session_id, lane="chat")
        roles = {m["role"] for m in messages}
        assert "tool" not in roles
        assert "computer" not in roles
        assert roles <= {"user", "assistant", "system"}

    asyncio.run(run())


# ── UIHub: computer_events storage ───────────────────────────────────────────


def test_hub_append_and_get_computer_events() -> None:
    """append_computer_events stores events; get_computer_events returns them."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        events = [
            {"kind": "tool_started", "tool": "workspace_read", "ts": 1},
            {"kind": "file_read", "tool": "workspace_read", "ts": 2, "ok": True, "path": "x.py"},
        ]
        await hub.append_computer_events(session_id, events)
        stored = await hub.get_computer_events(session_id)
        assert len(stored) == 2
        assert stored[0]["kind"] == "tool_started"
        assert stored[1]["kind"] == "file_read"

    asyncio.run(run())


def test_computer_events_in_session_snapshot() -> None:
    """get_session() snapshot includes computer_events as a separate key."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        events = [{"kind": "file_read", "tool": "workspace_read", "ts": 1, "ok": True}]
        await hub.append_computer_events(session_id, events)
        snapshot = await hub.get_session(session_id)
        assert snapshot is not None
        assert "computer_events" in snapshot
        stored = snapshot["computer_events"]
        assert isinstance(stored, list)
        assert len(stored) == 1
        assert stored[0]["kind"] == "file_read"

    asyncio.run(run())


def test_computer_events_separate_from_chat_messages() -> None:
    """computer_events key is separate from messages key in session snapshot."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        user_msg = hub.create_message(role="user", content="read x.py", lane="chat")
        asst_msg = hub.create_message(role="assistant", content="Done.", lane="chat")
        await hub.append_message(session_id, user_msg, lane="chat")
        await hub.append_message(session_id, asst_msg, lane="chat")
        await hub.append_computer_events(
            session_id,
            [{"kind": "file_read", "tool": "workspace_read", "ts": 1, "ok": True, "path": "x.py"}],
        )
        snapshot = await hub.get_session(session_id)
        assert snapshot is not None
        chat_messages: list[object] = snapshot["messages"]  # type: ignore[assignment]
        computer_events: list[object] = snapshot["computer_events"]  # type: ignore[assignment]
        assert len(chat_messages) == 2
        assert len(computer_events) == 1
        # messages must NOT contain computer events
        for msg in chat_messages:
            assert isinstance(msg, dict)
            assert msg.get("role") != "tool"
            assert msg.get("kind") != "file_read"

    asyncio.run(run())


def test_computer_events_capped_at_max() -> None:
    """Events beyond MAX_COMPUTER_EVENTS are trimmed (oldest dropped)."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        batch: list[dict[str, object]] = [
            {"kind": "file_read", "tool": "workspace_read", "ts": i, "ok": True}
            for i in range(MAX_COMPUTER_EVENTS + 10)
        ]
        await hub.append_computer_events(session_id, batch)
        stored = await hub.get_computer_events(session_id)
        assert len(stored) == MAX_COMPUTER_EVENTS
        # most recent events kept
        assert stored[-1]["ts"] == MAX_COMPUTER_EVENTS + 9

    asyncio.run(run())


def test_computer_event_error_ok_false_preserved() -> None:
    """Events with ok=False are stored intact."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        error_event: dict[str, object] = {
            "kind": "file_read",
            "tool": "workspace_read",
            "ts": 1,
            "ok": False,
            "error": "file not found",
        }
        await hub.append_computer_events(session_id, [error_event])
        stored = await hub.get_computer_events(session_id)
        assert len(stored) == 1
        assert stored[0]["ok"] is False
        assert stored[0]["error"] == "file not found"

    asyncio.run(run())


def test_computer_events_not_in_messages_lane() -> None:
    """computer_events do not appear in chat or workspace message lanes."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        await hub.append_computer_events(
            session_id,
            [{"kind": "command_finished", "tool": "workspace_terminal_run", "ts": 1, "ok": True}],
        )
        chat_msgs = await hub.get_messages(session_id, lane="chat")
        workspace_msgs = await hub.get_messages(session_id, lane="workspace")
        assert chat_msgs == []
        assert workspace_msgs == []

    asyncio.run(run())


# ── Drain pipeline: ComputerActivityLog → UIHub ───────────────────────────────


def test_drain_computer_events_returns_json_serializable_dicts() -> None:
    """Events from ComputerActivityLog.drain() are plain dicts suitable for hub storage."""
    registry = _make_registry("workspace_read", ToolResult.success({"content": "hi"}))
    log = ComputerActivityLog()
    gw = ToolGateway(
        registry=registry,
        pre_call=log.pre_call,
        post_call=log.post_call,
        approval_context=None,
    )
    tc = ToolCall(id="tc-drain", name="workspace_read", arguments={"path": "a.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="Read."),
        ]
    )
    AgentToolLoop().run(
        brain=brain,
        gateway=gw,
        messages=[LLMMessage(role="user", content="read a.py")],
        tools=registry.list_tool_specs(),
    )
    events = log.drain()
    assert len(events) >= 1
    for ev in events:
        assert isinstance(ev, dict)
        # Must be JSON-serialisable (same type UIHub stores and serialises to frontend)
        serialised = json.dumps(ev)
        assert json.loads(serialised) == ev


def test_drain_then_hub_append_computer_events_round_trip() -> None:
    """Events drained from ComputerActivityLog can be stored in UIHub and retrieved."""

    async def run() -> None:
        registry = _make_registry("workspace_read", ToolResult.success({"content": "hi"}))
        log = ComputerActivityLog()
        gw = ToolGateway(
            registry=registry,
            pre_call=log.pre_call,
            post_call=log.post_call,
            approval_context=None,
        )
        tc = ToolCall(id="tc-rt", name="workspace_read", arguments={"path": "a.py"})
        brain = _ScriptedBrain(
            [
                LLMResult(text="", tool_calls=[tc]),
                LLMResult(text="Done."),
            ]
        )
        loop_result = AgentToolLoop().run(
            brain=brain,
            gateway=gw,
            messages=[LLMMessage(role="user", content="read a.py")],
            tools=registry.list_tool_specs(),
        )
        events = log.drain()
        assert len(events) >= 1

        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")

        # Store user + assistant messages (prose only, no tool messages)
        user_msg = hub.create_message(role="user", content="read a.py", lane="chat")
        asst_msg = hub.create_message(role="assistant", content=loop_result.text, lane="chat")
        await hub.append_message(session_id, user_msg, lane="chat")
        await hub.append_message(session_id, asst_msg, lane="chat")

        # Store computer events separately
        await hub.append_computer_events(session_id, events)

        snapshot = await hub.get_session(session_id)
        assert snapshot is not None
        chat_messages: list[object] = snapshot["messages"]  # type: ignore[assignment]
        computer_events: list[object] = snapshot["computer_events"]  # type: ignore[assignment]

        # Chat: only user + assistant, not tool events
        assert len(chat_messages) == 2
        valid_roles = {"user", "assistant"}
        assert all(isinstance(m, dict) and m["role"] in valid_roles for m in chat_messages)

        # Computer events: includes at least one
        assert len(computer_events) >= 1
        kinds = [ev["kind"] for ev in computer_events if isinstance(ev, dict)]
        assert "tool_started" in kinds

    asyncio.run(run())


# ── Visible Chat body: prose, not raw JSON ────────────────────────────────────


def test_assistant_message_content_is_prose_not_raw_json() -> None:
    """The assistant message stored in hub contains prose, not raw tool JSON."""

    async def run() -> None:
        registry = _make_registry("workspace_read", ToolResult.success({"content": "hello"}))
        log = ComputerActivityLog()
        gw = ToolGateway(
            registry=registry,
            pre_call=log.pre_call,
            post_call=log.post_call,
            approval_context=None,
        )
        tc = ToolCall(id="tc-prose", name="workspace_read", arguments={"path": "a.py"})
        brain = _ScriptedBrain(
            [
                LLMResult(text="", tool_calls=[tc]),
                LLMResult(text="The file says: hello"),
            ]
        )
        loop_result = AgentToolLoop().run(
            brain=brain,
            gateway=gw,
            messages=[LLMMessage(role="user", content="read a.py")],
            tools=registry.list_tool_specs(),
        )
        # The final text is prose
        assert loop_result.text == "The file says: hello"
        assert not loop_result.text.startswith("{")

        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        asst = hub.create_message(role="assistant", content=loop_result.text, lane="chat")
        await hub.append_message(session_id, asst, lane="chat")
        msgs = await hub.get_messages(session_id, lane="chat")
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert isinstance(content, str)
        assert not content.startswith("{")
        assert "The file says" in content

    asyncio.run(run())


def test_tool_failure_assistant_prose_not_error_json() -> None:
    """Even on tool failure, the assistant message is prose and computer_events have ok=False."""

    async def run() -> None:
        registry = _make_registry("workspace_read", ToolResult.failure("file not found"))
        log = ComputerActivityLog()
        gw = ToolGateway(
            registry=registry,
            pre_call=log.pre_call,
            post_call=log.post_call,
            approval_context=None,
        )
        tc = ToolCall(id="tc-fail", name="workspace_read", arguments={"path": "missing.py"})
        brain = _ScriptedBrain(
            [
                LLMResult(text="", tool_calls=[tc]),
                LLMResult(text="Sorry, the file was not found."),
            ]
        )
        loop_result = AgentToolLoop().run(
            brain=brain,
            gateway=gw,
            messages=[LLMMessage(role="user", content="read missing.py")],
            tools=registry.list_tool_specs(),
        )
        assert loop_result.text == "Sorry, the file was not found."
        assert not loop_result.text.startswith("{")

        events = log.drain()
        failure_events = [e for e in events if e.get("ok") is False]
        assert len(failure_events) >= 1
        assert failure_events[0]["tool"] == "workspace_read"

        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        asst = hub.create_message(role="assistant", content=loop_result.text, lane="chat")
        await hub.append_message(session_id, asst, lane="chat")
        await hub.append_computer_events(session_id, events)

        snapshot = await hub.get_session(session_id)
        assert snapshot is not None
        chat_msgs: list[object] = snapshot["messages"]  # type: ignore[assignment]
        comp_events: list[object] = snapshot["computer_events"]  # type: ignore[assignment]

        # Chat shows prose
        assert len(chat_msgs) == 1
        assert isinstance(chat_msgs[0], dict)
        assert "Sorry" in str(chat_msgs[0]["content"])

        # Computer events show technical detail
        err_events = [e for e in comp_events if isinstance(e, dict) and e.get("ok") is False]
        assert len(err_events) >= 1

    asyncio.run(run())


# ── No lane="computer", no role="computer" ───────────────────────────────────


def test_no_lane_computer_in_any_stored_message() -> None:
    """No stored message carries lane='computer' — Computer is not a chat lane."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        user_msg = hub.create_message(role="user", content="hi", lane="chat")
        asst_msg = hub.create_message(role="assistant", content="hello", lane="chat")
        await hub.append_message(session_id, user_msg, lane="chat")
        await hub.append_message(session_id, asst_msg, lane="chat")
        snapshot = await hub.get_session(session_id)
        assert snapshot is not None
        for msg in snapshot["messages"]:  # type: ignore[union-attr]
            assert isinstance(msg, dict)
            assert msg.get("lane") != "computer"
            assert msg.get("role") != "computer"

    asyncio.run(run())


def test_hub_append_message_lane_computer_is_rejected() -> None:
    """append_message rejects lane='computer' in raw dict — not in _MESSAGE_LANES."""

    async def run() -> None:
        hub = _make_hub()
        session_id = await hub.get_or_create_session(None, "p1")
        raw_msg: dict[str, object] = {
            "message_id": "m-comp",
            "role": "user",
            "lane": "computer",
            "content": "x",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        with pytest.raises(ValueError, match="lane must be"):
            await hub.append_message(session_id, raw_msg, lane="chat")  # type: ignore[arg-type]

    asyncio.run(run())


# ── PR-18: build_computer_activity_summary ────────────────────────────────────


def test_summary_empty_events() -> None:
    """Empty events → all-zero summary, no errors, no diff."""
    s = build_computer_activity_summary([])
    assert s["total_events"] == 0
    assert s["tools_started"] == 0
    assert s["tools_finished"] == 0
    assert s["files_read"] == 0
    assert s["files_written"] == 0
    assert s["commands_run"] == 0
    assert s["errors_count"] == 0
    assert s["latest_error"] is None
    assert s["has_diff"] is False
    assert s["tests_seen"] is False


def test_summary_counts_file_read_events() -> None:
    events: list[dict[str, object]] = [
        {"kind": "tool_started", "tool": "workspace_read", "ts": 1.0},
        {"kind": "file_read", "tool": "workspace_read", "ts": 1.1, "ok": True},
        {"kind": "tool_started", "tool": "workspace_read", "ts": 2.0},
        {"kind": "file_read", "tool": "workspace_read", "ts": 2.1, "ok": True},
    ]
    s = build_computer_activity_summary(events)
    assert s["total_events"] == 4
    assert s["tools_started"] == 2
    assert s["tools_finished"] == 2
    assert s["files_read"] == 2
    assert s["files_written"] == 0


def test_summary_counts_file_written_events() -> None:
    events: list[dict[str, object]] = [
        {"kind": "tool_started", "tool": "workspace_write", "ts": 1.0},
        {"kind": "file_written", "tool": "workspace_write", "ts": 1.1, "ok": True},
        {"kind": "tool_started", "tool": "workspace_patch", "ts": 2.0},
        {"kind": "file_written", "tool": "workspace_patch", "ts": 2.1, "ok": True},
    ]
    s = build_computer_activity_summary(events)
    assert s["files_written"] == 2
    assert s["files_read"] == 0


def test_summary_counts_command_events() -> None:
    events: list[dict[str, object]] = [
        {"kind": "tool_started", "tool": "workspace_terminal_run", "ts": 1.0},
        {
            "kind": "command_started",
            "tool": "workspace_terminal_run",
            "ts": 1.5,
            "ok": True,
            "command": "pytest",
        },
    ]
    s = build_computer_activity_summary(events)
    assert s["commands_run"] == 1
    assert s["tools_started"] == 1
    assert s["tools_finished"] == 1


def test_summary_errors_count() -> None:
    events: list[dict[str, object]] = [
        {"kind": "tool_started", "tool": "workspace_read", "ts": 1.0},
        {
            "kind": "file_read",
            "tool": "workspace_read",
            "ts": 1.1,
            "ok": False,
            "error": "file not found",
        },
    ]
    s = build_computer_activity_summary(events)
    assert s["errors_count"] == 1
    assert s["latest_error"] == "file not found"


def test_summary_latest_error_is_last_one() -> None:
    events: list[dict[str, object]] = [
        {"kind": "tool_started", "tool": "workspace_read", "ts": 1.0},
        {"kind": "file_read", "tool": "workspace_read", "ts": 1.1, "ok": False, "error": "first"},
        {"kind": "tool_started", "tool": "workspace_write", "ts": 2.0},
        {
            "kind": "file_written",
            "tool": "workspace_write",
            "ts": 2.1,
            "ok": False,
            "error": "second",
        },
    ]
    s = build_computer_activity_summary(events)
    assert s["errors_count"] == 2
    assert s["latest_error"] == "second"


def test_summary_has_diff_flag() -> None:
    events: list[dict[str, object]] = [
        {"kind": "tool_started", "tool": "workspace_diff", "ts": 1.0},
        {"kind": "git_diff_updated", "tool": "workspace_diff", "ts": 1.1, "ok": True},
    ]
    s = build_computer_activity_summary(events)
    assert s["has_diff"] is True


def test_summary_has_diff_false_by_default() -> None:
    events: list[dict[str, object]] = [
        {"kind": "tool_started", "tool": "workspace_read", "ts": 1.0},
        {"kind": "file_read", "tool": "workspace_read", "ts": 1.1, "ok": True},
    ]
    s = build_computer_activity_summary(events)
    assert s["has_diff"] is False


def test_summary_tests_seen_flag() -> None:
    events: list[dict[str, object]] = [
        {"kind": "tool_started", "tool": "workspace_test", "ts": 1.0},
        {"kind": "test_result", "tool": "workspace_test", "ts": 1.2, "ok": True},
    ]
    s = build_computer_activity_summary(events)
    assert s["tests_seen"] is True


def test_summary_is_json_serializable() -> None:
    import json

    events: list[dict[str, object]] = [
        {"kind": "tool_started", "tool": "workspace_read", "ts": 1.0},
        {
            "kind": "file_read",
            "tool": "workspace_read",
            "ts": 1.1,
            "ok": False,
            "error": "err",
        },
    ]
    s = build_computer_activity_summary(events)
    serialised = json.dumps(s)
    assert json.loads(serialised) == s


def test_hub_session_snapshot_includes_computer_summary() -> None:
    """get_session() returns computer_summary key alongside computer_events."""

    async def run() -> None:
        hub = UIHub()
        session_id = await hub.get_or_create_session(None, "p1")
        events: list[dict[str, object]] = [
            {"kind": "tool_started", "tool": "workspace_read", "ts": 1.0},
            {"kind": "file_read", "tool": "workspace_read", "ts": 1.1, "ok": True, "path": "x.py"},
        ]
        await hub.append_computer_events(session_id, events)
        snapshot = await hub.get_session(session_id)
        assert snapshot is not None
        assert "computer_summary" in snapshot
        summary = snapshot["computer_summary"]
        assert isinstance(summary, dict)
        assert summary["files_read"] == 1
        assert summary["errors_count"] == 0
        assert summary["total_events"] == 2

    asyncio.run(run())


def test_hub_get_computer_summary_method() -> None:
    """get_computer_summary() returns the same summary as from the snapshot."""

    async def run() -> None:
        hub = UIHub()
        session_id = await hub.get_or_create_session(None, "p1")
        await hub.append_computer_events(
            session_id,
            [
                {"kind": "tool_started", "tool": "workspace_write", "ts": 1.0},
                {
                    "kind": "file_written",
                    "tool": "workspace_write",
                    "ts": 1.1,
                    "ok": False,
                    "error": "disk full",
                },
            ],
        )
        summary = await hub.get_computer_summary(session_id)
        assert summary["files_written"] == 1
        assert summary["errors_count"] == 1
        assert summary["latest_error"] == "disk full"

    asyncio.run(run())


def test_hub_computer_summary_absent_session_returns_zero_summary() -> None:
    """get_computer_summary() for unknown session returns zero summary, no crash."""

    async def run() -> None:
        hub = UIHub()
        summary = await hub.get_computer_summary("nonexistent-session")
        assert summary["total_events"] == 0
        assert summary["errors_count"] == 0

    asyncio.run(run())


def test_summary_not_in_messages() -> None:
    """computer_summary does not appear inside the messages list."""

    async def run() -> None:
        hub = UIHub()
        session_id = await hub.get_or_create_session(None, "p1")
        await hub.append_computer_events(
            session_id,
            [
                {"kind": "tool_started", "tool": "workspace_read", "ts": 1.0},
                {"kind": "file_read", "tool": "workspace_read", "ts": 1.1, "ok": True},
            ],
        )
        snapshot = await hub.get_session(session_id)
        assert snapshot is not None
        msgs: list[object] = snapshot["messages"]  # type: ignore[assignment]
        for msg in msgs:
            assert isinstance(msg, dict)
            assert "computer_summary" not in msg
            assert "total_events" not in msg

    asyncio.run(run())


# ── LocalComputerBackend wires through ToolGateway, not direct file access ───


def test_local_backend_read_file_uses_tool_gateway_not_direct_io() -> None:
    """LocalComputerBackend.read_file() calls ToolGateway, not direct Python file I/O."""
    gateway_mock = MagicMock()
    gateway_mock.call.return_value = ToolResult.success({"content": "hello"})
    backend = LocalComputerBackend(gateway=gateway_mock)
    result = backend.read_file("README.md")
    assert result.ok
    gateway_mock.call.assert_called_once_with(ToolRequest("workspace_read", {"path": "README.md"}))


# ── PR-19: build_computer_changes_review_decision ────────────────────────────


def test_changes_review_returns_none_when_no_files() -> None:
    """No changed files → returns None, no packet emitted."""
    result = build_computer_changes_review_decision(
        changed_files=[],
        diff_summary="some diff",
        commit_message="fix: something",
    )
    assert result is None


def test_changes_review_returns_none_when_commit_message_empty() -> None:
    """Empty commit message → returns None."""
    result = build_computer_changes_review_decision(
        changed_files=["src/main.py"],
        diff_summary="some diff",
        commit_message="",
    )
    assert result is None


def test_changes_review_returns_none_when_commit_message_blank() -> None:
    """Whitespace-only commit message → returns None."""
    result = build_computer_changes_review_decision(
        changed_files=["src/main.py"],
        diff_summary="some diff",
        commit_message="   ",
    )
    assert result is None


def test_changes_review_packet_has_required_keys() -> None:
    """Packet has all required decision keys."""
    packet = build_computer_changes_review_decision(
        changed_files=["src/foo.py"],
        diff_summary="diff --git ...",
        commit_message="feat: add foo",
    )
    assert packet is not None
    for key in (
        "id",
        "kind",
        "decision_type",
        "status",
        "blocking",
        "reason",
        "summary",
        "proposed_action",
        "options",
        "default_option_id",
    ):
        assert key in packet, f"Missing key: {key}"


def test_changes_review_packet_decision_type_is_computer_commit() -> None:
    packet = build_computer_changes_review_decision(
        changed_files=["a.py"],
        diff_summary="",
        commit_message="fix: x",
    )
    assert packet is not None
    assert packet["decision_type"] == "computer_commit"
    assert packet["kind"] == "decision"
    assert packet["status"] == "pending"
    assert packet["blocking"] is True


def test_changes_review_packet_contains_changed_files() -> None:
    files = ["core/foo.py", "tests/test_foo.py"]
    packet = build_computer_changes_review_decision(
        changed_files=files,
        diff_summary="",
        commit_message="refactor: foo",
    )
    assert packet is not None
    action = packet["proposed_action"]
    assert isinstance(action, dict)
    assert action["changed_files"] == files


def test_changes_review_packet_contains_commit_message() -> None:
    packet = build_computer_changes_review_decision(
        changed_files=["x.py"],
        diff_summary="",
        commit_message="  feat: trimmed  ",
    )
    assert packet is not None
    action = packet["proposed_action"]
    assert isinstance(action, dict)
    assert action["proposed_commit_message"] == "feat: trimmed"


def test_changes_review_packet_contains_diff_summary() -> None:
    packet = build_computer_changes_review_decision(
        changed_files=["x.py"],
        diff_summary="diff --git a/x.py b/x.py\n+added line",
        commit_message="chore: update",
    )
    assert packet is not None
    action = packet["proposed_action"]
    assert isinstance(action, dict)
    assert action["diff_summary"] == "diff --git a/x.py b/x.py\n+added line"


def test_changes_review_packet_has_approve_and_reject_options() -> None:
    packet = build_computer_changes_review_decision(
        changed_files=["f.py"],
        diff_summary="",
        commit_message="fix: y",
    )
    assert packet is not None
    options = packet["options"]
    assert isinstance(options, list)
    option_ids = {opt["id"] for opt in options if isinstance(opt, dict)}
    assert "approve_once" in option_ids
    assert "reject" in option_ids


def test_changes_review_packet_default_option_is_approve() -> None:
    packet = build_computer_changes_review_decision(
        changed_files=["f.py"],
        diff_summary="",
        commit_message="fix: z",
    )
    assert packet is not None
    assert packet["default_option_id"] == "approve_once"


def test_changes_review_packet_is_json_serializable() -> None:
    """Packet must be fully JSON-serializable (no datetime, no custom objects)."""
    packet = build_computer_changes_review_decision(
        changed_files=["a.py", "b.py"],
        diff_summary="--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new",
        commit_message="feat: serialize check",
    )
    assert packet is not None
    serialized = json.dumps(packet)
    parsed = json.loads(serialized)
    assert parsed["decision_type"] == "computer_commit"


def test_changes_review_packet_id_is_unique() -> None:
    """Each call generates a distinct packet id."""
    p1 = build_computer_changes_review_decision(["a.py"], "", "fix: a")
    p2 = build_computer_changes_review_decision(["a.py"], "", "fix: a")
    assert p1 is not None
    assert p2 is not None
    assert p1["id"] != p2["id"]


def test_changes_review_no_commit_side_effect() -> None:
    """build_computer_changes_review_decision is a pure function — no I/O, no commit."""
    import subprocess

    before = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    build_computer_changes_review_decision(
        changed_files=["core/agent_computer.py"],
        diff_summary="some diff",
        commit_message="feat: would-be commit",
    )
    after = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert before == after, "git log changed — commit side effect detected!"


# ── PR-20: execute_local_commit ───────────────────────────────────────────────


def _ok_gateway(calls_log: list[str] | None = None) -> MagicMock:
    """Gateway mock that succeeds all calls and optionally logs command strings."""
    gateway = MagicMock()

    def _call(req: object) -> ToolResult:
        if calls_log is not None and hasattr(req, "args"):
            cmd = req.args.get("command", "")
            if cmd:
                calls_log.append(cmd)
        return ToolResult.success({"output": "ok"})

    gateway.call.side_effect = _call
    return gateway


def test_execute_commit_returns_failure_for_blank_message() -> None:
    """Blank commit_message → ToolResult.failure, no gateway call."""
    gw = MagicMock()
    result = execute_local_commit(commit_message="   ", changed_files=["f.py"], gateway=gw)
    assert not result.ok
    gw.call.assert_not_called()


def test_execute_commit_returns_failure_for_empty_files() -> None:
    """Empty changed_files → ToolResult.failure, no gateway call."""
    gw = MagicMock()
    result = execute_local_commit(commit_message="fix: x", changed_files=[], gateway=gw)
    assert not result.ok
    gw.call.assert_not_called()


def test_execute_commit_calls_git_add_then_commit() -> None:
    """Gateway receives git add then git commit, in that order."""
    calls: list[str] = []
    gw = _ok_gateway(calls)
    result = execute_local_commit(
        commit_message="feat: new feature",
        changed_files=["core/foo.py"],
        gateway=gw,
    )
    assert result.ok
    assert len(calls) == 2
    assert calls[0].startswith("git add")
    assert calls[1].startswith("git commit")


def test_execute_commit_uses_gateway_not_subprocess() -> None:
    """execute_local_commit routes through ToolGateway, not direct subprocess."""
    gw = _ok_gateway()
    execute_local_commit(
        commit_message="fix: via gateway",
        changed_files=["x.py"],
        gateway=gw,
    )
    assert gw.call.call_count == 2


def test_execute_commit_does_not_call_git_push() -> None:
    """No git push command is ever issued."""
    calls: list[str] = []
    gw = _ok_gateway(calls)
    execute_local_commit("feat: safe", ["a.py"], gw)
    for cmd in calls:
        assert "push" not in cmd.lower(), f"Unexpected push in command: {cmd!r}"


def test_execute_commit_does_not_call_git_merge() -> None:
    """No git merge command is ever issued."""
    calls: list[str] = []
    gw = _ok_gateway(calls)
    execute_local_commit("feat: safe", ["a.py"], gw)
    for cmd in calls:
        assert "merge" not in cmd.lower(), f"Unexpected merge in command: {cmd!r}"


def test_execute_commit_does_not_call_git_checkout() -> None:
    """No git checkout command is ever issued."""
    calls: list[str] = []
    gw = _ok_gateway(calls)
    execute_local_commit("feat: safe", ["a.py"], gw)
    for cmd in calls:
        assert "checkout" not in cmd.lower(), f"Unexpected checkout in command: {cmd!r}"


def test_execute_commit_propagates_add_failure() -> None:
    """If git add fails, returns that failure and git commit is NOT called."""
    add_called = False
    commit_called = False

    def _call(req: object) -> ToolResult:
        nonlocal add_called, commit_called
        cmd = req.args.get("command", "") if hasattr(req, "args") else ""
        if "git add" in cmd:
            add_called = True
            return ToolResult.failure("nothing to stage")
        if "git commit" in cmd:
            commit_called = True
            return ToolResult.success({"output": "ok"})
        return ToolResult.success({})

    gw = MagicMock()
    gw.call.side_effect = _call

    result = execute_local_commit("fix: something", ["missing.py"], gw)
    assert not result.ok
    assert add_called
    assert not commit_called


def test_execute_commit_trims_commit_message() -> None:
    """Leading/trailing whitespace in commit_message is stripped."""
    calls: list[str] = []
    gw = _ok_gateway(calls)
    execute_local_commit("  feat: trimmed  ", ["f.py"], gw)
    commit_cmd = next(c for c in calls if "git commit" in c)
    assert "feat: trimmed" in commit_cmd
    assert "  feat" not in commit_cmd


def test_execute_commit_stages_only_specified_files() -> None:
    """git add command includes exactly the specified files, not '.' or '-A'."""
    calls: list[str] = []
    gw = _ok_gateway(calls)
    execute_local_commit("chore: specific", ["core/a.py", "tests/test_a.py"], gw)
    add_cmd = next(c for c in calls if "git add" in c)
    assert "core/a.py" in add_cmd
    assert "tests/test_a.py" in add_cmd
    # must not stage everything
    assert " ." not in add_cmd
    assert " -A" not in add_cmd


# ── PR-23: Computer Mode is not a manual IDE invariants ──────────────────────


def _read_text(rel_path: str) -> str:
    """Read file relative to repo root."""
    import pathlib

    return pathlib.Path(rel_path).read_text(encoding="utf-8")


def test_arch_canon_computer_mode_not_ide() -> None:
    """ARCH_CANON.md must state Computer Mode is not a manual IDE."""
    text = _read_text("docs/architecture/ARCH_CANON.md")
    assert "not a manual IDE" in text
    assert "live agent execution surface" in text


def test_arch_canon_personal_agent_computer() -> None:
    """ARCH_CANON.md must describe Slavikai as personal agent computer."""
    text = _read_text("docs/architecture/ARCH_CANON.md")
    assert "personal agent computer" in text.lower()


def test_arch_canon_computer_primary_secondary_surfaces() -> None:
    """ARCH_CANON.md must list primary surface items and secondary details."""
    text = _read_text("docs/architecture/ARCH_CANON.md")
    assert "primary surface" in text.lower() or "Primary surface" in text
    assert "secondary" in text.lower()
    # Primary items
    assert "activity timeline" in text.lower() or "activity" in text.lower()
    assert "approvals" in text.lower()
    # Secondary items
    assert "editor" in text.lower()
    assert "terminal" in text.lower()


def test_arch_canon_future_backends_marked_not_implemented() -> None:
    """Future backends (browser, VM/Desktop) must be explicitly marked as not implemented."""
    text = _read_text("docs/architecture/ARCH_CANON.md")
    assert "Browser automation" in text or "browser automation" in text
    assert "VM / Desktop" in text or "VM/Desktop" in text
    assert "не реализован" in text or "not implemented" in text.lower()


def test_arch_canon_no_computer_send_endpoint_invariant() -> None:
    """ARCH_CANON.md must forbid /ui/api/computer/send."""
    text = _read_text("docs/architecture/ARCH_CANON.md")
    assert "/ui/api/computer/send" in text


def test_arch_canon_explorer_editor_not_primary_surface() -> None:
    """ARCH_CANON.md must state explorer/editor must not be primary product surface."""
    text = _read_text("docs/architecture/ARCH_CANON.md")
    assert "Explorer/editor" in text or "explorer/editor" in text.lower()
    assert "primary" in text.lower()


def test_computer_ui_has_no_workspace_send_reference() -> None:
    """Computer UI source files must not reference /ui/api/computer/send."""
    paths = [
        "ui/src/app/components/workspace-ide.tsx",
        "ui/src/app/components/workspace-session-screen.tsx",
        "ui/src/features/workspace/workspace-assistant-panel.tsx",
        "ui/src/features/workspace/workspace-toolbar.tsx",
    ]
    for path in paths:
        content = _read_text(path)
        assert "/ui/api/computer/send" not in content, (
            f"{path} must not reference /ui/api/computer/send"
        )
        assert "handleSendWorkspace" not in content, (
            f"{path} must not reference handleSendWorkspace"
        )


def test_computer_assistant_panel_no_auto_open_first_file() -> None:
    """workspace-ide.tsx must not auto-open the first file on tree load (IDE behavior)."""
    content = _read_text("ui/src/app/components/workspace-ide.tsx")
    assert "findFirstFilePath" not in content, (
        "workspace-ide.tsx must not auto-open first file from tree (IDE behavior removed)"
    )


def test_architecture_md_computer_mode_pivot_cross_reference() -> None:
    """Architecture.md must cross-reference Computer Mode product invariant."""
    text = _read_text("docs/architecture/Architecture.md")
    assert "not a manual IDE" in text or "Computer Mode" in text
    assert "ARCH_CANON" in text


# ── PR-24: tabbed Computer Mode surface ──────────────────────────────────────


def test_computer_tabs_overview_is_first() -> None:
    """workspace-ide.tsx must declare Overview as the first tab and default state."""
    content = _read_text("ui/src/app/components/workspace-ide.tsx")
    # Default tab state must be 'overview'
    assert "'overview'" in content or '"overview"' in content
    # Overview must appear before Files in tab order
    overview_pos = content.find("'overview'")
    files_pos = content.find("'files'")
    assert overview_pos != -1, "overview tab id not found"
    assert files_pos != -1, "files tab id not found"
    assert overview_pos < files_pos, "Overview must appear before Files in tab list"


def test_computer_tabs_all_canonical_tabs_present() -> None:
    """workspace-ide.tsx must define all 8 canonical Computer Mode tabs."""
    content = _read_text("ui/src/app/components/workspace-ide.tsx")
    required_tabs = [
        "overview",
        "activity",
        "terminal",
        "changes",
        "checks",
        "environment",
        "files",
        "logs",
    ]
    for tab_id in required_tabs:
        assert f"'{tab_id}'" in content or f'"{tab_id}"' in content, (
            f"Computer Mode tab '{tab_id}' not found in workspace-ide.tsx"
        )


def test_computer_tabs_files_tab_has_editor_pane() -> None:
    """workspace-ide.tsx Files tab must include WorkspaceEditorPane."""
    content = _read_text("ui/src/app/components/workspace-ide.tsx")
    assert "WorkspaceEditorPane" in content, "Files tab must render WorkspaceEditorPane"
    # WorkspaceEditorPane must appear inside the files tab conditional block
    files_idx = content.find("computerTab === 'files'")
    editor_after_files = content.find("WorkspaceEditorPane", files_idx)
    assert files_idx != -1, "files tab conditional not found"
    assert editor_after_files != -1, "WorkspaceEditorPane not found after files tab conditional"


def test_computer_tabs_overview_uses_assistant_panel() -> None:
    """workspace-ide.tsx Overview tab must render WorkspaceAssistantPanel."""
    content = _read_text("ui/src/app/components/workspace-ide.tsx")
    overview_idx = content.find("computerTab === 'overview'")
    panel_after_overview = content.find("WorkspaceAssistantPanel", overview_idx)
    assert overview_idx != -1, "overview tab conditional not found"
    assert panel_after_overview != -1, "WorkspaceAssistantPanel not found in overview tab"


def test_computer_tabs_no_new_lane() -> None:
    """Tab system must not introduce lane='computer' or role='computer'."""
    content = _read_text("ui/src/app/components/workspace-ide.tsx")
    assert "lane='computer'" not in content
    assert 'lane="computer"' not in content
    assert "role='computer'" not in content
    assert 'role="computer"' not in content


def test_computer_tabs_no_computer_send_endpoint() -> None:
    """Tab system must not introduce /ui/api/computer/send endpoint."""
    content = _read_text("ui/src/app/components/workspace-ide.tsx")
    assert "/ui/api/computer/send" not in content


def test_computer_tabs_files_tab_uses_files_tab_columns() -> None:
    """Files tab grid must use filesTabColumns (not old workspaceGridColumns)."""
    ide_content = _read_text("ui/src/app/components/workspace-ide.tsx")
    layout_content = _read_text("ui/src/features/workspace/use-workspace-layout.ts")
    # filesTabColumns must be exported from the hook
    assert "filesTabColumns" in layout_content, "useWorkspaceLayout must export filesTabColumns"
    # workspace-ide.tsx must use filesTabColumns
    assert "filesTabColumns" in ide_content, "workspace-ide.tsx must use filesTabColumns"
    # old workspaceGridColumns must no longer be exported from layout hook
    assert "workspaceGridColumns" not in layout_content, (
        "workspaceGridColumns must be removed from useWorkspaceLayout (replaced by filesTabColumns)"
    )


def test_computer_tab_data_attributes_present() -> None:
    """Tab buttons and content areas must carry data-computer-tab attributes for testability."""
    content = _read_text("ui/src/app/components/workspace-ide.tsx")
    assert "data-computer-tab=" in content, "Tab buttons must carry data-computer-tab attribute"
    assert "data-computer-tab-content=" in content, (
        "Tab content areas must carry data-computer-tab-content attribute"
    )
