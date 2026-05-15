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

from core.computer_activity_log import ComputerActivityLog
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

    def generate_stream(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> None:
        raise NotImplementedError


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
        tools=[],
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
            tools=[],
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
            tools=[],
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
            tools=[],
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


# ── LocalComputerBackend wires through ToolGateway, not direct file access ───


def test_local_backend_read_file_uses_tool_gateway_not_direct_io() -> None:
    """LocalComputerBackend.read_file() calls ToolGateway, not direct Python file I/O."""
    gateway_mock = MagicMock()
    gateway_mock.call.return_value = ToolResult.success({"content": "hello"})
    backend = LocalComputerBackend(gateway=gateway_mock)
    result = backend.read_file("README.md")
    assert result.ok
    gateway_mock.call.assert_called_once_with(ToolRequest("workspace_read", {"path": "README.md"}))
