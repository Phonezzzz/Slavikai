"""End-to-end proof: Chat → AgentToolLoop → ToolGateway → ComputerBackend path.

Verifies that:
- The standard tool-loop path (LLM → tool_calls → gateway → registry) executes
  workspace tools without any workspace/send endpoint or lane="computer".
- ComputerActivityLog captures tool events separately from the visible chat body.
- AgentComputerRuntime (LocalComputerBackend) can be constructed and its operations
  flow through the same ToolGateway, producing results without a second conversational
  entrypoint.
- The loop's final text is the LLM response text, not raw internal tool JSON.

No Docker daemon needed. No FakeContainerRunner. Default backend = LocalComputerBackend.
"""

from __future__ import annotations

import json

import pytest

from core.agent_computer import AgentComputerRuntime
from core.computer_activity_log import ComputerActivityLog
from core.computer_backend import LocalComputerBackend
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig, ToolCall, ToolSpec
from shared.models import LLMMessage, ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_registry(*handlers: tuple[str, ToolRequest]) -> ToolRegistry:
    """Create a ToolRegistry with simple pass-through handlers."""
    registry = ToolRegistry()
    return registry


def _make_stub_registry(
    tool_name: str,
    return_result: ToolResult,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool_name, lambda req: return_result, enabled=True, capability="read")
    return registry


class _ScriptedBrain(Brain):
    """Brain that plays back a pre-recorded sequence of LLMResults."""

    def __init__(self, script: list[LLMResult]) -> None:
        self._script = list(script)
        self._calls: list[list[LLMMessage]] = []

    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        self._calls.append(list(messages))
        if not self._script:
            return LLMResult(text="(no more script)")
        return self._script.pop(0)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def file_content_result() -> ToolResult:
    return ToolResult.success({"content": "# Project README\nHello world."})


@pytest.fixture()
def registry(file_content_result: ToolResult) -> ToolRegistry:
    return _make_stub_registry("workspace_read", file_content_result)


@pytest.fixture()
def activity_log() -> ComputerActivityLog:
    return ComputerActivityLog()


@pytest.fixture()
def gateway(registry: ToolRegistry, activity_log: ComputerActivityLog) -> ToolGateway:
    return ToolGateway(
        registry=registry,
        pre_call=activity_log.pre_call,
        post_call=activity_log.post_call,
        approval_context=None,
    )


# ── Core path: AgentToolLoop → ToolGateway → workspace_read ──────────────────


def test_tool_loop_executes_workspace_read(
    gateway: ToolGateway,
    activity_log: ComputerActivityLog,
) -> None:
    """The loop calls workspace_read and returns the final LLM text."""
    tc = ToolCall(id="tc-1", name="workspace_read", arguments={"path": "README.md"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="The README contains: Hello world."),
        ]
    )

    loop = AgentToolLoop()
    result = loop.run(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="Show me README.md")],
        tools=[],
    )

    assert result.text == "The README contains: Hello world."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].call.name == "workspace_read"
    assert result.tool_calls[0].result.ok


def test_tool_loop_result_text_is_llm_response_not_raw_json(
    gateway: ToolGateway,
) -> None:
    """Visible result text is LLM prose, not raw tool JSON dump."""
    tc = ToolCall(id="tc-2", name="workspace_read", arguments={"path": "x.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="File has been read successfully."),
        ]
    )

    result = AgentToolLoop().run(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="read x.py")],
        tools=[],
    )

    assert "File has been read successfully." == result.text
    # final text must not start with raw JSON ({"ok": ...})
    assert not result.text.startswith("{")


def test_tool_result_appears_in_message_history(
    gateway: ToolGateway,
) -> None:
    """Tool result is injected as role='tool' message in history."""
    tc = ToolCall(id="tc-3", name="workspace_read", arguments={"path": "a.txt"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="Got it."),
        ]
    )

    result = AgentToolLoop().run(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="read a.txt")],
        tools=[],
    )

    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "tc-3"
    # content is serialized JSON, must be parseable
    payload = json.loads(tool_messages[0].content or "")
    assert payload["ok"] is True


def test_no_lane_computer_in_messages(
    gateway: ToolGateway,
) -> None:
    """No message carries lane='computer' — Computer is not a chat lane."""
    tc = ToolCall(id="tc-4", name="workspace_read", arguments={"path": "b.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="Done."),
        ]
    )

    result = AgentToolLoop().run(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="read b.py")],
        tools=[],
    )

    for msg in result.messages:
        content = msg.content or ""
        assert "lane" not in content or "computer" not in content, (
            f"Forbidden lane='computer' found in message: {msg}"
        )
        assert msg.role != "computer"


# ── ComputerActivityLog captures events separately ───────────────────────────


def test_activity_log_captures_tool_started_event(
    gateway: ToolGateway,
    activity_log: ComputerActivityLog,
) -> None:
    tc = ToolCall(id="tc-5", name="workspace_read", arguments={"path": "c.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="Done."),
        ]
    )

    AgentToolLoop().run(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="read c.py")],
        tools=[],
    )

    events = activity_log.drain()
    assert len(events) >= 2  # pre_call (tool_started) + post_call (file_read)
    kinds = [e["kind"] for e in events]
    assert "tool_started" in kinds


def test_activity_log_captures_file_read_event(
    gateway: ToolGateway,
    activity_log: ComputerActivityLog,
) -> None:
    tc = ToolCall(id="tc-6", name="workspace_read", arguments={"path": "d.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="Done."),
        ]
    )

    AgentToolLoop().run(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="read d.py")],
        tools=[],
    )

    events = activity_log.drain()
    post_events = [e for e in events if e["kind"] != "tool_started"]
    assert len(post_events) >= 1
    assert post_events[0]["kind"] == "file_read"
    assert post_events[0]["tool"] == "workspace_read"
    assert post_events[0]["ok"] is True


def test_activity_log_includes_path(
    gateway: ToolGateway,
    activity_log: ComputerActivityLog,
) -> None:
    tc = ToolCall(id="tc-7", name="workspace_read", arguments={"path": "src/main.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="Done."),
        ]
    )

    AgentToolLoop().run(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="read src/main.py")],
        tools=[],
    )

    events = activity_log.drain()
    post_events = [e for e in events if e.get("path")]
    assert any(e["path"] == "src/main.py" for e in post_events)


def test_activity_log_drain_clears_events(
    gateway: ToolGateway,
    activity_log: ComputerActivityLog,
) -> None:
    """Drain is destructive — second drain returns empty list."""
    tc = ToolCall(id="tc-8", name="workspace_read", arguments={"path": "e.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="Done."),
        ]
    )

    AgentToolLoop().run(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="x")],
        tools=[],
    )

    activity_log.drain()
    assert activity_log.drain() == []


# ── AgentComputerRuntime wired via same ToolGateway ──────────────────────────


def test_agent_computer_runtime_can_be_constructed_with_local_backend(
    gateway: ToolGateway,
    registry: ToolRegistry,
) -> None:
    """AgentComputerRuntime with LocalComputerBackend is constructible without Docker."""
    backend = LocalComputerBackend(gateway=gateway)
    runtime = AgentComputerRuntime(backend=backend)
    assert runtime.backend is backend
    assert runtime.backend.gateway is gateway


def test_agent_computer_runtime_read_file_calls_workspace_read(
    file_content_result: ToolResult,
    registry: ToolRegistry,
    activity_log: ComputerActivityLog,
) -> None:
    """AgentComputerRuntime.read_file() routes to workspace_read via gateway."""
    gateway = ToolGateway(
        registry=registry,
        pre_call=activity_log.pre_call,
        post_call=activity_log.post_call,
        approval_context=None,
    )
    backend = LocalComputerBackend(gateway=gateway)
    runtime = AgentComputerRuntime(backend=backend)

    result = runtime.read_file("README.md")

    assert result.ok
    assert result.data == {"content": "# Project README\nHello world."}

    events = activity_log.drain()
    assert any(e["tool"] == "workspace_read" for e in events)


def test_agent_computer_runtime_activity_is_separate_from_chat_body(
    gateway: ToolGateway,
    activity_log: ComputerActivityLog,
    registry: ToolRegistry,
) -> None:
    """Computer activity events are NOT injected into visible chat messages."""
    tc = ToolCall(id="tc-9", name="workspace_read", arguments={"path": "f.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="Here is the file."),
        ]
    )

    loop_result = AgentToolLoop().run(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="read f.py")],
        tools=[],
    )

    # Drain computer events separately — they must not appear as chat messages
    computer_events = activity_log.drain()
    assert len(computer_events) >= 1

    # Chat messages contain only: user, assistant, tool role messages
    chat_roles = {m.role for m in loop_result.messages}
    assert "computer" not in chat_roles
    assert chat_roles <= {"user", "assistant", "tool"}


# ── Tool failure path ─────────────────────────────────────────────────────────


def test_tool_failure_propagates_through_loop() -> None:
    """A failing workspace_read result is carried through the loop (no crash)."""
    fail_result = ToolResult.failure("file not found")
    registry = _make_stub_registry("workspace_read", fail_result)
    log = ComputerActivityLog()
    gw = ToolGateway(
        registry=registry,
        pre_call=log.pre_call,
        post_call=log.post_call,
        approval_context=None,
    )

    tc = ToolCall(id="tc-10", name="workspace_read", arguments={"path": "missing.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc]),
            LLMResult(text="Sorry, file was not found."),
        ]
    )

    result = AgentToolLoop().run(
        brain=brain,
        gateway=gw,
        messages=[LLMMessage(role="user", content="read missing.py")],
        tools=[],
    )

    assert result.text == "Sorry, file was not found."
    assert len(result.tool_calls) == 1
    assert not result.tool_calls[0].result.ok
    assert result.tool_calls[0].result.error == "file not found"

    events = log.drain()
    failure_events = [e for e in events if e.get("ok") is False]
    assert len(failure_events) >= 1


# ── Multi-tool call in single iteration ──────────────────────────────────────


def test_multiple_tool_calls_in_one_iteration() -> None:
    """Loop handles multiple tool_calls from a single LLM response."""
    read_result = ToolResult.success({"content": "hello"})
    registry = ToolRegistry()
    registry.register("workspace_read", lambda req: read_result, enabled=True, capability="read")

    log = ComputerActivityLog()
    gw = ToolGateway(
        registry=registry,
        pre_call=log.pre_call,
        post_call=log.post_call,
        approval_context=None,
    )

    tc1 = ToolCall(id="tc-11a", name="workspace_read", arguments={"path": "a.py"})
    tc2 = ToolCall(id="tc-11b", name="workspace_read", arguments={"path": "b.py"})
    brain = _ScriptedBrain(
        [
            LLMResult(text="", tool_calls=[tc1, tc2]),
            LLMResult(text="Both files read."),
        ]
    )

    result = AgentToolLoop().run(
        brain=brain,
        gateway=gw,
        messages=[LLMMessage(role="user", content="read both")],
        tools=[],
    )

    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].call.id == "tc-11a"
    assert result.tool_calls[1].call.id == "tc-11b"
    assert result.text == "Both files read."

    # Two tool calls → at least 4 events (pre + post for each)
    events = log.drain()
    assert len(events) >= 4
