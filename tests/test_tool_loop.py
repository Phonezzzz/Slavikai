from __future__ import annotations

from copy import deepcopy

import pytest

from core.approval_policy import ApprovalContext
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.stream_model import Done, Error, ToolCallCompleted
from llm.types import LLMResult, ToolCall, ToolSpec
from shared.models import LLMMessage, ToolResult
from tools.tool_registry import ToolRegistry


class LoopBrain:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_tool_specs: list[ToolSpec] = []

    def generate(self, messages, config=None, tools=None):  # type: ignore[override]
        del config
        self.calls += 1
        self.seen_tool_specs = list(tools or [])
        if self.calls == 1:
            return LLMResult(
                text="calling",
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"value": messages[-1].content})
                ],
            )
        assert messages[-1].role == "tool"
        return LLMResult(text=f"final:{messages[-1].content}")


def test_agent_tool_loop_executes_native_tool_call_and_appends_tool_message() -> None:
    registry = ToolRegistry()
    registry.register(
        "echo",
        lambda request: ToolResult.success({"output": request.args["value"]}),
        description="Echo a value",
        parameters_schema={"type": "object"},
    )
    brain = LoopBrain()

    result = AgentToolLoop().run(
        brain=brain,  # type: ignore[arg-type]
        gateway=ToolGateway(registry),
        messages=[LLMMessage(role="user", content="ping")],
        tools=registry.list_tool_specs(),
    )

    assert brain.calls == 2
    assert brain.seen_tool_specs == [
        ToolSpec(name="echo", description="Echo a value", parameters_schema={"type": "object"})
    ]
    assert result.iterations == 2
    assert result.tool_calls[0].call.name == "echo"
    assert result.tool_calls[0].result.ok
    assert result.messages[-2].role == "tool"
    assert result.text.startswith("final:")


def test_agent_tool_loop_stops_after_max_iterations() -> None:
    class AlwaysToolBrain:
        def generate(self, messages, config=None, tools=None):  # type: ignore[override]
            del messages, config, tools
            return LLMResult(
                text="again",
                tool_calls=[ToolCall(id="call-1", name="noop", arguments={})],
            )

    registry = ToolRegistry()
    registry.register("noop", lambda _request: ToolResult.success({}))

    result = AgentToolLoop(max_iterations=2).run(
        brain=AlwaysToolBrain(),  # type: ignore[arg-type]
        gateway=ToolGateway(registry),
        messages=[LLMMessage(role="user", content="go")],
        tools=registry.list_tool_specs(),
    )

    assert result.iterations == 2
    assert len(result.tool_calls) == 2


def test_agent_tool_loop_blocks_registered_tool_not_exposed_to_model() -> None:
    class HiddenToolBrain:
        def generate(self, messages, config=None, tools=None):  # type: ignore[override]
            del messages, config, tools
            return LLMResult(
                text="hidden",
                tool_calls=[ToolCall(id="hidden-1", name="runtime_cleanup", arguments={})],
            )

    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        "runtime_cleanup",
        lambda request: calls.append(request.name) or ToolResult.success({}),
        model_exposed=False,
    )

    result = AgentToolLoop(max_iterations=1).run(
        brain=HiddenToolBrain(),  # type: ignore[arg-type]
        gateway=ToolGateway(registry),
        messages=[LLMMessage(role="user", content="go")],
        tools=registry.list_tool_specs(),
    )

    assert calls == []
    assert result.tool_calls[0].result.meta["policy_reason"] == "model_tool_not_exposed"


def test_streaming_tool_loop_blocks_registered_tool_not_exposed_to_model() -> None:
    class HiddenStreamingBrain:
        def generate_stream_events(self, messages, config=None, tools=None):  # type: ignore[override]
            del messages, config, tools
            yield ToolCallCompleted(
                call=ToolCall(id="hidden-stream-1", name="runtime_cleanup", arguments={})
            )
            yield Done()

    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        "runtime_cleanup",
        lambda request: calls.append(request.name) or ToolResult.success({}),
        model_exposed=False,
    )

    events = list(
        AgentToolLoop(max_iterations=1).run_stream_events(
            brain=HiddenStreamingBrain(),  # type: ignore[arg-type]
            gateway=ToolGateway(registry),
            messages=[LLMMessage(role="user", content="go")],
            tools=registry.list_tool_specs(),
        )
    )

    completed = next(event for event in events if isinstance(event, ToolCallCompleted))
    assert calls == []
    assert completed.result is not None
    assert completed.result.meta["policy_reason"] == "model_tool_not_exposed"


def test_policy_denial_stops_remaining_calls_in_same_model_response() -> None:
    class BatchBrain:
        def generate(self, messages, config=None, tools=None):  # type: ignore[override]
            del messages, config, tools
            return LLMResult(
                text="try both",
                tool_calls=[
                    ToolCall(id="denied", name="shell", arguments={"command": "sudo reboot"}),
                    ToolCall(id="later", name="write", arguments={}),
                ],
            )

    calls: list[str] = []
    registry = ToolRegistry()
    registry.register("shell", lambda _request: calls.append("shell") or ToolResult.success({}))
    registry.register("write", lambda _request: calls.append("write") or ToolResult.success({}))

    result = AgentToolLoop(max_iterations=3).run(
        brain=BatchBrain(),  # type: ignore[arg-type]
        gateway=ToolGateway(
            registry,
            approval_context=ApprovalContext(
                safe_mode=False, session_id="test-session", approved_categories=set()
            ),
        ),
        messages=[LLMMessage(role="user", content="go")],
        tools=registry.list_tool_specs(),
    )

    assert result.error == "tool_policy_denied"
    assert [item.call.id for item in result.tool_calls] == ["denied"]
    assert result.tool_calls[0].result.meta["policy_reason"] == "command_denied:hard_safety"
    assert calls == []


def test_streaming_policy_denial_stops_remaining_calls() -> None:
    class BatchBrain:
        def generate_stream_events(self, messages, config=None, tools=None):  # type: ignore[override]
            del messages, config, tools
            yield ToolCallCompleted(
                call=ToolCall(id="denied", name="shell", arguments={"command": "sudo reboot"})
            )
            yield ToolCallCompleted(call=ToolCall(id="later", name="write", arguments={}))
            yield Done()

    calls: list[str] = []
    registry = ToolRegistry()
    registry.register("shell", lambda _request: calls.append("shell") or ToolResult.success({}))
    registry.register("write", lambda _request: calls.append("write") or ToolResult.success({}))

    events = list(
        AgentToolLoop(max_iterations=3).run_stream_events(
            brain=BatchBrain(),  # type: ignore[arg-type]
            gateway=ToolGateway(
                registry,
                approval_context=ApprovalContext(
                    safe_mode=False, session_id="test-session", approved_categories=set()
                ),
            ),
            messages=[LLMMessage(role="user", content="go")],
            tools=registry.list_tool_specs(),
        )
    )

    assert calls == []
    completed = [event for event in events if isinstance(event, ToolCallCompleted)]
    assert [event.call.id for event in completed] == ["denied"]
    assert completed[0].result is not None
    assert completed[0].result.meta["policy_reason"] == "command_denied:hard_safety"
    assert any(isinstance(event, Error) and event.code == "tool_policy_denied" for event in events)
    assert isinstance(events[-1], Done) and events[-1].finish_reason == "error"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("second_value", ["shown", "other"])
def test_approval_continuation_preserves_call_and_remaining_batch(streaming, second_value) -> None:
    import pytest

    from core.approval_policy import ApprovalRequired

    class BatchBrain:
        calls = 0

        def generate_stream_events(self, messages, config=None, tools=None):
            result = self.generate(messages, config, tools)
            for call in result.tool_calls:
                yield ToolCallCompleted(call=call)
            yield Done()

        def generate(self, messages, config=None, tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResult(
                    text="",
                    tool_calls=[
                        ToolCall(id="original", name="lookup", arguments={"value": "shown"}),
                        ToolCall(id="second", name="lookup", arguments={"value": second_value}),
                    ],
                )
            raise AssertionError("Approval must dispatch saved call before regeneration")

    executions = []
    registry = ToolRegistry()
    registry.register(
        "lookup",
        lambda request: executions.append(request) or ToolResult.success({"output": "ok"}),
        capability="read",
        risk_classes=["network"],
    )
    gateway = ToolGateway(
        registry,
        approval_context=ApprovalContext(
            safe_mode=True, session_id="session", approved_categories=set()
        ),
    )
    brain = BatchBrain()
    loop = AgentToolLoop()
    with pytest.raises(ApprovalRequired) as initial:
        if streaming:
            list(
                loop.run_stream_events(
                    brain=brain,
                    gateway=gateway,
                    messages=[LLMMessage(role="user", content="lookup")],
                    tools=registry.list_tool_specs(),
                )
            )
        else:
            loop.run(
                brain=brain,
                gateway=gateway,
                messages=[LLMMessage(role="user", content="lookup")],
                tools=registry.list_tool_specs(),
            )
    assert executions == []
    continuation = initial.value.continuation
    assert continuation is not None
    tampered = deepcopy(continuation)
    tampered.pending_calls[0].arguments["value"] = "changed"
    with pytest.raises(ValueError, match="approval_subject_mismatch"):
        loop.run(brain=brain, gateway=gateway, messages=[], tools=[], continuation=tampered)
    assert executions == []
    with pytest.raises(ApprovalRequired) as second:
        loop.run(
            brain=brain,
            gateway=gateway,
            messages=[],
            tools=registry.list_tool_specs(),
            continuation=continuation,
        )
    assert brain.calls == 1
    assert [request.args for request in executions] == [{"value": "shown"}]
    assert second.value.continuation is not None
    assert second.value.continuation.executed[0].call.id == "original"
    assert second.value.continuation.history[-1].tool_call_id == "original"


def test_continuation_retains_completed_tool_when_provider_raises() -> None:
    from core.approval_policy import ApprovalRequired
    from core.tool_loop import ToolLoopExecutionError

    class FailingBrain:
        calls = 0

        def generate(self, messages, config=None, tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResult(
                    text="", tool_calls=[ToolCall(id="original", name="lookup", arguments={})]
                )
            raise RuntimeError("provider crashed after dispatch")

    canonical = ToolResult.success({"output": "complete diagnostic"})
    executions = []
    registry = ToolRegistry()
    registry.register(
        "lookup",
        lambda request: executions.append(request) or canonical,
        capability="read",
        risk_classes=["network"],
    )
    gateway = ToolGateway(
        registry,
        approval_context=ApprovalContext(
            safe_mode=True, session_id="session", approved_categories=set()
        ),
    )
    brain = FailingBrain()
    with pytest.raises(ApprovalRequired) as initial:
        AgentToolLoop().run(
            brain=brain, gateway=gateway, messages=[], tools=registry.list_tool_specs()
        )
    with pytest.raises(ToolLoopExecutionError) as failed:
        AgentToolLoop().run(
            brain=brain,
            gateway=gateway,
            messages=[],
            tools=[],
            continuation=initial.value.continuation,
        )
    assert len(executions) == 1
    assert failed.value.result.tool_calls[0].result is canonical
    assert failed.value.result.messages[-1].tool_call_id == "original"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tool_ok", [False, True])
def test_cancel_during_dispatch_retains_completed_observation(streaming, tool_ok):
    import asyncio
    import json

    token = asyncio.Event()
    executions = []
    observed = ToolResult(
        ok=tool_ok,
        data={"output": "full diagnostics\nsecond line", "stderr": "warning"},
        error=None if tool_ok else "command failed",
        meta={"exit_code": 0 if tool_ok else 7},
    )

    class BatchBrain:
        calls = 0

        def generate(self, messages, config=None, tools=None):
            self.calls += 1
            assert self.calls == 1
            return LLMResult(
                text="",
                tool_calls=[
                    ToolCall(id="first", name="lookup", arguments={"index": 1}),
                    ToolCall(id="second", name="lookup", arguments={"index": 2}),
                ],
            )

        def generate_stream_events(
            self, messages, config=None, tools=None, cancellation_token=None
        ):
            for call in self.generate(messages, config, tools).tool_calls:
                yield ToolCallCompleted(call=call)
            yield Done()

    def execute(request):
        executions.append(request)
        token.set()
        return observed

    registry = ToolRegistry()
    registry.register("lookup", execute, capability="read")
    brain = BatchBrain()
    loop = AgentToolLoop()
    args = dict(
        brain=brain,
        gateway=ToolGateway(registry),
        messages=[LLMMessage(role="user", content="lookup")],
        tools=registry.list_tool_specs(),
        cancellation_token=token,
    )
    events = []
    if streaming:
        iterator = loop.run_stream_events(**args)
        while True:
            try:
                events.append(next(iterator))
            except StopIteration as stopped:
                result = stopped.value
                break
        completed = [event for event in events if isinstance(event, ToolCallCompleted)]
        assert len(completed) == 1 and completed[0].result is observed
        assert events[-1].finish_reason == "cancelled"
    else:
        result = loop.run(**args)
    assert result.cancelled is True
    assert brain.calls == 1
    assert len(executions) == 1 and executions[0].args == {"index": 1}
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].call.id == "first"
    assert result.tool_calls[0].result is observed
    assert result.tool_calls[0].result.ok is tool_ok
    assert result.messages[-1].role == "tool"
    assert result.messages[-1].tool_call_id == "first"
    serialized = json.loads(result.messages[-1].content)
    assert serialized["data"] == observed.data
    assert serialized["meta"] == observed.meta
    assert serialized["ok"] is tool_ok


@pytest.mark.parametrize("stage", ["creation", "iteration", "cancelled"])
@pytest.mark.parametrize("tool_ok", [False, True])
def test_stream_provider_exception_retains_previous_dispatch(stage, tool_ok):
    from core.tool_loop import ToolLoopExecutionError
    from llm.cancellation import GenerationCancelled

    observed = ToolResult(
        ok=tool_ok,
        data={"output": "original diagnostics", "stderr": "warning"},
        error=None if tool_ok else "execution failed",
        meta={"exit_code": 0 if tool_ok else 8},
    )
    executions = []
    failure = GenerationCancelled() if stage == "cancelled" else RuntimeError("provider offline")

    class Provider:
        calls = 0

        def generate_stream_events(self, messages, config=None, tools=None):
            self.calls += 1
            if self.calls == 1:
                return iter(
                    [
                        ToolCallCompleted(
                            call=ToolCall(id="original", name="lookup", arguments={})
                        ),
                        Done(),
                    ]
                )
            if stage == "creation":
                raise failure

            def interrupted():
                yield from ()
                raise failure

            return interrupted()

    registry = ToolRegistry()
    registry.register("lookup", lambda request: executions.append(request) or observed)
    provider = Provider()
    iterator = AgentToolLoop().run_stream_events(
        brain=provider,
        gateway=ToolGateway(registry),
        messages=[LLMMessage(role="user", content="lookup")],
        tools=registry.list_tool_specs(),
    )
    if stage == "cancelled":
        events = []
        while True:
            try:
                events.append(next(iterator))
            except StopIteration as stopped:
                result = stopped.value
                break
        assert events[-1].finish_reason == "cancelled"
        assert result.cancelled is True
    else:
        with pytest.raises(ToolLoopExecutionError) as caught:
            list(iterator)
        assert caught.value.cause is failure
        result = caught.value.result
        assert result.error == "provider_generation_failed"
    assert provider.calls == 2 and len(executions) == 1
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].result is observed
    assert result.tool_calls[0].call.id == "original"
    assert result.messages[-1].tool_call_id == "original"
