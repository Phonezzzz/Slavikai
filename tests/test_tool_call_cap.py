from dataclasses import replace

import pytest

from core.approval_policy import ApprovalContext, ApprovalRequired
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop, ToolLoopExecutionError
from llm.brain_base import Brain
from llm.stream_model import Done, ToolCallCompleted
from llm.types import LLMResult, LLMUsage, ToolCall, ToolSpec
from shared.models import LLMMessage, ToolResult
from tools.tool_logger import ToolCallLogger
from tools.tool_registry import ToolRegistry


class BatchBrain(Brain):
    supports_native_tools = True

    def __init__(self, batches):
        self.batches = iter(batches)

    def generate(self, messages, config=None, tools=None):
        return LLMResult(usage=LLMUsage(0, 0, 0), text="answer", tool_calls=next(self.batches, []))

    def generate_stream_events(self, messages, config=None, tools=None):
        for call in self.generate(messages, config, tools).tool_calls:
            yield ToolCallCompleted(call=call)
        yield Done()


def calls(*names):
    return [ToolCall(id=f"{name}-{i}", name=name, arguments={}) for i, name in enumerate(names)]


def setup(tmp_path):
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "calls.jsonl"))
    dispatched = []
    for name, risks in [("ok", []), ("fail", []), ("network", ["network"])]:
        registry.register(
            name,
            lambda request: dispatched.append(request.name)
            or (ToolResult.failure("failed") if request.name == "fail" else ToolResult.success({})),
            capability="read",
            risk_classes=risks,
        )
    return registry, dispatched


def run(loop, brain, gateway, streaming=False, tools=None):
    kwargs = dict(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage(role="user", content="go")],
        tools=gateway.registry.list_tool_specs() if tools is None else tools,
    )
    if not streaming:
        return loop.run(**kwargs)
    events = loop.run_stream_events(**kwargs)
    observed = []
    while True:
        try:
            observed.append(next(events))
        except StopIteration as done:
            return done.value


@pytest.mark.parametrize("streaming", [False, True])
def test_cap_counts_failures_across_batches_and_keeps_prefix(tmp_path, streaming):
    registry, dispatched = setup(tmp_path)
    result = run(
        AgentToolLoop(max_tool_calls=2),
        BatchBrain([calls("fail"), calls("ok", "network")]),
        ToolGateway(registry, approval_context=ApprovalContext(True, "s", set())),
        streaming,
    )
    assert dispatched == ["fail", "ok"]
    assert result.error == "tool_call_budget_exhausted"
    assert [c.call.name for c in result.tool_calls] == ["fail", "ok"]
    assert [m.tool_call_id for m in result.messages if m.role == "tool"] == ["fail-0", "ok-0"]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
def test_approval_resume_uses_captured_cap_even_in_new_uncapped_loop(tmp_path, streaming, prefix):
    registry, dispatched = setup(tmp_path)
    gateway = ToolGateway(registry, approval_context=ApprovalContext(True, "s", set()))
    batch = calls(*(["ok"] if prefix else []), "network", "ok", "network")
    brain = BatchBrain([batch])
    cap = 2 if prefix else 1
    with pytest.raises(ApprovalRequired) as stopped:
        run(AgentToolLoop(max_tool_calls=cap), brain, gateway, streaming)
    checkpoint = stopped.value.continuation
    assert len(checkpoint.executed) == int(prefix)
    assert checkpoint.max_tool_calls == cap
    with pytest.raises(ToolLoopExecutionError) as mismatch:
        AgentToolLoop().run(
            brain=brain,
            gateway=replace(gateway, approval_context=ApprovalContext(True, "foreign", set())),
            messages=[],
            tools=[],
            continuation=checkpoint,
        )
    assert isinstance(mismatch.value.cause, ValueError)
    assert str(mismatch.value.cause) == "approval_subject_mismatch"
    assert dispatched == (["ok"] if prefix else [])
    result = AgentToolLoop(max_tool_calls=100).run(
        brain=brain,
        gateway=gateway,
        messages=[],
        tools=[],
        continuation=checkpoint,
    )
    assert dispatched == (["ok"] if prefix else []) + ["network"]
    assert len(result.tool_calls) == cap
    assert result.error == "tool_call_budget_exhausted"


@pytest.mark.parametrize("streaming", [False, True])
def test_dispatch_observation_fault_retains_available_prefix_without_replay(tmp_path, streaming):
    registry, dispatched = setup(tmp_path)

    def observe(request, result, context):
        if request.name == "fail":
            raise RuntimeError("post effect observation fault")

    with pytest.raises(ToolLoopExecutionError) as failed:
        run(
            AgentToolLoop(max_tool_calls=2),
            BatchBrain([calls("ok", "fail", "ok")]),
            ToolGateway(registry, post_call=observe),
            streaming,
        )
    assert dispatched == ["ok", "fail"]
    assert [c.call.name for c in failed.value.result.tool_calls] == ["ok"]
    assert failed.value.result.error == "tool_dispatch_failed"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_cap_is_rejected_without_silent_clamp(value):
    with pytest.raises(ValueError, match="max_tool_calls"):
        AgentToolLoop(max_tool_calls=value)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("cap", [None, 2])
def test_exact_cap_can_finish_and_iteration_guard_is_separate(tmp_path, streaming, cap):
    registry, dispatched = setup(tmp_path)
    result = run(
        AgentToolLoop(max_tool_calls=cap),
        BatchBrain([calls("ok", "ok")]),
        ToolGateway(registry),
        streaming,
    )
    assert result.error is None and len(dispatched) == 2
    exhausted = run(
        AgentToolLoop(max_iterations=1, max_tool_calls=cap),
        BatchBrain([calls("ok")]),
        ToolGateway(registry),
        streaming,
    )
    assert exhausted.error is not None
    assert exhausted.error != "tool_call_budget_exhausted"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("hidden", [False, True])
def test_denied_or_unexposed_call_counts_without_handler_dispatch(tmp_path, streaming, hidden):
    registry, dispatched = setup(tmp_path)
    if hidden:
        batch = calls("unexposed", "ok")
        gateway = ToolGateway(registry)
    else:
        registry.register(
            "blocked",
            lambda request: dispatched.append("BAD") or ToolResult.success({}),
            enabled=False,
            capability="read",
        )
        batch = calls("blocked", "ok")
        gateway = ToolGateway(registry)
    result = run(
        AgentToolLoop(max_tool_calls=1),
        BatchBrain([batch]),
        gateway,
        streaming,
        None if hidden else [ToolSpec(name="blocked", description="blocked", parameters_schema={})],
    )
    assert not dispatched
    assert len(result.tool_calls) == 1 and not result.tool_calls[0].result.ok
    assert result.error == ("tool_policy_denied" if hidden else "tool_call_budget_exhausted")
