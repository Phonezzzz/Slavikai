import pytest

from core.approval_policy import ApprovalContext, ApprovalRequired
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.brain_base import Brain
from llm.cancellation import GenerationCancelled
from llm.types import LLMResult, LLMUsage
from tests.test_tool_call_cap import calls, setup


class UsageBrain(Brain):
    supports_native_tools = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = 0

    def generate(self, messages, config=None, tools=None):
        self.requests += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def execute(loop, brain, gateway, **kwargs):
    return loop.run(
        brain=brain,
        gateway=gateway,
        messages=[],
        tools=gateway.registry.list_tool_specs(),
        **kwargs,
    )


def test_two_model_responses_sum_reported_counts(tmp_path):
    registry, dispatched = setup(tmp_path)
    brain = UsageBrain(
        [
            LLMResult(text="tool", tool_calls=calls("ok"), usage=LLMUsage(2, 3, 99)),
            LLMResult(text="done", usage=LLMUsage(4, 5, 9)),
        ]
    )
    result = execute(AgentToolLoop(), brain, ToolGateway(registry), record_model_usage=True)
    usage = result.model_usage
    assert (usage.started_calls, usage.reported_calls, usage.unknown_calls) == (2, 2, 0)
    assert usage.reported_sum == LLMUsage(6, 8, 108)
    assert usage.total_tokens == 108
    assert brain.requests == 2 and dispatched == ["ok"]


def test_approval_resume_counts_original_response_once(tmp_path):
    registry, dispatched = setup(tmp_path)
    brain = UsageBrain(
        [
            LLMResult(text="approval", tool_calls=calls("network"), usage=LLMUsage(2, 3, 5)),
            LLMResult(text="done", usage=LLMUsage(0, 0, 0)),
        ]
    )
    gateway = ToolGateway(registry, approval_context=ApprovalContext(True, "s", set()))
    with pytest.raises(ApprovalRequired) as paused:
        execute(AgentToolLoop(), brain, gateway, record_model_usage=True)
    checkpoint = paused.value.continuation
    assert checkpoint.model_usage.started_calls == 1
    result = execute(AgentToolLoop(), brain, gateway, continuation=checkpoint)
    assert result.model_usage.started_calls == 2
    assert result.model_usage.reported_sum == LLMUsage(2, 3, 5)
    assert brain.requests == 2 and dispatched == ["network"]


def test_auto_publishes_usage_on_terminal_and_paused_paths(tmp_path, monkeypatch):
    from core.auto_runtime import AutoOrchestrator
    from tests.test_auto_runtime import _AutoV1Agent

    monkeypatch.chdir(tmp_path)
    agent = _AutoV1Agent()
    agent._brain = UsageBrain([LLMResult(text="done", usage=LLMUsage(1, 2, 3))])
    outcome = AutoOrchestrator(agent, workspace_root=tmp_path).run_v1("go")
    assert outcome.verifier.ok
    usage = agent.last_auto_state["execution_metrics"]["tool_loop_model_usage"]
    assert usage == {
        "started_calls": 1,
        "reported_calls": 1,
        "unknown_calls": 0,
        "reported_sum": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "total_tokens": 3,
    }


@pytest.mark.parametrize("usage", [None, LLMUsage(0, 0, 0)])
def test_unknown_is_distinct_from_reported_zero(tmp_path, usage):
    registry, _ = setup(tmp_path)
    result = execute(
        AgentToolLoop(),
        UsageBrain([LLMResult(text="done", usage=usage)]),
        ToolGateway(registry),
        record_model_usage=True,
    )
    observed = result.model_usage
    assert observed.started_calls == 1
    assert observed.unknown_calls == int(usage is None)
    assert observed.reported_calls == int(usage is not None)
    assert observed.total_tokens == (None if usage is None else 0)


@pytest.mark.parametrize("failure", [RuntimeError("fixture failure"), GenerationCancelled()])
def test_failed_generation_retains_prior_usage_and_unknown(tmp_path, failure):
    from core.tool_loop import ToolLoopExecutionError
    from llm.cancellation import GenerationCancelled

    registry, dispatched = setup(tmp_path)
    brain = UsageBrain(
        [LLMResult(text="tool", tool_calls=calls("ok"), usage=LLMUsage(1, 2, 3)), failure]
    )
    if isinstance(failure, GenerationCancelled):
        result = execute(AgentToolLoop(), brain, ToolGateway(registry), record_model_usage=True)
        assert result.cancelled
    else:
        with pytest.raises(ToolLoopExecutionError) as failed:
            execute(AgentToolLoop(), brain, ToolGateway(registry), record_model_usage=True)
        result = failed.value.result
    observed = result.model_usage
    assert (observed.started_calls, observed.reported_calls, observed.unknown_calls) == (2, 1, 1)
    assert observed.reported_sum == LLMUsage(1, 2, 3) and observed.total_tokens is None
    assert dispatched == ["ok"]


@pytest.mark.parametrize("after_model", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_stop_boundary_retains_received_usage(tmp_path, monkeypatch, after_model, cancel):
    import asyncio
    from types import SimpleNamespace

    import core.tool_loop as tool_loop

    token = asyncio.Event()
    timer = SimpleNamespace(monotonic=lambda: 10.0)
    monkeypatch.setattr(tool_loop, "time", timer)
    registry, dispatched = setup(tmp_path)

    class StoppingBrain(UsageBrain):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            if cancel:
                token.set()
            else:
                timer.monotonic = lambda: 11.0
            return result

    if not after_model and cancel:
        token.set()
    brain = StoppingBrain([LLMResult(text="done", usage=LLMUsage(2, 3, 5))])
    result = execute(
        AgentToolLoop(expires_at_monotonic=11.0 if after_model else 10.0),
        brain,
        ToolGateway(registry),
        record_model_usage=True,
        cancellation_token=token,
    )
    assert brain.requests == int(after_model)
    assert result.model_usage.started_calls == int(after_model)
    assert result.model_usage.total_tokens == (5 if after_model else None)
    assert not dispatched


@pytest.mark.parametrize("tracking", [False, True])
def test_non_auto_opt_out_and_stream_are_not_aggregated(tmp_path, tracking):
    registry, _ = setup(tmp_path)
    brain = UsageBrain([LLMResult(text="done", usage=LLMUsage(1, 2, 3))])
    result = execute(AgentToolLoop(), brain, ToolGateway(registry), record_model_usage=tracking)
    assert (result.model_usage is not None) == tracking
    stream = AgentToolLoop().run_stream_events(
        brain=UsageBrain([LLMResult(text="done", usage=LLMUsage(1, 2, 3))]),
        gateway=ToolGateway(registry),
        messages=[],
        tools=[],
    )
    while True:
        try:
            next(stream)
        except StopIteration as end:
            assert end.value.model_usage is None
            break


def test_repeated_approval_checkpoints_keep_independent_frozen_summary(tmp_path):
    registry, dispatched = setup(tmp_path)
    gateway = ToolGateway(registry, approval_context=ApprovalContext(True, "s", set()))
    brain = UsageBrain(
        [
            LLMResult(
                text="batch", tool_calls=calls("network", "network"), usage=LLMUsage(2, 3, 5)
            ),
            LLMResult(text="done", usage=None),
        ]
    )
    with pytest.raises(ApprovalRequired) as first:
        execute(AgentToolLoop(), brain, gateway, record_model_usage=True)
    checkpoint = first.value.continuation
    with pytest.raises(ApprovalRequired) as second:
        execute(AgentToolLoop(), brain, gateway, continuation=checkpoint)
    assert second.value.continuation.model_usage.started_calls == 1
    result = execute(AgentToolLoop(), brain, gateway, continuation=second.value.continuation)
    assert checkpoint.model_usage.started_calls == 1
    assert result.model_usage.started_calls == 2 and result.model_usage.unknown_calls == 1
    assert dispatched == ["network", "network"] and brain.requests == 2


def test_auto_projection_survives_pause_cancel_and_independent_principals(tmp_path, monkeypatch):
    from core.auto_runtime import AutoOrchestrator
    from shared.models import ToolResult
    from tests.test_auto_runtime import _AutoV1Agent

    monkeypatch.chdir(tmp_path)
    runs = []
    for principal, count in [("one", 3), ("two", 8)]:
        agent = _AutoV1Agent()
        agent.user_id = agent.session_id = principal
        agent._brain = UsageBrain(
            [
                LLMResult(
                    text="batch", tool_calls=calls("network"), usage=LLMUsage(count, 0, count)
                ),
                LLMResult(text="done", usage=None),
            ]
        )
        agent.tool_registry.register(
            "network",
            lambda request: ToolResult.success({}),
            capability="read",
            risk_classes=["network"],
        )
        gateway = ToolGateway(
            agent.tool_registry, approval_context=ApprovalContext(True, principal, set())
        )
        agent._build_tool_gateway = lambda current=gateway: current
        orchestrator = AutoOrchestrator(agent, workspace_root=tmp_path)
        with pytest.raises(ApprovalRequired):
            orchestrator.run_v1("go")
        usage = agent.last_auto_state["execution_metrics"]["tool_loop_model_usage"]
        assert usage["total_tokens"] == count
        runs.append((agent, orchestrator, agent.last_auto_state["run_id"], count))
    agent, orchestrator, run_id, count = runs[0]
    assert (
        orchestrator.cancel(run_id)["execution_metrics"]["tool_loop_model_usage"]["total_tokens"]
        == count
    )
    agent, orchestrator, run_id, count = runs[1]
    orchestrator.resume(run_id)
    usage = agent.last_auto_state["execution_metrics"]["tool_loop_model_usage"]
    assert usage["started_calls"] == 2 and usage["unknown_calls"] == 1
    assert usage["total_tokens"] is None and usage["reported_sum"]["total_tokens"] == count


@pytest.mark.parametrize("kind", ["exception", "dispatch", "cap", "expiry"])
def test_auto_failure_projection_keeps_observed_usage(tmp_path, monkeypatch, kind):
    from types import SimpleNamespace

    import core.auto_runtime as auto_runtime
    import core.tool_loop as tool_loop
    from shared.models import ToolResult
    from tests.test_auto_runtime import _AutoV1Agent

    monkeypatch.chdir(tmp_path)
    timer = SimpleNamespace(monotonic=lambda: 10.0)
    monkeypatch.setattr(auto_runtime, "time", timer)
    monkeypatch.setattr(tool_loop, "time", timer)
    monkeypatch.setenv(auto_runtime.AUTO_MAX_RUNTIME_SECONDS_ENV, "1")
    monkeypatch.setenv(auto_runtime.AUTO_MAX_TOOL_CALLS_ENV, "1" if kind == "cap" else "3")
    agent = _AutoV1Agent()

    class FaultBrain(UsageBrain):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            if kind == "expiry":
                timer.monotonic = lambda: 11.0
            return result

    agent._brain = FaultBrain(
        [
            LLMResult(
                text="tool",
                tool_calls=calls("ok", "ok") if kind == "cap" else calls("ok"),
                usage=LLMUsage(1, 2, 3),
            ),
            RuntimeError("fixture failure"),
        ]
    )
    agent.tool_registry.register("ok", lambda request: ToolResult.success({}), capability="read")

    def observe(*args):
        raise RuntimeError("observation fault")

    agent._build_tool_gateway = lambda: ToolGateway(
        agent.tool_registry, post_call=observe if kind == "dispatch" else None
    )
    auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path).run_v1("go")
    usage = agent.last_auto_state["execution_metrics"]["tool_loop_model_usage"]
    assert usage["reported_calls"] == 1
    assert usage["started_calls"] == (2 if kind == "exception" else 1)
    assert usage["total_tokens"] == (None if kind == "exception" else 3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("started_calls", True),
        ("reported_calls", -1),
        ("unknown_calls", "0"),
        ("started_calls", 3),
        ("total_tokens", True),
        ("total_tokens", None),
        ("reported_sum", {"prompt_tokens": 1, "completion_tokens": 2}),
        ("reported_sum", {"prompt_tokens": False, "completion_tokens": 2, "total_tokens": 3}),
    ],
)
def test_projection_normalizer_rejects_invalid_summary_without_echo(field, value):
    from shared.auto_models import normalize_auto_state

    usage = {
        "started_calls": 1,
        "reported_calls": 1,
        "unknown_calls": 0,
        "reported_sum": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "total_tokens": 3,
    }
    state = {
        "run_id": "fixture",
        "status": "completed",
        "execution_metrics": {"tool_loop_model_usage": usage, "private_raw": "SECRET"},
    }
    normalized = normalize_auto_state(state)
    assert normalized["execution_metrics"]["tool_loop_model_usage"] == usage
    assert "private_raw" not in normalized["execution_metrics"]
    usage[field] = value
    assert "execution_metrics" not in normalize_auto_state(state)


def test_projection_normalization_preserves_unknown_and_zero_calls():
    from shared.auto_models import normalize_auto_state

    for unknown in [0, 1]:
        usage = {
            "started_calls": unknown,
            "reported_calls": 0,
            "unknown_calls": unknown,
            "reported_sum": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "total_tokens": None,
        }
        state = {
            "run_id": "fixture",
            "status": "cancelled",
            "execution_metrics": {"tool_loop_model_usage": usage},
        }
        assert normalize_auto_state(state)["execution_metrics"]["tool_loop_model_usage"] == usage
        usage["reported_sum"]["total_tokens"] = 2
        assert "execution_metrics" not in normalize_auto_state(state)
