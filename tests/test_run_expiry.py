import asyncio
from types import SimpleNamespace

import pytest

import core.auto_runtime as auto_runtime
import core.tool_loop as tool_loop
from core.approval_policy import ApprovalContext, ApprovalRequired
from core.mwv.models import StopReasonCode, VerificationResult, VerificationStatus
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.brain_base import Brain
from llm.types import LLMResult
from shared.auto_models import AutoRunStatus
from shared.models import ToolResult
from tests.test_auto_runtime import _AutoV1Agent
from tests.test_tool_call_cap import BatchBrain, calls, run, setup


@pytest.fixture
def clock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    clock = SimpleNamespace(now=100.0)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(auto_runtime, "time", clock)
    monkeypatch.setattr(tool_loop, "time", clock, raising=False)
    monkeypatch.setenv(auto_runtime.AUTO_MAX_RUNTIME_SECONDS_ENV, "1")
    return clock


@pytest.mark.parametrize("has_tools", [False, True])
def test_expired_provider_response_starts_neither_tools_nor_verifier(tmp_path, clock, has_tools):
    requested = []
    dispatched = []

    class SlowBrain(Brain):
        supports_native_tools = True

        def generate(self, messages, config=None, tools=None):
            requested.append(True)
            if len(requested) > 1:
                return LLMResult(text="unexpected follow-up")
            clock.now = 101.0
            return LLMResult(text="late answer", tool_calls=calls("tick") if has_tools else [])

    agent = _AutoV1Agent()
    agent._brain = SlowBrain()
    agent.tool_registry.register(
        "tick", lambda request: dispatched.append(True) or ToolResult.success({}), capability="read"
    )
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)

    def forbidden_verifier(**kwargs):
        pytest.fail("Verifier must not start after expiry")

    orchestrator._run_verifier = forbidden_verifier
    outcome = orchestrator.run_v1("go")
    assert requested == [True]
    assert dispatched == []
    assert outcome.stop_reason_code == StopReasonCode.BUDGET_EXHAUSTED
    assert outcome.verifier is None


@pytest.mark.parametrize("cancel", [False, True])
def test_expiry_before_verifier_and_cancellation_precedence(tmp_path, clock, cancel):
    agent = _AutoV1Agent()
    agent._brain = BatchBrain([calls("network")])
    agent.tool_registry.register(
        "network",
        lambda request: ToolResult.success({}),
        capability="read",
        risk_classes=["network"],
    )
    gateway = ToolGateway(agent.tool_registry, approval_context=ApprovalContext(True, "s", set()))
    agent._build_tool_gateway = lambda: gateway
    token = asyncio.Event()

    def progress(state):
        if state["status"] == "verifying":
            clock.now = 101.0
            if cancel:
                token.set()

    orchestrator = auto_runtime.AutoOrchestrator(
        agent, workspace_root=tmp_path, progress_callback=progress
    )
    orchestrator._run_verifier = lambda **kwargs: pytest.fail("Expired verifier was started")
    with pytest.raises(ApprovalRequired):
        orchestrator.run_v1("go")
    outcome = orchestrator.resume(agent.last_auto_state["run_id"], cancellation_token=token)
    assert len(agent.last_auto_state["coders"]) == 1
    assert outcome.verifier is None
    if cancel:
        assert outcome.status == AutoRunStatus.CANCELLED
    else:
        assert outcome.stop_reason_code == StopReasonCode.BUDGET_EXHAUSTED


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("offset", [-0.001, 0.0, 0.5])
def test_exact_expiry_blocks_initial_generation(tmp_path, clock, streaming, offset):
    registry, dispatched = setup(tmp_path)
    requested = []

    class ObservedBrain(BatchBrain):
        def generate(self, *args, **kwargs):
            requested.append(True)
            return super().generate(*args, **kwargs)

    clock.now = 101.0 + offset
    result = run(
        AgentToolLoop(expires_at_monotonic=101.0),
        ObservedBrain([calls("ok")]),
        ToolGateway(registry),
        streaming,
    )
    if offset < 0:
        assert requested == [True, True] and dispatched == ["ok"]
        assert result.error is None
    else:
        assert not requested and not dispatched
        assert result.error == "runtime_budget_exhausted"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("same_batch", [False, True])
def test_expiry_after_handler_blocks_next_model_or_approval(tmp_path, clock, streaming, same_batch):
    registry, dispatched = setup(tmp_path)

    def handler(request):
        dispatched.append(request.name)
        clock.now = 101.0
        return ToolResult.success({"output": "completed before observation returned"})

    registry.register("ok", handler, capability="read")

    class OneRequestBrain(BatchBrain):
        def generate(self, messages, config=None, tools=None):
            if messages[-1].role == "tool":
                pytest.fail("A new model request started after expiry")
            return super().generate(messages, config, tools)

    result = run(
        AgentToolLoop(expires_at_monotonic=101.0),
        OneRequestBrain([calls("ok", "network") if same_batch else calls("ok")]),
        ToolGateway(registry, approval_context=ApprovalContext(True, "s", set())),
        streaming,
    )
    assert dispatched == ["ok"]
    assert result.error == "runtime_budget_exhausted"
    assert len(result.tool_calls) == 1 and result.tool_calls[0].result.ok
    assert len([m for m in result.messages if m.role == "tool"]) == 1


def test_late_stream_response_does_not_dispatch(tmp_path, clock):
    registry, dispatched = setup(tmp_path)

    class SlowStream(BatchBrain):
        def generate(self, messages, config=None, tools=None):
            clock.now = 101.0
            return super().generate(messages, config, tools)

    result = run(
        AgentToolLoop(expires_at_monotonic=101.0),
        SlowStream([calls("ok")]),
        ToolGateway(registry),
        True,
    )
    assert not dispatched and not result.tool_calls
    assert result.error == "runtime_budget_exhausted"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("expired_before_resume", [False, True])
def test_approval_continuation_retains_original_expiry(
    tmp_path, clock, streaming, expired_before_resume
):
    registry, dispatched = setup(tmp_path)

    def handler(request):
        dispatched.append(request.name)
        clock.now = 101.0
        return ToolResult.success({})

    registry.register("network", handler, capability="read", risk_classes=["network"])
    gateway = ToolGateway(registry, approval_context=ApprovalContext(True, "s", set()))
    brain = BatchBrain([calls("network", "ok")])
    with pytest.raises(ApprovalRequired) as stopped:
        run(AgentToolLoop(expires_at_monotonic=101.0), brain, gateway, streaming)
    checkpoint = stopped.value.continuation
    assert checkpoint.expires_at_monotonic == 101.0
    clock.now = 101.0 if expired_before_resume else 100.5
    result = AgentToolLoop(expires_at_monotonic=1000.0).run(
        brain=brain, gateway=gateway, messages=[], tools=[], continuation=checkpoint
    )
    assert dispatched == ([] if expired_before_resume else ["network"])
    assert len(result.tool_calls) == (0 if expired_before_resume else 1)
    assert result.error == "runtime_budget_exhausted"


def test_auto_runs_keep_separate_original_expiry_after_env_change(tmp_path, clock, monkeypatch):
    runs = []
    for principal in ["one", "two"]:
        agent = _AutoV1Agent()
        agent.user_id = agent.session_id = principal
        agent._brain = BatchBrain([calls("network")])
        dispatched = []
        agent.tool_registry.register(
            "network",
            lambda request, seen=dispatched: seen.append(True) or ToolResult.success({}),
            capability="read",
            risk_classes=["network"],
        )
        gateway = ToolGateway(
            agent.tool_registry, approval_context=ApprovalContext(True, principal, set())
        )
        agent._build_tool_gateway = lambda current=gateway: current
        orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)
        with pytest.raises(ApprovalRequired):
            orchestrator.run_v1("go")
        runs.append((orchestrator, agent.last_auto_state["run_id"], dispatched))
        clock.now += 0.5
    monkeypatch.setenv(auto_runtime.AUTO_MAX_RUNTIME_SECONDS_ENV, "1000")
    for index, (orchestrator, run_id, dispatched) in enumerate(runs):
        outcome = orchestrator.resume(run_id)
        if index == 0:
            assert outcome.stop_reason_code == StopReasonCode.BUDGET_EXHAUSTED
            assert not dispatched
        else:
            assert outcome.status == AutoRunStatus.COMPLETED
            assert dispatched == [True]
        with pytest.raises(auto_runtime.AutoContinuationUnavailable):
            orchestrator.resume(run_id)


def test_already_started_verifier_is_not_interrupted(tmp_path, clock):
    agent = _AutoV1Agent()
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)
    verified = []

    def verifier(**kwargs):
        verified.append(True)
        clock.now = 102.0
        return VerificationResult(
            status=VerificationStatus.PASSED,
            command=[],
            exit_code=0,
            stdout="verified",
            stderr="",
            duration_seconds=2.0,
        )

    orchestrator._run_verifier = verifier
    outcome = orchestrator.run_v1("go")
    assert verified == [True] and outcome.status == AutoRunStatus.COMPLETED


@pytest.mark.parametrize("value", [True, "soon", float("nan"), float("inf"), -float("inf")])
def test_expiry_rejects_nonfinite_or_non_numeric_values(value):
    with pytest.raises(ValueError, match="expires_at_monotonic"):
        AgentToolLoop(expires_at_monotonic=value)
