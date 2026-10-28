import pytest

import core.auto_runtime as auto_runtime
from core.mwv.models import StopReasonCode
from llm.types import LLMResult, LLMUsage
from shared.models import ToolResult
from tests.test_auto_runtime import _AutoV1Agent
from tests.test_auto_usage_accounting import UsageBrain
from tests.test_tool_call_cap import calls


@pytest.mark.parametrize("unknown", [False, True])
def test_auto_blocks_next_generation_at_limit_or_unknown(tmp_path, monkeypatch, unknown):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(auto_runtime.AUTO_MAX_LLM_TOKENS_ENV, "5")
    agent = _AutoV1Agent()
    responses = [
        LLMResult(
            text="tools", tool_calls=calls("tick"), usage=None if unknown else LLMUsage(2, 3, 5)
        ),
        LLMResult(
            text="unexpected third" if not unknown else "unexpected second", usage=LLMUsage(0, 0, 0)
        ),
    ]
    if not unknown:
        responses.insert(
            0, LLMResult(text="first", tool_calls=calls("tick"), usage=LLMUsage(0, 0, 0))
        )
    agent._brain = UsageBrain(responses)
    dispatched = []
    agent.tool_registry.register(
        "tick", lambda request: dispatched.append(True) or ToolResult.success({}), capability="read"
    )
    outcome = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path).run_v1("go")
    assert agent._brain.requests == (1 if unknown else 2)
    assert len(dispatched) == (1 if unknown else 2)
    assert outcome.stop_reason_code == StopReasonCode.BUDGET_EXHAUSTED
    assert agent.last_auto_state["error_code"] == (
        "budget_tokens_unknown" if unknown else "budget_tokens"
    )
    if unknown:
        assert "exhausted" not in agent.last_stop_kwargs["what"]
        assert "неизвест" in agent.last_stop_kwargs["why"]


@pytest.mark.parametrize("total", [0, 4, 5, 6])
def test_exact_threshold_blocks_only_next_genuine_request(tmp_path, total):
    from core.tool_gateway import ToolGateway
    from core.tool_loop import AgentToolLoop
    from tests.test_auto_usage_accounting import execute
    from tests.test_tool_call_cap import setup

    registry, dispatched = setup(tmp_path)
    brain = UsageBrain(
        [
            LLMResult(text="tool", tool_calls=calls("ok"), usage=LLMUsage(0, 0, total)),
            LLMResult(text="final", usage=None),
        ]
    )
    result = execute(AgentToolLoop(max_model_tokens=5), brain, ToolGateway(registry))
    assert brain.requests == (2 if total < 5 else 1)
    assert dispatched == ["ok"]
    assert result.error == (None if total < 5 else "model_token_budget_exhausted")
    assert result.model_usage.started_calls == brain.requests


@pytest.mark.parametrize("usage", [None, LLMUsage(2, 3, 5), LLMUsage(3, 4, 7)])
def test_final_response_can_pass_verifier_without_another_request(tmp_path, monkeypatch, usage):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(auto_runtime.AUTO_MAX_LLM_TOKENS_ENV, "5")
    agent = _AutoV1Agent()
    agent._brain = UsageBrain([LLMResult(text="final", usage=usage)])
    outcome = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path).run_v1("go")
    assert outcome.verifier.ok and agent._brain.requests == 1
    assert agent.last_auto_state["error_code"] is None


@pytest.mark.parametrize("unknown", [False, True])
def test_captured_cap_allows_repeated_approved_remainder_then_blocks_model(tmp_path, unknown):
    from core.approval_policy import ApprovalContext, ApprovalRequired
    from core.tool_gateway import ToolGateway
    from core.tool_loop import AgentToolLoop
    from tests.test_auto_usage_accounting import execute
    from tests.test_tool_call_cap import setup

    registry, dispatched = setup(tmp_path)
    gateway = ToolGateway(registry, approval_context=ApprovalContext(True, "s", set()))
    brain = UsageBrain(
        [
            LLMResult(
                text="batch",
                tool_calls=calls("network", "network"),
                usage=None if unknown else LLMUsage(2, 3, 5),
            )
        ]
    )
    with pytest.raises(ApprovalRequired) as first:
        execute(AgentToolLoop(max_model_tokens=5), brain, gateway)
    assert first.value.continuation.max_model_tokens == 5
    with pytest.raises(ApprovalRequired) as second:
        execute(
            AgentToolLoop(max_model_tokens=500),
            brain,
            gateway,
            continuation=first.value.continuation,
        )
    result = execute(AgentToolLoop(), brain, gateway, continuation=second.value.continuation)
    assert brain.requests == 1 and dispatched == ["network", "network"]
    assert result.error == ("model_usage_unknown" if unknown else "model_token_budget_exhausted")
    assert len(result.tool_calls) == 2 and result.model_usage.started_calls == 1


@pytest.mark.parametrize("cancel", [False, True])
def test_cancel_and_expiry_precede_token_stop_after_tool(tmp_path, monkeypatch, cancel):
    import asyncio
    from types import SimpleNamespace

    import core.tool_loop as tool_loop
    from core.tool_gateway import ToolGateway
    from core.tool_loop import AgentToolLoop
    from tests.test_auto_usage_accounting import execute
    from tests.test_tool_call_cap import setup

    timer = SimpleNamespace(monotonic=lambda: 10.0)
    monkeypatch.setattr(tool_loop, "time", timer)
    token = asyncio.Event()
    registry, _ = setup(tmp_path)

    def stop(request):
        if cancel:
            token.set()
        else:
            timer.monotonic = lambda: 11.0
        return ToolResult.success({})

    registry.register("stop", stop, capability="read")
    brain = UsageBrain([LLMResult(text="tool", tool_calls=calls("stop"), usage=LLMUsage(2, 3, 5))])
    result = execute(
        AgentToolLoop(max_model_tokens=5, expires_at_monotonic=11.0),
        brain,
        ToolGateway(registry),
        cancellation_token=token,
    )
    assert result.cancelled == cancel
    assert result.error == (None if cancel else "runtime_budget_exhausted")
    assert brain.requests == 1 and len(result.tool_calls) == 1


def test_uncapped_checkpoint_does_not_inherit_new_loop_limit(tmp_path):
    from core.approval_policy import ApprovalContext, ApprovalRequired
    from core.tool_gateway import ToolGateway
    from core.tool_loop import AgentToolLoop
    from tests.test_auto_usage_accounting import execute
    from tests.test_tool_call_cap import setup

    registry, dispatched = setup(tmp_path)
    gateway = ToolGateway(registry, approval_context=ApprovalContext(True, "s", set()))
    brain = UsageBrain(
        [
            LLMResult(text="tool", tool_calls=calls("network"), usage=None),
            LLMResult(text="final", usage=None),
        ]
    )
    with pytest.raises(ApprovalRequired) as first:
        execute(AgentToolLoop(), brain, gateway)
    result = execute(
        AgentToolLoop(max_model_tokens=1), brain, gateway, continuation=first.value.continuation
    )
    assert result.error is None and result.model_usage is None
    assert brain.requests == 2 and dispatched == ["network"]


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "5"])
def test_invalid_token_cap_is_rejected(cap):
    from core.tool_loop import AgentToolLoop

    with pytest.raises(ValueError, match="max_model_tokens"):
        AgentToolLoop(max_model_tokens=cap)


def test_auto_env_change_does_not_replace_captured_cap_across_principals(tmp_path, monkeypatch):
    from core.approval_policy import ApprovalContext, ApprovalRequired
    from core.tool_gateway import ToolGateway

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(auto_runtime.AUTO_MAX_LLM_TOKENS_ENV, "5")
    runs = []
    for principal, total in [("one", 5), ("two", 2)]:
        agent = _AutoV1Agent()
        agent.user_id = agent.session_id = principal
        agent._brain = UsageBrain(
            [
                LLMResult(text="tool", tool_calls=calls("network"), usage=LLMUsage(0, 0, total)),
                LLMResult(text="final", usage=LLMUsage(0, 0, 0)),
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
        orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)
        with pytest.raises(ApprovalRequired):
            orchestrator.run_v1("go")
        runs.append((agent, orchestrator, agent.last_auto_state["run_id"], total))
    monkeypatch.setenv(auto_runtime.AUTO_MAX_LLM_TOKENS_ENV, "1")
    for agent, orchestrator, run_id, total in runs:
        outcome = orchestrator.resume(run_id)
        assert agent._brain.requests == (1 if total == 5 else 2)
        assert outcome.stop_reason_code == (StopReasonCode.BUDGET_EXHAUSTED if total == 5 else None)
        assert len(agent.last_auto_state["coders"]) == 1
