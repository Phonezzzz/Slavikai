import asyncio
from pathlib import Path

import pytest

from core.agent import Agent
from core.agent_response import ResponseProduced
from core.mwv.manager import MWVRunResult
from core.mwv.models import TaskStepContract, VerificationStatus, WorkStatus
from llm.stream_model import Done
from shared.models import LLMMessage, ToolResult
from tests.test_agent_response import SimpleBrain


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tool_ok", [False, True])
@pytest.mark.parametrize("verifier_ok", [False, True])
@pytest.mark.parametrize("projection_failure", [None, "format", "log"])
def test_mwv_result_survives_agent_projection(
    tmp_path: Path,
    monkeypatch,
    streaming: bool,
    tool_ok: bool,
    verifier_ok: bool,
    projection_failure: str | None,
) -> None:
    brain = SimpleBrain("must not regenerate")
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.apply_runtime_workspace_root(str(tmp_path))
    agent.memory.get_recent = lambda *a, **k: []
    agent.memory.get_user_prefs = lambda: []
    agent.vectors.search = lambda *a, **k: []
    (tmp_path / "Makefile").write_text(
        "check:\n\t@echo verifier-diagnostic\n\t@exit " + ("0" if verifier_ok else "1") + "\n"
    )
    executions = []
    observed = (
        ToolResult.success({"output": "worker-diagnostic"})
        if tool_ok
        else (ToolResult.failure("worker-diagnostic", data={"output": "worker-diagnostic"}))
    )
    agent.tool_registry.register(
        "mwv_probe",
        lambda request: executions.append(request) or observed,
        enabled=True,
        capability="read",
        description="Diagnostic read",
        parameters_schema={"type": "object"},
    )
    monkeypatch.setattr(
        agent,
        "_build_mwv_task_steps",
        lambda goal: [
            TaskStepContract(
                step_id="probe-1",
                title="Probe",
                description=goal,
                allowed_tool_kinds=["mwv_probe"],
                inputs={"operation": "mwv_probe", "tool_args": {}},
            )
        ],
    )
    # Настоящие Worker/Gateway/verifier; spy сохраняет исходный manager result.
    from core.mwv.manager import ManagerRuntime

    actual_results = []
    original = ManagerRuntime.run_flow

    def retain(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        actual_results.append(result)
        return result

    monkeypatch.setattr(ManagerRuntime, "run_flow", retain)
    if projection_failure:

        def fail(*args, **kwargs):
            raise RuntimeError("presentation unavailable")

        monkeypatch.setattr(
            agent,
            "_format_mwv_response" if projection_failure == "format" else "_log_chat_interaction",
            fail,
        )
        if projection_failure == "format" and not verifier_ok:
            monkeypatch.setattr(agent, "_handle_decision_packet", fail)
    messages = [LLMMessage(role="user", content="почини тесты")]
    if streaming:
        from server.http.handlers.ui_chat import _iterate_stream_in_thread

        async def consume():
            return [
                event
                async for event in _iterate_stream_in_thread(
                    agent.respond_stream(messages),
                    asyncio.Event(),
                )
            ]

        events = asyncio.run(consume())
        produced = [event.response for event in events if isinstance(event, ResponseProduced)]
        assert len(produced) == 1
        response = produced[0]
        assert isinstance(events[-1], Done)
        assert events[-1].finish_reason == ("error" if projection_failure else "stop")
    else:
        response = agent.respond(messages)
    result = response.runtime_result
    assert isinstance(result, MWVRunResult)
    assert result is actual_results[0]
    assert result.work_result.status == (WorkStatus.SUCCESS if tool_ok else WorkStatus.FAILURE)
    assert result.verification_result.status == (
        VerificationStatus.PASSED if verifier_ok else VerificationStatus.FAILED
    )
    assert "verifier-diagnostic" in result.verification_result.stdout
    assert "worker-diagnostic" in str(result.work_result.diagnostics)
    assert len(executions) == result.attempt
    assert brain.calls == 0
    if projection_failure:
        assert response.failure is not None
        assert response.failure.code == "mwv_projection_error"
    else:
        assert response.failure is None
    agent.close()
