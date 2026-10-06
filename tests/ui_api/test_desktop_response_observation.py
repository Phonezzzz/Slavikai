import asyncio

import pytest

from core.agent_response import ResponseProduced
from core.desktop_policy import DesktopApprovalRule, DesktopApprovalScope
from core.desktop_runtime import DesktopRunOutcome, DesktopRuntime
from llm.stream_model import Done
from server.http.handlers.ui_chat import _iterate_stream_in_thread
from shared.models import LLMMessage

from .test_desktop_mode import DesktopApprovalAgent


@pytest.mark.parametrize("entry", ["sync", "stream", "resume"])
@pytest.mark.parametrize("projection_fault", [None, "project", "log"])
@pytest.mark.parametrize("execution", ["ok", "provider_fault", "tool_and_provider_fault"])
def test_desktop_outcome_reaches_response(
    tmp_path, monkeypatch, entry, projection_fault, execution
):
    provider_fault = execution != "ok"
    tool_ok = execution != "tool_and_provider_fault"
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "Downloads" / "observation.iso"
    target.parent.mkdir()
    target.write_text("fixture")
    agent = DesktopApprovalAgent(str(target))
    agent.set_session_context("session", set())
    agent.set_runtime_state(
        mode="desktop", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    returned = []
    method = "resume" if entry == "resume" else "run"
    original = getattr(DesktopRuntime, method)

    def capture(self, *args, **kwargs):
        outcome = original(self, *args, **kwargs)
        returned.append(outcome)
        return outcome

    monkeypatch.setattr(DesktopRuntime, method, capture)
    try:
        if entry == "resume":
            paused = agent.respond([LLMMessage(role="user", content="delete file")])
            assert paused.runtime_result is None
            assert agent.completed == 0
            identity = agent.last_approval_resume_payload["continuation_id"]
            scope = agent.last_approval_request.scope
        else:
            scope = DesktopApprovalScope(
                tool="desktop_file_delete",
                action="delete",
                target_pattern=str(target),
                risk_class="destructive",
            )
        agent.set_desktop_policy_context(
            [DesktopApprovalRule.create(effect="allow", source="once", scope=scope)], "local"
        )
        if not tool_ok:
            target.unlink()
        if provider_fault:
            original_generate = agent.brain.generate

            def generate(messages, *args, **kwargs):
                if messages[-1].role == "tool":
                    raise RuntimeError("provider unavailable after dispatch")
                return original_generate(messages, *args, **kwargs)

            monkeypatch.setattr(agent.brain, "generate", generate)
        if projection_fault:

            def fail(*args, **kwargs):
                raise RuntimeError("presentation unavailable")

            monkeypatch.setattr(
                agent,
                "_project_desktop_outcome"
                if projection_fault == "project"
                else "_log_chat_interaction",
                fail,
            )
        if entry == "resume":
            response = agent.resume_chat_approval(identity)
        elif entry == "stream":

            async def consume():
                return [
                    event
                    async for event in _iterate_stream_in_thread(
                        agent.respond_stream([LLMMessage(role="user", content="delete file")]),
                        asyncio.Event(),
                    )
                ]

            events = asyncio.run(consume())
            produced = [event.response for event in events if isinstance(event, ResponseProduced)]
            assert len(produced) == 1
            response = produced[0]
            assert isinstance(events[-1], Done)
        else:
            response = agent.respond([LLMMessage(role="user", content="delete file")])
        outcome = response.runtime_result
        assert isinstance(outcome, DesktopRunOutcome)
        assert outcome is returned[0]
        assert agent.completed == (1 if tool_ok else 0) and not target.exists()
        assert outcome.loop_result.tool_calls[0].call.id == "delete-original"
        assert outcome.loop_result.tool_calls[0].result.ok is tool_ok
        if not tool_ok:
            assert outcome.loop_result.tool_calls[0].result.error
        assert len(outcome.loop_result.tool_calls) == (1 if provider_fault else 2)
        assert outcome.loop_result.error == (
            "provider_generation_failed" if provider_fault else None
        )
        assert outcome.verification.ok is (not provider_fault)
        assert (response.failure is not None) is (projection_fault is not None)
        if projection_fault:
            assert response.failure.code == "desktop_projection_error"
        assert agent.desktop_runtime.run_coordinator.try_acquire()
        agent.desktop_runtime.run_coordinator.release()
    finally:
        agent.close()
