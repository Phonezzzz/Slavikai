from __future__ import annotations

from core.mwv.interfaces import AgentFacade, VerifierFacade, WorkerFacade
from core.mwv.models import (
    RunContext,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkResult,
    WorkStatus,
)
from shared.models import LLMMessage


class DummyAgent:
    def respond(self, messages: list[LLMMessage]) -> str:
        return messages[-1].content if messages else ""


class DummyWorker:
    def run(self, task: TaskPacket, context: RunContext) -> WorkResult:
        return WorkResult(task_id=task.task_id, status=WorkStatus.SUCCESS, summary="ok")


class DummyVerifier:
    def run(self, context: RunContext) -> VerificationResult:
        return VerificationResult(
            status=VerificationStatus.PASSED,
            command=["check"],
            exit_code=0,
            stdout="",
            stderr="",
            duration_seconds=0.0,
            error=None,
        )


def test_interfaces_are_runtime_checkable() -> None:
    assert isinstance(DummyAgent(), AgentFacade)
    assert isinstance(DummyWorker(), WorkerFacade)
    assert isinstance(DummyVerifier(), VerifierFacade)
