from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from core.mwv.manager import MWVRunResult
from core.mwv.models import (
    RetryDecision,
    RetryPolicy,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkResult,
    WorkStatus,
)
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        _ = messages
        _ = config
        return LLMResult(text="ok")


def test_skill_failure_note_when_verifier_fails(tmp_path: Path) -> None:
    agent = Agent(
        brain=DummyBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    run_result = MWVRunResult(
        task=TaskPacket(
            task_id="t",
            session_id="s",
            trace_id="trace",
            goal="g",
            context={"skill_id": "alpha"},
        ),
        work_result=WorkResult(task_id="t", status=WorkStatus.SUCCESS, summary="ok"),
        verification_result=VerificationResult(
            status=VerificationStatus.FAILED,
            command=["check"],
            exit_code=1,
            stdout="",
            stderr="fail",
            duration_seconds=0.1,
            error=None,
        ),
        attempt=1,
        max_attempts=2,
        retry_decision=RetryDecision(
            policy=RetryPolicy.LIMITED,
            allow_retry=True,
            reason="verifier_failed",
            attempt=1,
            max_retries=1,
        ),
    )
    steps = agent._mwv_next_steps(run_result)  # noqa: SLF001
    assert any("Навык alpha" in step for step in steps)
