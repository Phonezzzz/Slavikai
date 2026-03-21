from __future__ import annotations

import json
from pathlib import Path

import core.agent as agent_module
from core.agent import Agent
from core.decision.models import DecisionAction
from core.decision.verifier_fail import build_verifier_fail_packet
from core.mwv.models import (
    ChangeType,
    VerificationResult,
    VerificationStatus,
    WorkChange,
    WorkResult,
    WorkStatus,
)
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class DummyBrain(Brain):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text="chat")


def _make_agent(tmp_path: Path) -> Agent:
    agent = Agent(
        brain=DummyBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent


def test_decision_packet_on_verifier_fail(tmp_path: Path, monkeypatch) -> None:
    agent = _make_agent(tmp_path)

    def _worker(_task: object, _context: object) -> WorkResult:
        return WorkResult(
            task_id="t1",
            status=WorkStatus.SUCCESS,
            summary="ok",
            changes=[WorkChange(path="foo.py", change_type=ChangeType.UPDATE, summary="+1/-0")],
        )

    class DummyVerifierRuntime:
        def run(self, _context: object) -> VerificationResult:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                command=["check"],
                exit_code=1,
                stdout="",
                stderr="tests failed",
                duration_seconds=0.1,
                error=None,
            )

    monkeypatch.setattr(agent, "_mwv_worker_runner", _worker)
    monkeypatch.setattr(agent_module, "VerifierRuntime", DummyVerifierRuntime)

    response = agent.respond([LLMMessage(role="user", content="почини тесты")])
    payload = json.loads(response)
    assert payload["reason"] == "verifier_fail"
    assert 3 <= len(payload["options"]) <= 5
    actions = {option["action"] for option in payload["options"]}
    assert "retry" not in actions
    assert {"ask_user", "proceed_safe", "abort"} <= actions
    assert agent.brain.calls == 0


def test_build_verifier_fail_packet_respects_retry_flag() -> None:
    result = VerificationResult(
        status=VerificationStatus.FAILED,
        command=["check"],
        exit_code=1,
        stdout="line1\nline2\nline3\nline4",
        stderr="tests failed",
        duration_seconds=0.1,
        error=None,
    )

    no_retry = build_verifier_fail_packet(
        result,
        task_id="t1",
        trace_id="trace-1",
        attempt=1,
        max_attempts=3,
        retry_allowed=False,
    )
    assert DecisionAction.RETRY not in [option.action for option in no_retry.options]

    with_retry = build_verifier_fail_packet(
        result,
        task_id="t1",
        trace_id="trace-1",
        attempt=1,
        max_attempts=3,
        retry_allowed=True,
    )
    retry_actions = [option.action for option in with_retry.options]
    assert retry_actions.count(DecisionAction.RETRY) == 1
