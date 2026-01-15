from __future__ import annotations

import json
from pathlib import Path

import core.agent as agent_module
from core.agent import Agent
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
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="chat")


def _make_agent(tmp_path: Path) -> Agent:
    agent = Agent(brain=DummyBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
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
    assert any(option["action"] == "retry" for option in payload["options"])
