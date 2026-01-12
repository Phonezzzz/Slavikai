from __future__ import annotations

from pathlib import Path
from typing import cast

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
from tests.report_utils import extract_report_block


class DummyBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)


def _make_agent(tmp_path: Path, brain: DummyBrain) -> Agent:
    agent = Agent(brain=brain, memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent


def test_m10_chat_route_does_not_call_mwv(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain("chat")
    agent = _make_agent(tmp_path, brain)

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("MWV should not be called for chat input.")

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="что такое переменная")])
    assert response.startswith("chat")
    assert brain.calls == 1
    report = extract_report_block(response)
    assert report["route"] == "chat"


def test_m10_code_route_returns_mwv_report(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain("chat")
    agent = _make_agent(tmp_path, brain)

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
                status=VerificationStatus.PASSED,
                command=["check"],
                exit_code=0,
                stdout="",
                stderr="",
                duration_seconds=0.1,
                error=None,
            )

    monkeypatch.setattr(agent, "_mwv_worker_runner", _worker)
    monkeypatch.setattr(agent_module, "VerifierRuntime", DummyVerifierRuntime)
    response = agent.respond([LLMMessage(role="user", content="поправь баг в коде")])

    assert "Итог:" in response
    assert "Verifier: PASS" in response
    assert "Изменения:" in response
    assert "foo.py" in response
    assert brain.calls == 0
    report = extract_report_block(response)
    assert report["route"] == "mwv"
    verifier = cast(dict[str, object], report["verifier"])
    assert verifier["status"] == "ok"
    assert isinstance(verifier["duration_ms"], int)
    attempts = cast(dict[str, object], report["attempts"])
    assert attempts["current"] == 1


def test_m10_verifier_failure_returns_diagnostics(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain("chat")
    agent = _make_agent(tmp_path, brain)

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

    lowered = response.lower()
    assert "что случилось" in lowered
    assert "проверки" in lowered
    assert "tests failed" in response
    assert "что делать дальше" in lowered
    assert "trace_id=" in response
    assert brain.calls == 0
    report = extract_report_block(response)
    assert report["route"] == "mwv"
    assert report["stop_reason_code"] == "VERIFIER_FAILED"
    verifier = cast(dict[str, object], report["verifier"])
    assert verifier["status"] == "fail"
    assert isinstance(verifier["duration_ms"], int)
