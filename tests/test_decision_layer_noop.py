from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from core.agent import Agent
from core.decision.models import DecisionAction, DecisionOption, DecisionPacket, DecisionReason
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def _make_agent(tmp_path: Path) -> Agent:
    return Agent(brain=DummyBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))


def _sample_packet() -> DecisionPacket:
    options = [
        DecisionOption(id="opt-1", title="Спросить уточнение", action=DecisionAction.ASK_USER),
        DecisionOption(
            id="opt-2",
            title="Продолжить безопасно",
            action=DecisionAction.PROCEED_SAFE,
        ),
        DecisionOption(id="opt-3", title="Отмена", action=DecisionAction.ABORT),
    ]
    return DecisionPacket(
        id="decision-1",
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        reason=DecisionReason.BLOCKED,
        summary="Нужно решение пользователя.",
        context={"note": "blocked"},
        options=options,
        default_option_id="opt-1",
    )


def test_decision_layer_noop(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    response = agent.respond([LLMMessage(role="user", content="Привет")])
    assert response
    assert agent.last_decision_packet is None


def test_decision_layer_forced_packet(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    packet = _sample_packet()
    agent.decision_handler.force_next(packet)
    response = agent.respond([LLMMessage(role="user", content="Нужно решение")])
    payload = json.loads(response)
    assert payload["id"] == packet.id
    assert payload["reason"] == packet.reason.value
    assert agent.last_decision_packet == packet
