from __future__ import annotations

import json
from pathlib import Path

from core.agent import Agent
from core.skills.index import SkillIndex, SkillResolution
from core.skills.models import SkillEntry, SkillManifest
from llm.brain_base import Brain
from llm.stream_model import Done, TextDelta
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage
from tests.report_utils import extract_report_block


class CountingBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)


def _prepare_agent(tmp_path: Path) -> tuple[Agent, CountingBrain]:
    main = CountingBrain("main")
    agent = Agent(
        brain=main,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent, main


def test_agent_chat_uses_main_brain(tmp_path: Path) -> None:
    agent, main = _prepare_agent(tmp_path)
    response = agent.respond([LLMMessage(role="user", content="привет")])
    assert response.startswith("main")
    assert main.calls == 1
    report = extract_report_block(response)
    assert report["route"] == "chat"


def test_agent_mwv_route_bypasses_brain(tmp_path: Path, monkeypatch) -> None:
    agent, main = _prepare_agent(tmp_path)

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        return "mwv"

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="исправь тесты")])
    assert response == "mwv"
    assert main.calls == 0


def test_agent_ask_mode_uses_chat_path_for_action_text(tmp_path: Path, monkeypatch) -> None:
    agent, main = _prepare_agent(tmp_path)
    agent.runtime_mode = "ask"

    def _mwv_unreachable(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("MWV should not run in ask mode")

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_unreachable)
    response = agent.respond([LLMMessage(role="user", content="исправь тесты")])
    assert response.startswith("main")
    assert main.calls == 1


def test_agent_auto_mode_chat_like_request_uses_auto_runtime(tmp_path: Path, monkeypatch) -> None:
    agent, main = _prepare_agent(tmp_path)
    agent.runtime_mode = "auto"
    calls: dict[str, object] = {}

    def _chat_unreachable(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("Chat fallback must be disabled for runtime_mode=auto")

    def _auto_stub(
        goal: str,
        *,
        command_lane: bool = False,
        skill_resolution: SkillResolution | None = None,
    ) -> str:
        calls["goal"] = goal
        calls["command_lane"] = command_lane
        calls["skill_resolution"] = skill_resolution
        return "auto-advisory"

    monkeypatch.setattr(agent, "_run_chat_response", _chat_unreachable)
    monkeypatch.setattr(agent, "handle_auto_command", _auto_stub)
    response = agent.respond(
        [LLMMessage(role="user", content="Какой софт нужен для Raspberry Pi 4 для умной колонки?")]
    )
    assert response == "auto-advisory"
    assert main.calls == 0
    assert calls.get("goal") == "Какой софт нужен для Raspberry Pi 4 для умной колонки?"
    assert calls.get("command_lane") is False
    assert calls.get("skill_resolution") is None


def test_agent_auto_mode_execution_request_uses_auto_runtime(tmp_path: Path, monkeypatch) -> None:
    agent, main = _prepare_agent(tmp_path)
    agent.runtime_mode = "auto"
    calls: dict[str, object] = {}

    def _chat_unreachable(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("Chat fallback should not run for execution-like AUTO request")

    def _auto_stub(
        goal: str,
        *,
        command_lane: bool = False,
        skill_resolution: SkillResolution | None = None,
    ) -> str:
        calls["goal"] = goal
        calls["command_lane"] = command_lane
        calls["skill_resolution"] = skill_resolution
        return "auto-run"

    monkeypatch.setattr(agent, "_run_chat_response", _chat_unreachable)
    monkeypatch.setattr(agent, "handle_auto_command", _auto_stub)

    response = agent.respond(
        [LLMMessage(role="user", content="исправь тесты и обнови файл src/main.py")]
    )
    assert response == "auto-run"
    assert main.calls == 0
    assert calls.get("goal") == "исправь тесты и обнови файл src/main.py"
    assert calls.get("command_lane") is False
    assert calls.get("skill_resolution") is None


def test_agent_auto_mode_does_not_call_route_classifier(tmp_path: Path, monkeypatch) -> None:
    agent, main = _prepare_agent(tmp_path)
    agent.runtime_mode = "auto"

    def _classifier_unreachable(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Auto mode must not call classify_request")

    def _auto_stub(
        goal: str,
        *,
        command_lane: bool = False,
        skill_resolution: SkillResolution | None = None,
    ) -> str:
        assert command_lane is False
        assert skill_resolution is None
        return f"auto:{goal}"

    monkeypatch.setattr("core.agent_routing.classify_request", _classifier_unreachable)
    monkeypatch.setattr(agent, "handle_auto_command", _auto_stub)

    response = agent.respond([LLMMessage(role="user", content="workspace_read status")])

    assert response == "auto:workspace_read status"
    assert main.calls == 0


def test_agent_auto_stream_does_not_call_route_classifier(tmp_path: Path, monkeypatch) -> None:
    agent, main = _prepare_agent(tmp_path)
    agent.runtime_mode = "auto"

    def _classifier_unreachable(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Auto stream mode must not call classify_request")

    def _auto_stub(
        goal: str,
        *,
        command_lane: bool = False,
        skill_resolution: SkillResolution | None = None,
    ) -> str:
        assert command_lane is False
        assert skill_resolution is None
        return f"auto-stream:{goal}"

    monkeypatch.setattr("core.agent_routing.classify_request", _classifier_unreachable)
    monkeypatch.setattr(agent, "handle_auto_command", _auto_stub)

    chunks = list(agent.respond_stream([LLMMessage(role="user", content="image_generate plan")]))

    assert chunks == [TextDelta(text="auto-stream:image_generate plan"), Done()]
    assert agent.last_stream_response_raw == "auto-stream:image_generate plan"
    assert main.calls == 0


def test_agent_auto_mode_passes_resolved_skill_to_auto_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agent, main = _prepare_agent(tmp_path)
    agent.runtime_mode = "auto"
    captured: dict[str, SkillResolution | None] = {}

    def _auto_stub(
        goal: str,
        *,
        command_lane: bool = False,
        skill_resolution: SkillResolution | None = None,
    ) -> str:
        assert goal == "implement spec for skill runtime"
        assert command_lane is False
        captured["resolution"] = skill_resolution
        return "auto-skill"

    monkeypatch.setattr(agent, "handle_auto_command", _auto_stub)

    response = agent.respond([LLMMessage(role="user", content="implement spec for skill runtime")])

    assert response == "auto-skill"
    resolution = captured["resolution"]
    assert resolution is not None
    assert resolution.primary.id == "implement"
    assert [entry.id for entry in resolution.supporting] == ["codebase-design"]
    assert main.calls == 0


def test_agent_auto_mode_returns_structured_ambiguous_skill_decision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agent, main = _prepare_agent(tmp_path)
    agent.runtime_mode = "auto"
    entries = [
        SkillEntry(
            id=skill_id,
            version="1.0.0",
            title=skill_id,
            entrypoints=["auto"],
            patterns=["same trigger"],
            requires=[],
            risk="low",
            tests=[],
            path=f"skills/{skill_id}/skill.md",
            content_hash=f"hash-{skill_id}",
            instructions=f"Instructions for {skill_id}",
        )
        for skill_id in ("alpha", "beta")
    ]
    agent.skill_index = SkillIndex(SkillManifest(manifest_version=2, skills=entries))

    def _auto_unreachable(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("Ambiguous skill must stop before Auto runtime")

    monkeypatch.setattr(agent, "handle_auto_command", _auto_unreachable)

    response = agent.respond([LLMMessage(role="user", content="same trigger")])
    decision = json.loads(response)

    assert decision["reason"] == "ambiguous_skill"
    assert any(option["action"] == "select_skill" for option in decision["options"])
    assert agent.last_decision_packet is not None
    assert main.calls == 0


def test_agent_auto_mode_stream_returns_structured_ambiguous_skill_decision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agent, main = _prepare_agent(tmp_path)
    agent.runtime_mode = "auto"
    entries = [
        SkillEntry(
            id=skill_id,
            version="1.0.0",
            title=skill_id,
            entrypoints=["auto"],
            patterns=["same trigger"],
            requires=[],
            risk="low",
            tests=[],
            path=f"skills/{skill_id}/skill.md",
            content_hash=f"hash-{skill_id}",
            instructions=f"Instructions for {skill_id}",
        )
        for skill_id in ("alpha", "beta")
    ]
    agent.skill_index = SkillIndex(SkillManifest(manifest_version=2, skills=entries))

    def _auto_unreachable(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("Ambiguous skill must stop before Auto runtime")

    monkeypatch.setattr(agent, "handle_auto_command", _auto_unreachable)

    events = list(agent.respond_stream([LLMMessage(role="user", content="same trigger")]))
    text = "".join(event.text for event in events if isinstance(event, TextDelta))
    decision = json.loads(text)

    assert decision["reason"] == "ambiguous_skill"
    assert any(option["action"] == "select_skill" for option in decision["options"])
    assert agent.last_decision_packet is not None
    assert main.calls == 0
