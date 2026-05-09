from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class SimpleBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def test_agent_unknown_tool_command(tmp_path: Path) -> None:
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    resp = agent.handle_tool_command("/unknown")
    assert "неизвестен" in resp.lower() or "неактив" in resp.lower()


def test_agent_shell_disabled_in_safe_mode(tmp_path: Path) -> None:
    agent = Agent(
        brain=SimpleBrain(),
        enable_tools={"safe_mode": True},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    resp = agent.handle_tool_command("/sh ls")
    lowered = resp.lower()
    assert "отключена" in lowered
    assert "/trace" in lowered
    assert "command_lane" in lowered


def test_agent_remember_command_is_not_command_lane_tool(tmp_path: Path) -> None:
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )

    response = agent.handle_tool_command("/remember i prefer markdown output")

    assert "command_lane" in response.lower()
    assert "отключена" in response.lower()
    atom = agent._canonical_store.get_by_stable_key("preference:response_format")
    assert atom is None


def test_agent_end_session_command_saves_canonical_summary(tmp_path: Path) -> None:
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.short_term.extend(
        [
            LLMMessage(role="user", content="обсудили память"),
            LLMMessage(role="assistant", content="сохраним summary"),
        ]
    )

    response = agent.handle_tool_command("/end-session")

    assert "command_lane" in response.lower()
    assert "Сессия закрыта, резюме сохранено: session:" in response
    atoms = agent._canonical_store.list_atoms(stable_key_prefix="session:")
    assert len(atoms) == 1
    assert atoms[0].value_json == {"text": "ok"}
    assert atoms[0].summary_text.startswith("session summary ")
    assert agent.short_term == []


def test_agent_end_session_command_skips_empty_session(tmp_path: Path) -> None:
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )

    response = agent.handle_tool_command("/end-session")

    assert "command_lane" in response.lower()
    assert "Сессия не содержит сообщений для резюме." in response
    assert agent._canonical_store.list_atoms(stable_key_prefix="session:") == []
