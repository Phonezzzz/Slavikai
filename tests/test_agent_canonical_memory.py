from __future__ import annotations

import numpy as np

from config.memory_config import ContextBudgetConfig
from core.agent import Agent
from core.mwv.models import MWVMessage
from core.mwv.routing import RouteDecision
from llm.brain_base import Brain
from llm.types import LLMResult
from shared.models import LLMMessage


class DummyModel:
    def encode(self, texts):
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


class StaticBrain(Brain):
    def generate(self, messages, config=None):
        del messages, config
        return LLMResult(text="ok")


class JsonMemoryBrain(Brain):
    def generate(self, messages, config=None):
        del messages, config
        return LLMResult(
            text=(
                '[{"claim_type":"ENVIRONMENT","stable_key":"environment:editor",'
                '"value_json":{"value":"neovim"},"summary_text":"user editor is neovim",'
                '"confidence":0.82}]'
            )
        )


def _build_agent(tmp_path, monkeypatch) -> Agent:
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model",
        lambda _self, _name: DummyModel(),
    )
    return Agent(
        brain=StaticBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "canonical.db"),
    )


def test_agent_build_context_includes_canonical_memory(tmp_path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch)
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]

    applied = agent.capture_memory_claims_from_text(
        "запомни: я предпочитаю короткие ответы",
        source_kind="chat.user_input",
        source_id="session-1",
    )
    assert applied

    context = agent._build_context_messages([LLMMessage(role="user", content="hi")], "коротко")
    assert context
    system = context[0]
    assert system.role == "system"
    assert context[1].role == "user"
    assert "[[SLOT:canonical_memory]]" in system.content
    assert "Каноническая память" in system.content


def test_agent_mwv_task_contains_memory_capsule(tmp_path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch)

    agent.capture_memory_claims_from_text(
        "remember i prefer concise responses",
        source_kind="chat.user_input",
        source_id="session-2",
    )

    decision = RouteDecision(route="mwv", reason="trigger:tools", risk_flags=["tools"])
    builder = agent._mwv_task_builder(decision)
    context = agent._build_mwv_context(trace_id="trace-1")
    task = builder([MWVMessage(role="user", content="fix tests")], context)

    capsule_raw = task.context.get("memory_capsule")
    assert isinstance(capsule_raw, dict)
    count_raw = capsule_raw.get("count")
    assert isinstance(count_raw, int)
    assert count_raw >= 1


def test_agent_explicit_remember_response_persists_claim(tmp_path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch)
    agent.runtime_mode = "ask"

    response = agent.respond(
        [LLMMessage(role="user", content="запомни: я предпочитаю короткие ответы")]
    )

    assert response == "Запомнил: preference:response_length"
    atom = agent._canonical_store.get_by_stable_key("preference:response_length")
    assert atom is not None
    assert atom.value_json == {
        "raw": "я предпочитаю короткие ответы",
        "value": "короткие ответы",
    }


def test_agent_explicit_remember_stream_persists_fallback_fact(tmp_path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch)
    agent.runtime_mode = "ask"

    chunks = list(
        agent.respond_stream(
            [LLMMessage(role="user", content="remember my laptop hostname is alpha")]
        )
    )

    assert chunks == ["Запомнил: fact:my_laptop_hostname_is_alpha"]
    assert agent.last_stream_response_raw == "Запомнил: fact:my_laptop_hostname_is_alpha"
    atom = agent._canonical_store.get_by_stable_key("fact:my_laptop_hostname_is_alpha")
    assert atom is not None
    assert atom.value_json == {"text": "my laptop hostname is alpha"}


def test_agent_explicit_remember_uses_llm_enrichment(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model",
        lambda _self, _name: DummyModel(),
    )
    agent = Agent(
        brain=JsonMemoryBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "canonical.db"),
    )
    agent.runtime_mode = "ask"

    response = agent.respond([LLMMessage(role="user", content="remember my editor is neovim")])

    assert "environment:editor" in response
    atom = agent._canonical_store.get_by_stable_key("environment:editor")
    assert atom is not None
    assert atom.value_json == {"value": "neovim"}


def test_agent_context_budget_caps_workspace_and_keeps_system_first(
    tmp_path,
    monkeypatch,
) -> None:
    agent = _build_agent(tmp_path, monkeypatch)
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    object.__setattr__(
        agent.memory_config,
        "context_budget",
        ContextBudgetConfig(
            total_chars=300,
            workspace_file_chars=120,
        ),
    )
    agent.set_workspace_context("big.txt", "x" * 1000)

    context = agent._build_context_messages([LLMMessage(role="user", content="hi")], "hi")

    assert context[0].role == "system"
    assert context[1].role == "user"
    assert agent.last_context_text is not None
    assert len(agent.last_context_text) <= 300
    assert "[[SLOT:workspace_file]]" in agent.last_context_text
    assert "x" * 200 not in agent.last_context_text


def test_agent_context_includes_pinned_atoms_before_canonical_memory(tmp_path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch)
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]

    applied = agent.capture_memory_claims_from_text(
        "remember i prefer markdown output",
        source_kind="chat.explicit_remember",
        source_id="session-3",
    )
    assert applied
    pinned = agent.pin_memory_atom("preference:response_format")
    assert pinned is not None
    assert pinned["pinned"] is True

    context = agent._build_context_messages([LLMMessage(role="user", content="hi")], "format")

    assert context[0].role == "system"
    assert agent.last_context_text is not None
    assert "[[SLOT:pinned_atoms]]" in agent.last_context_text
    assert "preference:response_format" in agent.last_context_text
    assert agent.last_context_text.index("[[SLOT:pinned_atoms]]") < agent.last_context_text.index(
        "[[SLOT:canonical_memory]]"
    )


def test_agent_context_includes_session_summary_slot(tmp_path, monkeypatch) -> None:
    agent = _build_agent(tmp_path, monkeypatch)
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.short_term.extend(
        [
            LLMMessage(role="user", content="обсудили PR-05"),
            LLMMessage(role="assistant", content="сделали session summary"),
        ]
    )

    summary = agent.summarize_current_session()
    assert summary is not None

    context = agent._build_context_messages([LLMMessage(role="user", content="hi")], "memory")

    assert context[0].role == "system"
    assert agent.last_context_text is not None
    assert "[[SLOT:session_summary]]" in agent.last_context_text
    assert "session:" in agent.last_context_text
    assert "ok" in agent.last_context_text
