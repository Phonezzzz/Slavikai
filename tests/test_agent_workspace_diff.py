from __future__ import annotations

from pathlib import Path

import pytest

import core.agent as agent_module
from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import ToolRequest, ToolResult


class DummyBrain(Brain):
    def generate(self, messages, config: ModelConfig | None = None) -> LLMResult:  # noqa: ANN001
        return LLMResult(text="ok")


def _make_agent(tmp_path: Path) -> Agent:
    return Agent(
        brain=DummyBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )


def test_workspace_diff_pre_post_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(tmp_path)
    monkeypatch.setattr(agent_module, "WORKSPACE_ROOT", tmp_path)

    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")

    request = ToolRequest(name="workspace_write", args={"path": "file.txt"})
    context = agent._workspace_diff_pre_call(request)  # noqa: SLF001
    assert context == "file.txt"

    target.write_text("new", encoding="utf-8")
    result = ToolResult.success()
    agent._workspace_diff_post_call(request, result, context)  # noqa: SLF001

    diffs = agent.consume_workspace_diffs()
    assert diffs and diffs[0].path == "file.txt"
    assert "file.txt" in diffs[0].diff


def test_workspace_diff_dry_run_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(tmp_path)
    monkeypatch.setattr(agent_module, "WORKSPACE_ROOT", tmp_path)

    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")

    request = ToolRequest(
        name="workspace_patch",
        args={"path": "file.txt", "dry_run": True},
    )
    assert agent._workspace_diff_pre_call(request) is None  # noqa: SLF001


def test_normalize_workspace_path_rejects_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _make_agent(tmp_path)
    monkeypatch.setattr(agent_module, "WORKSPACE_ROOT", tmp_path)

    assert agent._normalize_workspace_path("../outside.txt") is None  # noqa: SLF001


def test_read_workspace_text_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(tmp_path)
    monkeypatch.setattr(agent_module, "MAX_FILE_BYTES", 1)

    target = tmp_path / "big.txt"
    target.write_text("ab", encoding="utf-8")

    assert agent._read_workspace_text(target) is None  # noqa: SLF001
