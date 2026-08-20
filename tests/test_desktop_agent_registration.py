from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from core.desktop_policy import DesktopPolicyStore
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig, ToolSpec
from shared.models import LLMMessage


class NoopBrain(Brain):
    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        del messages, config, tools
        return LLMResult(text="ok")


def test_phase2_tools_are_registered_only_for_desktop_execution_target(tmp_path: Path) -> None:
    agent = Agent(
        brain=NoopBrain(),
        memory_companion_db_path=str(tmp_path / "memory-companion.db"),
        memory_inbox_db_path=str(tmp_path / "memory-inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
        desktop_policy_store=DesktopPolicyStore(tmp_path / "desktop-approvals.json"),
        desktop_home=tmp_path,
    )
    phase2 = {
        "desktop_clipboard",
        "desktop_system_info",
        "desktop_process",
        "desktop_systemd",
        "desktop_package",
        "desktop_session",
        "desktop_browser",
        "desktop_gui",
    }

    agent.tool_registry.set_execution_policy(mode="ask")
    chat = {spec.name for spec in agent.tool_registry.list_tool_specs()}
    agent.tool_registry.set_execution_policy(mode="auto")
    sandbox_agent = {spec.name for spec in agent.tool_registry.list_tool_specs()}
    agent.tool_registry.set_execution_policy(mode="desktop")
    desktop = {spec.name for spec in agent.tool_registry.list_tool_specs()}

    assert phase2.isdisjoint(chat)
    assert phase2.isdisjoint(sandbox_agent)
    assert phase2 <= desktop
    assert "desktop_launch" not in desktop
