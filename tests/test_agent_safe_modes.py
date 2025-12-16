from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage, ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry


class SimpleBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


class DummyTool:
    def __init__(self) -> None:
        self.calls = 0

    def handle(self, request: ToolRequest) -> ToolResult:
        self.calls += 1
        return ToolResult.success({"output": request.name})


def test_safe_mode_off_allows_tools(tmp_path: Path) -> None:
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    registry = ToolRegistry(safe_block={"web", "shell"})
    dummy = DummyTool()
    registry.register("web", dummy, enabled=True)
    registry.apply_safe_mode(False)
    agent.tool_registry = registry

    result = agent.tool_registry.call(ToolRequest(name="web", args={}))
    assert result.ok
    assert dummy.calls == 1


def test_safe_mode_on_blocks_web_shell(tmp_path: Path) -> None:
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    registry = ToolRegistry(safe_block={"web", "shell"})
    dummy = DummyTool()
    registry.register("web", dummy, enabled=True)
    registry.register("shell", dummy, enabled=True)
    registry.apply_safe_mode(True)
    agent.tool_registry = registry

    res_web = agent.tool_registry.call(ToolRequest(name="web", args={}))
    res_shell = agent.tool_registry.call(ToolRequest(name="shell", args={"command": "ls"}))
    assert not res_web.ok and res_web.error
    assert not res_shell.ok and res_shell.error
    assert "Safe mode" in res_web.error or "safe mode" in res_web.error.lower()
    assert dummy.calls == 0
