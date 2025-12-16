from __future__ import annotations

from pathlib import Path

from core.agent import SAFE_MODE_TOOLS_OFF, Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage, ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry


class SimpleBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def _dummy_tool(ok: bool, output: str = "done") -> ToolResult:
    if ok:
        return ToolResult.success({"output": output})
    return ToolResult.failure("fail")


def test_image_commands_success_and_error(tmp_path: Path) -> None:
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.tool_registry = ToolRegistry(safe_block=SAFE_MODE_TOOLS_OFF)
    agent.tool_registry.register(
        "image_generate", lambda req: _dummy_tool(True, "img-ok"), enabled=True
    )
    agent.tool_registry.register("image_analyze", lambda req: _dummy_tool(False), enabled=True)

    ok_resp = agent.handle_tool_command("/imggen prompt")
    assert "img-ok" in ok_resp

    err_resp = agent.handle_tool_command("/imganalyze path")
    assert "Ошибка инструмента" in err_resp


def test_tts_stt_error_paths(tmp_path: Path) -> None:
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.tool_registry = ToolRegistry(safe_block=SAFE_MODE_TOOLS_OFF)
    agent.tool_registry.register("tts", lambda req: _dummy_tool(False), enabled=True)
    agent.tool_registry.register("stt", lambda req: _dummy_tool(False), enabled=True)

    tts_result = agent.synthesize_speech("hello")
    assert not tts_result.ok
    assert "fail" in (tts_result.error or "")

    stt_result = agent.transcribe_audio("file.wav")
    assert not stt_result.ok
    assert "fail" in (stt_result.error or "")


def test_safe_mode_blocks_web_not_workspace(tmp_path: Path) -> None:
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    # заменить registry, чтобы web включить и проверить safe mode
    agent.tool_registry = ToolRegistry(safe_block=SAFE_MODE_TOOLS_OFF)
    agent.tool_registry.register("web", lambda req: _dummy_tool(True, "ok"), enabled=True)
    agent.tool_registry.register(
        "workspace_list", lambda req: _dummy_tool(True, "ok"), enabled=True
    )

    agent.tool_registry.apply_safe_mode(True)
    web_result = agent.tool_registry.call(ToolRequest(name="web", args={"query": "x"}))
    assert not web_result.ok
    assert "Safe mode" in (web_result.error or "")

    ws_result = agent.tool_registry.call(ToolRequest(name="workspace_list", args={}))
    assert ws_result.ok
