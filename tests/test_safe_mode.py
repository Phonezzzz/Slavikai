from __future__ import annotations

from shared.models import ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry


def test_safe_mode_blocks_network_and_system_tools() -> None:
    blocked_tools = {
        "web",
        "shell",
        "tts",
        "stt",
        "web_search",
        "http_client",
        "image_analyze",
        "image_generate",
    }
    registry = ToolRegistry(safe_block=blocked_tools)

    def ok_handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({"output": "ok"})

    for tool_name in blocked_tools:
        registry.register(tool_name, ok_handler, enabled=True)

    registry.apply_safe_mode(True)

    for tool_name in blocked_tools:
        result = registry.call(ToolRequest(name=tool_name))
        assert not result.ok
        assert (
            "safe mode" in (result.error or "").lower()
            or "отключён" in (result.error or "").lower()
        )

    registry.apply_safe_mode(False)
    for tool_name in blocked_tools:
        result = registry.call(ToolRequest(name=tool_name))
        assert result.ok
