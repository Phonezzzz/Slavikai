from __future__ import annotations

import pytest

from shared.models import ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry


@pytest.mark.parametrize(
    "tool_name",
    [
        "web",
        "shell",
        "tts",
        "stt",
        "web_search",
        "http_client",
        "image_analyze",
        "image_generate",
    ],
)
def test_safe_mode_blocks_network_and_system_tools(tool_name: str) -> None:
    registry = ToolRegistry(
        safe_block={
            "web",
            "shell",
            "tts",
            "stt",
            "web_search",
            "http_client",
            "image_analyze",
            "image_generate",
        }
    )

    def ok_handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({"output": "ok"})

    registry.register(tool_name, ok_handler, enabled=True)
    registry.apply_safe_mode(True)

    result = registry.call(ToolRequest(name=tool_name))
    assert not result.ok
    assert "safe mode" in (result.error or "").lower() or "отключён" in (result.error or "").lower()

    registry.apply_safe_mode(False)
    result_enabled = registry.call(ToolRequest(name=tool_name))
    assert result_enabled.ok
