from __future__ import annotations

from shared.models import ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry


def test_safe_mode_blocks_web_shell() -> None:
    registry = ToolRegistry(safe_block={"web", "shell"})

    def ok_handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({"output": "ok"})

    registry.register("web", ok_handler, enabled=True)
    registry.register("shell", ok_handler, enabled=True)
    registry.apply_safe_mode(True)

    res_web = registry.call(ToolRequest(name="web"))
    assert not res_web.ok
    assert (
        "safe mode" in (res_web.error or "").lower() or "отключён" in (res_web.error or "").lower()
    )

    registry.apply_safe_mode(False)
    res_web2 = registry.call(ToolRequest(name="web"))
    assert res_web2.ok
