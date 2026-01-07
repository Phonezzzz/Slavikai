from __future__ import annotations

from core.tool_gateway import ToolGateway
from shared.models import ToolRequest, ToolResult


class _OkRegistry:
    def __init__(self) -> None:
        self.called = False

    def call(
        self, request: ToolRequest, *, bypass_safe_mode: bool = False
    ) -> ToolResult:
        self.called = True
        return ToolResult.success({"value": "ok"})


class _BoomRegistry:
    def call(
        self, request: ToolRequest, *, bypass_safe_mode: bool = False
    ) -> ToolResult:
        raise RuntimeError("boom")


def test_tool_gateway_passes_through_result() -> None:
    registry = _OkRegistry()
    gateway = ToolGateway(registry=registry)  # type: ignore[arg-type]
    result = gateway.call(ToolRequest(name="demo"))
    assert result.ok
    assert result.data.get("value") == "ok"
    assert registry.called


def test_tool_gateway_handles_registry_exception() -> None:
    gateway = ToolGateway(registry=_BoomRegistry())  # type: ignore[arg-type]
    result = gateway.call(ToolRequest(name="demo"))
    assert not result.ok
    assert "demo" in (result.error or "")
