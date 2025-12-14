from __future__ import annotations

from typing import Protocol, runtime_checkable

from shared.models import ToolRequest, ToolResult


@runtime_checkable
class Tool(Protocol):
    def handle(self, request: ToolRequest) -> ToolResult: ...
