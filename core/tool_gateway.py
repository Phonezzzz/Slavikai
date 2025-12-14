from __future__ import annotations

import logging
from dataclasses import dataclass

from shared.models import ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry

logger = logging.getLogger("SlavikAI.ToolGateway")


@dataclass
class ToolGateway:
    registry: ToolRegistry

    def call(self, request: ToolRequest) -> ToolResult:
        try:
            return self.registry.call(request)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка инструмента %s: %s", request.name, exc)
            return ToolResult.failure(f"Ошибка инструмента {request.name}: {exc}")
