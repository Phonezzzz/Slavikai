from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from shared.models import JSONValue, ToolCallRecord, ToolRequest, ToolResult
from tools.protocols import Tool
from tools.tool_logger import ToolCallLogger

ToolHandler = Callable[[ToolRequest], ToolResult]


@dataclass
class ToolDescriptor:
    name: str
    handler: ToolHandler
    enabled: bool = True


class ToolRegistry:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        call_logger: ToolCallLogger | None = None,
        safe_block: set[str] | None = None,
    ) -> None:
        self._tools: dict[str, ToolDescriptor] = {}
        self._logger = logger or logging.getLogger("SlavikAI.ToolRegistry")
        self._call_logger = call_logger or ToolCallLogger()
        self._safe_mode = False
        self._safe_block = safe_block or set()

    def register(self, name: str, handler: Tool | ToolHandler, enabled: bool = True) -> None:
        resolved: ToolHandler
        if isinstance(handler, Tool):
            resolved = handler.handle
        else:
            resolved = handler
        self._tools[name] = ToolDescriptor(name=name, handler=resolved, enabled=enabled)
        self._logger.info("tool_registered", extra={"tool": name, "enabled": enabled})

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name in self._tools:
            self._tools[name].enabled = enabled
            self._logger.info("tool_enabled", extra={"tool": name, "enabled": enabled})

    def is_enabled(self, name: str) -> bool:
        descriptor = self._tools.get(name)
        return bool(descriptor and descriptor.enabled)

    def call(self, request: ToolRequest, *, bypass_safe_mode: bool = False) -> ToolResult:
        descriptor = self._tools.get(request.name)
        if not descriptor:
            self._logger.warning("tool_not_found", extra={"tool": request.name})
            self._log_call(
                request.name,
                ok=False,
                error="Инструмент не зарегистрирован",
                args=request.args,
            )
            return ToolResult.failure(f"Инструмент {request.name} не зарегистрирован")

        if self._safe_mode and request.name in self._safe_block and not bypass_safe_mode:
            self._logger.info("tool_safe_blocked", extra={"tool": request.name})
            self._log_call(
                request.name,
                ok=False,
                error="Safe mode: инструмент отключён",
                args=request.args,
            )
            return ToolResult.failure("Safe mode: инструмент отключён")

        if not descriptor.enabled:
            self._logger.info("tool_disabled_call", extra={"tool": request.name})
            self._log_call(request.name, ok=False, error="Инструмент отключён", args=request.args)
            return ToolResult.failure(f"Инструмент {request.name} отключён")

        self._logger.info("tool_call_start", extra={"tool": request.name})
        try:
            result = descriptor.handler(request)
            self._logger.info(
                "tool_call_end",
                extra={"tool": request.name, "ok": result.ok, "error": result.error},
            )
            self._log_call(
                request.name,
                ok=result.ok,
                error=result.error,
                meta=result.meta,
                args=request.args,
            )
            return result
        except Exception as exc:
            self._logger.exception("tool_call_error", extra={"tool": request.name})
            self._log_call(request.name, ok=False, error=str(exc), args=request.args)
            return ToolResult.failure(f"Ошибка инструмента {request.name}: {exc}")

    def list_tools(self) -> dict[str, bool]:
        return {name: desc.enabled for name, desc in self._tools.items()}

    def read_recent_calls(self, limit: int = 50) -> list[ToolCallRecord]:
        return self._call_logger.read_recent(limit)

    def _log_call(
        self,
        tool: str,
        ok: bool,
        error: str | None = None,
        meta: dict[str, JSONValue] | None = None,
        args: dict[str, JSONValue] | None = None,
    ) -> None:
        try:
            self._call_logger.log(tool, ok=ok, error=error, meta=meta, args=args)
        except Exception:  # noqa: BLE001
            self._logger.debug("failed to write tool call log", exc_info=True)

    def apply_safe_mode(self, enabled: bool) -> None:
        if enabled:
            if self._safe_mode:
                self._logger.info("safe_mode_enabled")
                return
            self._safe_mode = True
            self._logger.info("safe_mode_enabled")
            return
        if not self._safe_mode:
            self._logger.info("safe_mode_disabled")
            return
        self._safe_mode = False
        self._logger.info("safe_mode_disabled")
