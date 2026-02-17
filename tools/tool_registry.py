from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from shared.models import JSONValue, ToolCallRecord, ToolRequest, ToolResult
from tools.protocols import Tool
from tools.tool_logger import ToolCallLogger

ToolHandler = Callable[[ToolRequest], ToolResult]
ToolCapability = Literal["read", "write", "exec"]


@dataclass
class ToolDescriptor:
    name: str
    handler: ToolHandler
    enabled: bool = True
    capability: ToolCapability = "exec"


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
        self._mode: Literal["ask", "plan", "act"] = "act"
        self._active_plan: dict[str, JSONValue] | None = None
        self._active_task: dict[str, JSONValue] | None = None
        self._enforce_plan_guard = False

    def register(
        self,
        name: str,
        handler: Tool | ToolHandler,
        enabled: bool = True,
        capability: ToolCapability = "exec",
    ) -> None:
        resolved: ToolHandler
        if isinstance(handler, Tool):
            resolved = handler.handle
        else:
            resolved = handler
        self._tools[name] = ToolDescriptor(
            name=name,
            handler=resolved,
            enabled=enabled,
            capability=self._normalize_capability(capability),
        )
        self._logger.info(
            "tool_registered",
            extra={"tool": name, "enabled": enabled, "capability": capability},
        )

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name in self._tools:
            self._tools[name].enabled = enabled
            self._logger.info("tool_enabled", extra={"tool": name, "enabled": enabled})

    def is_enabled(self, name: str) -> bool:
        descriptor = self._tools.get(name)
        return bool(descriptor and descriptor.enabled)

    def get_capability(self, name: str) -> ToolCapability | None:
        descriptor = self._tools.get(name)
        if descriptor is None:
            return None
        return descriptor.capability

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

        mode_error = self._mode_policy_error(request.name, descriptor.capability)
        if mode_error is not None:
            self._logger.info("tool_mode_blocked", extra={"tool": request.name, "mode": self._mode})
            self._log_call(request.name, ok=False, error=mode_error, args=request.args)
            return ToolResult.failure(mode_error)

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

    def set_execution_policy(
        self,
        *,
        mode: str,
        active_plan: dict[str, JSONValue] | None = None,
        active_task: dict[str, JSONValue] | None = None,
        enforce_plan_guard: bool = False,
    ) -> None:
        self._mode = self._normalize_mode(mode)
        self._active_plan = dict(active_plan) if isinstance(active_plan, dict) else None
        self._active_task = dict(active_task) if isinstance(active_task, dict) else None
        self._enforce_plan_guard = bool(enforce_plan_guard)

    def _normalize_mode(self, mode: str) -> Literal["ask", "plan", "act"]:
        normalized = mode.strip().lower()
        if normalized == "plan":
            return "plan"
        if normalized == "act":
            return "act"
        return "ask"

    def _normalize_capability(self, capability: str) -> ToolCapability:
        normalized = capability.strip().lower()
        if normalized == "read":
            return "read"
        if normalized == "write":
            return "write"
        return "exec"

    def _mode_policy_error(
        self,
        tool_name: str,
        capability: ToolCapability,
    ) -> str | None:
        if self._mode == "plan" and capability != "read":
            return "PLAN_READ_ONLY_BLOCK: plan-режим допускает только read-only инструменты."
        if self._mode != "act" or not self._enforce_plan_guard:
            return None
        return self._plan_guard_error(tool_name)

    def _plan_guard_error(self, tool_name: str) -> str | None:
        active_plan = self._active_plan
        active_task = self._active_task
        if not isinstance(active_plan, dict) or not isinstance(active_task, dict):
            return "BLOCKED_OUTSIDE_PLAN: отсутствует активный план/таск."
        plan_id = active_plan.get("plan_id")
        plan_hash = active_plan.get("plan_hash")
        task_plan_id = active_task.get("plan_id")
        task_plan_hash = active_task.get("plan_hash")
        if (
            not isinstance(plan_id, str)
            or not isinstance(task_plan_id, str)
            or plan_id != task_plan_id
        ):
            return "BLOCKED_OUTSIDE_PLAN: plan_id не совпадает с active_task."
        if (
            not isinstance(plan_hash, str)
            or not isinstance(task_plan_hash, str)
            or plan_hash != task_plan_hash
        ):
            return "BLOCKED_OUTSIDE_PLAN: plan_hash не совпадает с active_task."
        current_step_id = active_task.get("current_step_id")
        if not isinstance(current_step_id, str) or not current_step_id.strip():
            return "BLOCKED_OUTSIDE_PLAN: current_step_id отсутствует."
        steps_raw = active_plan.get("steps")
        if not isinstance(steps_raw, list):
            return "BLOCKED_OUTSIDE_PLAN: steps не определены."
        step_payload: dict[str, JSONValue] | None = None
        for item in steps_raw:
            if not isinstance(item, dict):
                continue
            step_id = item.get("step_id")
            if isinstance(step_id, str) and step_id == current_step_id:
                step_payload = item
                break
        if step_payload is None:
            return "BLOCKED_OUTSIDE_PLAN: текущий шаг не найден в плане."
        allowed_raw = step_payload.get("allowed_tool_kinds")
        if not isinstance(allowed_raw, list):
            return "BLOCKED_OUTSIDE_PLAN: allowed_tool_kinds отсутствует."
        allowed = {item.strip() for item in allowed_raw if isinstance(item, str) and item.strip()}
        if tool_name not in allowed:
            return f"BLOCKED_OUTSIDE_PLAN: tool '{tool_name}' не разрешён в текущем шаге."
        return None
