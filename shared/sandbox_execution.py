"""Отказ в process execution до появления квалифицированной sandbox environment."""

from __future__ import annotations

from enum import StrEnum

from shared.models import ToolResult


class SandboxExecutionReason(StrEnum):
    UNAVAILABLE = "sandbox_execution_unavailable"


SANDBOX_EXECUTION_MESSAGE = (
    "Запуск недоступен: изолированная среда выполнения не подтверждена. "
    "Нужна настройка и проверка среды владельцем."
)


class SandboxExecutionUnavailable(RuntimeError):
    def __init__(self) -> None:
        super().__init__(SANDBOX_EXECUTION_MESSAGE)


def sandbox_execution_failure() -> ToolResult:
    return ToolResult.failure(
        SANDBOX_EXECUTION_MESSAGE,
        meta={"reason_code": SandboxExecutionReason.UNAVAILABLE.value, "execution_started": False},
    )
