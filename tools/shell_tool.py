from __future__ import annotations

import shlex
from pathlib import Path
from typing import Final

from config.shell_config import (
    DEFAULT_SHELL_CONFIG_PATH,
    ShellConfig,
    load_shell_config,
)
from shared.command_safety import is_hard_unsafe_command as _is_unsafe
from shared.models import ToolRequest, ToolResult
from shared.sandbox import SandboxViolationError, normalize_shell_sandbox_root
from shared.sandbox_execution import sandbox_execution_failure

CHAIN_TOKENS: Final[set[str]] = {"&&", "||", ";"}


def _validate_args(args: list[str], allowed_commands: set[str]) -> str | None:
    if not args:
        return "Команда пуста."

    command_name = args[0]
    if command_name not in allowed_commands:
        return f"Команда '{command_name}' запрещена политикой shell."

    if any(token in CHAIN_TOKENS for token in args):
        return "Командные цепочки запрещены."

    for arg in args[1:]:
        if arg.startswith("/"):
            return "Абсолютные пути запрещены в shell-инструменте."
        if ".." in arg:
            return "Выход за пределы песочницы запрещён."

    return None


def handle_shell(
    command: str, config: ShellConfig | None = None, config_path: Path | None = None
) -> ToolResult:
    """Проверяет shell policy и возвращает отказ до появления isolated executor."""
    if not command.strip():
        return ToolResult.failure("Команда пуста.")

    if _is_unsafe(command):
        return ToolResult.failure("🚫 Опасная команда заблокирована.")

    try:
        cfg = config or load_shell_config(config_path or DEFAULT_SHELL_CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Ошибка загрузки shell config: {exc}")

    allowed_commands = set(cfg.allowed_commands)
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return ToolResult.failure(f"Ошибка парсинга команды: {exc}")

    validation_error = _validate_args(args, allowed_commands)
    if validation_error:
        return ToolResult.failure(validation_error)

    try:
        normalize_shell_sandbox_root(cfg.sandbox_root)
    except SandboxViolationError as exc:
        return ToolResult.failure(
            "sandbox_root должен оставаться внутри sandbox/.",
            meta={"raw": exc.raw_path, "normalized": str(exc.normalized_path)},
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Ошибка sandbox_root: {exc}")

    return sandbox_execution_failure()


def handle_shell_request(request: ToolRequest) -> ToolResult:
    cmd = str(request.args.get("command") or "").strip()
    return handle_shell(cmd)


class ShellTool:
    """Класс-обёртка для shell инструмента (Tool.handle совместимый)."""

    def handle(self, request: ToolRequest) -> ToolResult:  # noqa: D401 - очевидно
        return handle_shell_request(request)
