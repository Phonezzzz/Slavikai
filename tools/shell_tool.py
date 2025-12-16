from __future__ import annotations

import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Final

from config.shell_config import (
    DEFAULT_SHELL_CONFIG_PATH,
    ShellConfig,
    load_shell_config,
    save_shell_config,
)
from shared.models import ToolRequest, ToolResult
from shared.sandbox import SandboxViolationError, normalize_shell_sandbox_root

DISALLOWED_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\brm\b\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*;\s*\}\s*;", re.IGNORECASE),  # fork bomb
    re.compile(r"\bsudo\b", re.IGNORECASE),
]

CHAIN_TOKENS: Final[set[str]] = {"&&", "||", ";"}


def _is_unsafe(command: str) -> bool:
    lowered = command.lower()
    if ">" in command and ("/etc" in command or "/dev" in command):
        return True
    return any(pattern.search(lowered) for pattern in DISALLOWED_PATTERNS)


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
    """
    Безопасный shell-инструмент:
    /sh <команда> — выполнит системную команду с фильтрацией.
    """
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
        sandbox_root = normalize_shell_sandbox_root(cfg.sandbox_root)
    except SandboxViolationError as exc:
        return ToolResult.failure(
            "sandbox_root должен оставаться внутри sandbox/.",
            meta={"raw": exc.raw_path, "normalized": str(exc.normalized_path)},
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Ошибка sandbox_root: {exc}")

    try:
        sandbox_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Не удалось подготовить sandbox_root: {exc}")

    try:
        started = time.monotonic()
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds,
            check=False,
            cwd=sandbox_root,
        )
        duration = time.monotonic() - started
        combined = (result.stdout or "") + (result.stderr or "")
        output = combined.strip() or "(пустой вывод)"
        if len(output) > cfg.max_output_chars:
            output = output[: cfg.max_output_chars] + "\n…[output truncated]"
        return ToolResult.success(
            {"output": output, "returncode": result.returncode},
            meta={
                "duration_sec": round(duration, 3),
                "cwd": str(sandbox_root),
            },
        )
    except subprocess.TimeoutExpired:
        return ToolResult.failure(
            "⏳ Команда превысила лимит времени.",
            {"timeout": cfg.timeout_seconds, "cwd": str(sandbox_root)},
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"Shell ошибка: {exc}")


def handle_shell_request(request: ToolRequest) -> ToolResult:
    cmd = str(request.args.get("command") or "").strip()
    config_path_raw = request.args.get("config_path")
    if config_path_raw is not None and not isinstance(config_path_raw, str):
        return ToolResult.failure("config_path должен быть строкой.")
    config_path = (
        Path(config_path_raw.strip())
        if isinstance(config_path_raw, str) and config_path_raw.strip()
        else DEFAULT_SHELL_CONFIG_PATH
    )
    if "shell_config" in request.args:
        # горячее применение настроек из UI
        cfg_payload = request.args.get("shell_config")
        if isinstance(cfg_payload, dict):
            try:
                allowed_raw = cfg_payload.get("allowed_commands")
                allowed_commands = (
                    [str(x) for x in allowed_raw] if isinstance(allowed_raw, list) else []
                )
                timeout_raw = cfg_payload.get("timeout_seconds", 10)
                max_out_raw = cfg_payload.get("max_output_chars", 6000)
                sandbox_raw = cfg_payload.get("sandbox_root", "sandbox")
                cfg = ShellConfig(
                    allowed_commands=allowed_commands,
                    timeout_seconds=int(timeout_raw),
                    max_output_chars=int(max_out_raw),
                    sandbox_root=str(sandbox_raw),
                )
                normalize_shell_sandbox_root(cfg.sandbox_root)
                save_shell_config(cfg, config_path)
                return handle_shell(cmd, config=cfg, config_path=config_path)
            except Exception as exc:  # noqa: BLE001
                return ToolResult.failure(f"Ошибка применения shell config: {exc}")
    return handle_shell(cmd, config_path=config_path)


class ShellTool:
    """Класс-обёртка для shell инструмента (Tool.handle совместимый)."""

    def handle(self, request: ToolRequest) -> ToolResult:  # noqa: D401 - очевидно
        return handle_shell_request(request)
