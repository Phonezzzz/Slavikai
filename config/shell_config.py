from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from shared.sandbox import SandboxViolationError, normalize_shell_sandbox_root

DEFAULT_SHELL_CONFIG_PATH = Path("config/shell_config.json")
SHELL_CONFIG_DIR = Path("config").resolve()


def normalize_shell_config_path(path: Path | None, config_dir: Path | None = None) -> Path:
    """Ограничивает config_path каталогом config/ (запрет абсолютных/../~ путей)."""
    if path is None:
        return DEFAULT_SHELL_CONFIG_PATH
    root = (config_dir or SHELL_CONFIG_DIR).resolve()
    raw_str = str(path).strip()
    if not raw_str:
        return DEFAULT_SHELL_CONFIG_PATH
    if raw_str.startswith(("~", "\\")):
        raise ValueError("config_path должен быть относительным путём внутри config/.")
    if any(part == ".." for part in Path(raw_str).parts):
        raise ValueError("config_path не может содержать '..'.")
    candidate = Path(raw_str)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"config_path выходит за пределы config/: {path}") from exc
    return candidate


@dataclass
class ShellConfig:
    allowed_commands: list[str] = field(
        default_factory=lambda: [
            "ls",
            "pwd",
            "cat",
            "head",
            "tail",
            "sed",
            "grep",
            "find",
            "python",
            "pytest",
            "rg",
            "echo",
        ]
    )
    timeout_seconds: int = 10
    max_output_chars: int = 6_000
    sandbox_root: str = "sandbox"


def load_shell_config(path: Path | None = None) -> ShellConfig:
    cfg_path = normalize_shell_config_path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists():
        return ShellConfig()
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        sandbox_root = str(data.get("sandbox_root", ShellConfig().sandbox_root))
        try:
            normalize_shell_sandbox_root(sandbox_root)
        except SandboxViolationError as exc:
            raise RuntimeError(
                f"Некорректный sandbox_root: {sandbox_root} ({exc.normalized_path})"
            ) from exc
        return ShellConfig(
            allowed_commands=list(data.get("allowed_commands", ShellConfig().allowed_commands)),
            timeout_seconds=int(data.get("timeout_seconds", ShellConfig().timeout_seconds)),
            max_output_chars=int(data.get("max_output_chars", ShellConfig().max_output_chars)),
            sandbox_root=sandbox_root,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ошибка загрузки shell_config.json: {exc}") from exc


def save_shell_config(config: ShellConfig, path: Path | None = None) -> None:
    cfg_path = normalize_shell_config_path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        normalize_shell_sandbox_root(config.sandbox_root)
    except SandboxViolationError as exc:
        raise RuntimeError(
            f"Некорректный sandbox_root: {config.sandbox_root} ({exc.normalized_path})"
        ) from exc
    data = {
        "allowed_commands": config.allowed_commands,
        "timeout_seconds": config.timeout_seconds,
        "max_output_chars": config.max_output_chars,
        "sandbox_root": config.sandbox_root,
    }
    cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
