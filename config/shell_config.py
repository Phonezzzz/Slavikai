from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SHELL_CONFIG_PATH = Path("config/shell_config.json")


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
    cfg_path = path or DEFAULT_SHELL_CONFIG_PATH
    if not cfg_path.exists():
        return ShellConfig()
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return ShellConfig(
            allowed_commands=list(data.get("allowed_commands", ShellConfig().allowed_commands)),
            timeout_seconds=int(data.get("timeout_seconds", ShellConfig().timeout_seconds)),
            max_output_chars=int(data.get("max_output_chars", ShellConfig().max_output_chars)),
            sandbox_root=str(data.get("sandbox_root", ShellConfig().sandbox_root)),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ошибка загрузки shell_config.json: {exc}") from exc


def save_shell_config(config: ShellConfig, path: Path | None = None) -> None:
    cfg_path = path or DEFAULT_SHELL_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "allowed_commands": config.allowed_commands,
        "timeout_seconds": config.timeout_seconds,
        "max_output_chars": config.max_output_chars,
        "sandbox_root": config.sandbox_root,
    }
    cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
