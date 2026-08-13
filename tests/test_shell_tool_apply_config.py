from __future__ import annotations

import config.shell_config as shell_config_module
from shared.models import ToolRequest
from tools.shell_tool import handle_shell_request


def test_shell_tool_applies_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(shell_config_module, "SHELL_CONFIG_DIR", config_dir)
    req = ToolRequest(
        name="shell",
        args={
            "command": "echo hi",
            "config_path": "shell_config.json",
            "shell_config": {
                "allowed_commands": ["echo"],
                "timeout_seconds": 1,
                "max_output_chars": 10,
                "sandbox_root": "sandbox",
            },
        },
    )
    result = handle_shell_request(req)
    assert result.ok
    assert (config_dir / "shell_config.json").exists()
