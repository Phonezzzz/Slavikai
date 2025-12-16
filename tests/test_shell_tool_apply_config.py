from __future__ import annotations

from shared.models import ToolRequest
from tools.shell_tool import handle_shell_request


def test_shell_tool_applies_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_path = tmp_path / "shell_config.json"
    req = ToolRequest(
        name="shell",
        args={
            "command": "echo hi",
            "config_path": str(config_path),
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
