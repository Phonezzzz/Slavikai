from __future__ import annotations

from shared.models import ToolRequest
from tools.shell_tool import handle_shell_request


def test_shell_tool_applies_config(tmp_path) -> None:
    req = ToolRequest(
        name="shell",
        args={
            "command": "echo hi",
            "shell_config": {
                "allowed_commands": ["echo"],
                "timeout_seconds": 1,
                "max_output_chars": 10,
                "sandbox_root": str(tmp_path),
            },
        },
    )
    result = handle_shell_request(req)
    assert result.ok
