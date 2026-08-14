from __future__ import annotations

import pytest

import config.shell_config as shell_config_module
from shared.models import ToolRequest
from tools.shell_tool import handle_shell_request


@pytest.fixture
def cfg_ctx(tmp_path, monkeypatch):
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    monkeypatch.setattr(shell_config_module, "SHELL_CONFIG_DIR", tmp_path)
    return tmp_path


def test_shell_tool_applies_config(cfg_ctx) -> None:
    req = ToolRequest(
        name="shell",
        args={
            "command": "echo hi",
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
    assert (cfg_ctx / "shell_config.json").exists()


def test_shell_tool_apply_writes_only_canonical_file(cfg_ctx) -> None:
    other = cfg_ctx / "tools_config.py"
    other.write_text("ALLOWED = []", encoding="utf-8")
    req = ToolRequest(
        name="shell",
        args={
            "command": "echo hi",
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
    assert (cfg_ctx / "shell_config.json").exists()
    assert other.read_text(encoding="utf-8") == "ALLOWED = []"
