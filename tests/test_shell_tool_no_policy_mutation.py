from __future__ import annotations

from config.shell_config import ShellConfig, save_shell_config
from shared.models import ToolRequest
from tools.shell_tool import handle_shell_request
from tools.tool_descriptors import get_tool_metadata


def test_shell_request_cannot_replace_canonical_policy(tmp_path, monkeypatch) -> None:
    sandbox_root = tmp_path / "sandbox"
    canonical_path = tmp_path / "canonical-shell-config.json"
    attacker_path = tmp_path / "request-selected-shell-config.json"
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", sandbox_root)
    monkeypatch.setattr("tools.shell_tool.DEFAULT_SHELL_CONFIG_PATH", canonical_path)
    save_shell_config(
        ShellConfig(allowed_commands=["echo"], sandbox_root="sandbox"),
        canonical_path,
    )
    original = canonical_path.read_text(encoding="utf-8")

    result = handle_shell_request(
        ToolRequest(
            name="shell",
            args={
                "command": "ls",
                "config_path": str(attacker_path),
                "shell_config": {
                    "allowed_commands": ["ls"],
                    "sandbox_root": "sandbox",
                },
            },
        )
    )

    assert not result.ok
    assert "запрещ" in (result.error or "").lower()
    assert canonical_path.read_text(encoding="utf-8") == original
    assert not attacker_path.exists()


def test_shell_descriptor_exposes_execution_only() -> None:
    metadata = get_tool_metadata("shell")

    assert metadata.parameters_schema["properties"] == {
        "command": {
            "type": "string",
            "description": "Single shell command allowed by shell policy.",
        }
    }
    assert metadata.parameters_schema["additionalProperties"] is False
