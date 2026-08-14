from __future__ import annotations

import json

import pytest

import config.shell_config as shell_config_module
from config.shell_config import ShellConfig, save_shell_config
from shared.models import ToolRequest
from tools.shell_tool import handle_shell_request


@pytest.fixture
def cfg_ctx(tmp_path, monkeypatch):
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    monkeypatch.setattr(shell_config_module, "SHELL_CONFIG_DIR", tmp_path)
    return tmp_path


def _request(**extra_args: object) -> ToolRequest:
    args: dict[str, object] = {"command": "echo hi"}
    args.update(extra_args)
    return ToolRequest(name="shell", args=args)


def test_shell_request_cannot_alter_allowed_commands(cfg_ctx) -> None:
    save_shell_config(ShellConfig(allowed_commands=["echo"], sandbox_root="sandbox"))
    payload = {"allowed_commands": ["ls"], "sandbox_root": "sandbox"}
    blocked = handle_shell_request(_request(command="ls", shell_config=payload))
    assert not blocked.ok
    assert "запрещ" in (blocked.error or "").lower()
    allowed = handle_shell_request(_request(shell_config=payload))
    assert allowed.ok
    assert "hi" in str(allowed.data.get("output"))


def test_shell_request_cannot_persist_modified_shell_config(cfg_ctx) -> None:
    canonical = cfg_ctx / "shell_config.json"
    canonical.write_text(
        json.dumps(
            {
                "allowed_commands": ["echo"],
                "timeout_seconds": 10,
                "max_output_chars": 6000,
                "sandbox_root": "sandbox",
            }
        ),
        encoding="utf-8",
    )
    original = canonical.read_text(encoding="utf-8")
    payload = {
        "allowed_commands": ["rm", "shutdown"],
        "timeout_seconds": 999,
        "max_output_chars": 999999,
        "sandbox_root": "sandbox",
    }
    res = handle_shell_request(_request(shell_config=payload))
    assert res.ok
    assert canonical.read_text(encoding="utf-8") == original


def test_shell_request_cannot_persist_config_when_absent(cfg_ctx) -> None:
    assert not (cfg_ctx / "shell_config.json").exists()
    payload = {"allowed_commands": ["echo"], "sandbox_root": "sandbox"}
    res = handle_shell_request(_request(shell_config=payload))
    assert res.ok
    assert not (cfg_ctx / "shell_config.json").exists()


def test_shell_request_ignores_payload_sandbox_root_override(cfg_ctx) -> None:
    save_shell_config(ShellConfig(allowed_commands=["echo"], sandbox_root="sandbox"))
    outside_dir = cfg_ctx / "outside_dir"
    payload = {"allowed_commands": ["echo"], "sandbox_root": str(outside_dir)}
    res = handle_shell_request(_request(shell_config=payload))
    assert res.ok
    assert not outside_dir.exists()


def test_normal_shell_execution_with_canonical_config(cfg_ctx) -> None:
    save_shell_config(ShellConfig(allowed_commands=["echo"], sandbox_root="sandbox"))
    res = handle_shell_request(_request())
    assert res.ok
    assert "hi" in str(res.data.get("output"))
