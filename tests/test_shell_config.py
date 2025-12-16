from __future__ import annotations

from config.shell_config import ShellConfig
from tools.shell_tool import handle_shell


def test_shell_respects_custom_allowed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    cfg = ShellConfig(
        allowed_commands=["echo"],
        timeout_seconds=2,
        max_output_chars=100,
        sandbox_root="sandbox",
    )
    result = handle_shell("echo hi", config=cfg)
    assert result.ok


def test_shell_blocks_unlisted_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    cfg = ShellConfig(
        allowed_commands=["echo"],
        timeout_seconds=2,
        max_output_chars=100,
        sandbox_root="sandbox",
    )
    result = handle_shell("ls", config=cfg)
    assert not result.ok
