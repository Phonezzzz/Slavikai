from __future__ import annotations

from pathlib import Path

import pytest

from config.shell_config import ShellConfig
from tools.terminal_tool import TerminalTool


@pytest.mark.parametrize("exit_code", [0, 7])
def test_terminal_oneshot_does_not_claim_unexecuted_script_output(
    tmp_path, monkeypatch, exit_code
) -> None:
    script = tmp_path / "result.py"
    script.write_text(
        f"import sys\nprint('before failure')\nprint('diagnostic', file=sys.stderr)\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tools.terminal_tool.load_shell_config", lambda _: ShellConfig(allowed_commands=["python"])
    )
    result = TerminalTool().run_oneshot(
        command="python result.py",
        cwd_mode="session_root",
        workspace_root=tmp_path,
        sandbox_root=tmp_path,
    )

    assert not result.ok
    assert result.meta["reason_code"] == "sandbox_execution_unavailable"
    assert "exit_code" not in result.data


def test_terminal_oneshot_blocks_absolute_paths() -> None:
    tool = TerminalTool()
    result = tool.run_oneshot(
        command="echo /etc/hostname",
        cwd_mode="session_root",
        workspace_root=Path("sandbox/project"),
        sandbox_root=Path("sandbox"),
    )
    assert not result.ok
    assert "Абсолютные пути" in (result.error or "")


def test_terminal_oneshot_blocks_parent_traversal() -> None:
    tool = TerminalTool()
    result = tool.run_oneshot(
        command="echo ../secret.txt",
        cwd_mode="session_root",
        workspace_root=Path("sandbox/project"),
        sandbox_root=Path("sandbox"),
    )
    assert not result.ok
    assert "песочницы" in (result.error or "")


def test_terminal_oneshot_blocks_disallowed_command() -> None:
    tool = TerminalTool()
    result = tool.run_oneshot(
        command="python -c 'print(1)'",
        cwd_mode="session_root",
        workspace_root=Path("sandbox/project"),
        sandbox_root=Path("sandbox"),
    )
    assert not result.ok
    assert "запрещена политикой shell" in (result.error or "")
