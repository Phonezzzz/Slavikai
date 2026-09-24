from __future__ import annotations

from pathlib import Path

from tools.terminal_tool import TerminalTool


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
