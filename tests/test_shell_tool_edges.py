from __future__ import annotations

from shared.models import ToolRequest
from tools.shell_tool import ShellConfig, handle_shell_request


def test_shell_allowed_command(tmp_path) -> None:
    cfg = ShellConfig(
        allowed_commands=["echo"],
        timeout_seconds=2,
        max_output_chars=100,
        sandbox_root=str(tmp_path),
    )
    req = ToolRequest(name="shell", args={"command": "echo hello", "shell_config": cfg.__dict__})
    res = handle_shell_request(req)
    assert res.ok
    assert "hello" in str(res.data.get("output"))


def test_shell_blocks_abs_path(tmp_path) -> None:
    cfg = ShellConfig(allowed_commands=["ls"], sandbox_root=str(tmp_path))
    req = ToolRequest(name="shell", args={"command": "/bin/ls", "shell_config": cfg.__dict__})
    res = handle_shell_request(req)
    assert not res.ok
    assert "запрещ" in (res.error or "").lower()


def test_shell_blocks_dangerous_and_chain(tmp_path) -> None:
    cfg = ShellConfig(allowed_commands=["ls"], sandbox_root=str(tmp_path))
    res_rm = handle_shell_request(
        ToolRequest(name="shell", args={"command": "rm -rf /", "shell_config": cfg.__dict__})
    )
    res_chain = handle_shell_request(
        ToolRequest(name="shell", args={"command": "ls; whoami", "shell_config": cfg.__dict__})
    )
    assert not res_rm.ok and not res_chain.ok
    assert "блок" in (res_rm.error or "").lower() or "опасн" in (res_rm.error or "").lower()
    assert "цепоч" in (res_chain.error or "").lower() or "запрещ" in (res_chain.error or "").lower()


def test_shell_timeout(tmp_path) -> None:
    cfg = ShellConfig(
        allowed_commands=["sleep"],
        timeout_seconds=1,
        max_output_chars=100,
        sandbox_root=str(tmp_path),
    )
    res = handle_shell_request(
        ToolRequest(name="shell", args={"command": "sleep 2", "shell_config": cfg.__dict__})
    )
    assert not res.ok
    assert "лимит" in (res.error or "").lower() or "timeout" in (res.error or "").lower()
