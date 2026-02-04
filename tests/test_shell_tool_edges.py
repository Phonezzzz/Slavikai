from __future__ import annotations

from shared.models import ToolRequest
from tools.shell_tool import ShellConfig, handle_shell_request


def test_shell_allowed_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_path = tmp_path / "shell_config.json"
    cfg = ShellConfig(
        allowed_commands=["echo"],
        timeout_seconds=2,
        max_output_chars=100,
        sandbox_root="sandbox",
    )
    req = ToolRequest(
        name="shell",
        args={
            "command": "echo hello",
            "shell_config": cfg.__dict__,
            "config_path": str(config_path),
        },
    )
    res = handle_shell_request(req)
    assert res.ok
    assert "hello" in str(res.data.get("output"))


def test_shell_blocks_abs_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_path = tmp_path / "shell_config.json"
    cfg = ShellConfig(allowed_commands=["ls"], sandbox_root="sandbox")
    req = ToolRequest(
        name="shell",
        args={"command": "/bin/ls", "shell_config": cfg.__dict__, "config_path": str(config_path)},
    )
    res = handle_shell_request(req)
    assert not res.ok
    assert "запрещ" in (res.error or "").lower()


def test_shell_blocks_dangerous_and_chain(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_path = tmp_path / "shell_config.json"
    cfg = ShellConfig(allowed_commands=["ls"], sandbox_root="sandbox")
    res_rm = handle_shell_request(
        ToolRequest(
            name="shell",
            args={
                "command": "rm -rf /",
                "shell_config": cfg.__dict__,
                "config_path": str(config_path),
            },
        )
    )
    res_chain = handle_shell_request(
        ToolRequest(
            name="shell",
            args={
                "command": "ls; whoami",
                "shell_config": cfg.__dict__,
                "config_path": str(config_path),
            },
        )
    )
    assert not res_rm.ok and not res_chain.ok
    assert "блок" in (res_rm.error or "").lower() or "опасн" in (res_rm.error or "").lower()
    assert "цепоч" in (res_chain.error or "").lower() or "запрещ" in (res_chain.error or "").lower()


def test_shell_timeout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_path = tmp_path / "shell_config.json"
    cfg = ShellConfig(
        allowed_commands=["sleep"],
        timeout_seconds=1,
        max_output_chars=100,
        sandbox_root="sandbox",
    )
    res = handle_shell_request(
        ToolRequest(
            name="shell",
            args={
                "command": "sleep 2",
                "shell_config": cfg.__dict__,
                "config_path": str(config_path),
            },
        )
    )
    assert not res.ok
    assert "лимит" in (res.error or "").lower() or "timeout" in (res.error or "").lower()


def test_shell_rejects_absolute_sandbox_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_path = tmp_path / "shell_config.json"
    outside_dir = tmp_path / "outside_dir"
    assert not outside_dir.exists()

    cfg = ShellConfig(
        allowed_commands=["echo"],
        timeout_seconds=2,
        max_output_chars=100,
        sandbox_root=str(outside_dir),
    )
    req = ToolRequest(
        name="shell",
        args={"command": "echo hi", "shell_config": cfg.__dict__, "config_path": str(config_path)},
    )
    res = handle_shell_request(req)
    assert not res.ok
    assert not outside_dir.exists()


def test_shell_rejects_parent_reference_sandbox_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_path = tmp_path / "shell_config.json"
    outside_dir = tmp_path / "outside_dir"
    assert not outside_dir.exists()

    cfg = ShellConfig(
        allowed_commands=["echo"],
        timeout_seconds=2,
        max_output_chars=100,
        sandbox_root="../outside_dir",
    )
    req = ToolRequest(
        name="shell",
        args={"command": "echo hi", "shell_config": cfg.__dict__, "config_path": str(config_path)},
    )
    res = handle_shell_request(req)
    assert not res.ok
    assert not outside_dir.exists()


def test_shell_uses_defaults_when_config_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_path = tmp_path / "missing_shell_config.json"
    assert not config_path.exists()

    res = handle_shell_request(
        ToolRequest(name="shell", args={"command": "echo hi", "config_path": str(config_path)})
    )
    assert res.ok
    assert "hi" in str(res.data.get("output"))
    assert not config_path.exists()


def test_normalize_shell_sandbox_root_empty_uses_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    from shared.sandbox import normalize_shell_sandbox_root

    assert normalize_shell_sandbox_root("") == (tmp_path / "sandbox").resolve()
