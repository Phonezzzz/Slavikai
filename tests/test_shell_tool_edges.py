from __future__ import annotations

import pytest

import config.shell_config as shell_config_module
from shared.models import ToolRequest
from tools.shell_tool import ShellConfig, handle_shell_request


@pytest.fixture
def cfg_ctx(tmp_path, monkeypatch):
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    monkeypatch.setattr(shell_config_module, "SHELL_CONFIG_DIR", tmp_path)
    return tmp_path


def test_shell_allowed_command(cfg_ctx) -> None:
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
        },
    )
    res = handle_shell_request(req)
    assert res.ok
    assert "hello" in str(res.data.get("output"))


def test_shell_blocks_abs_path(cfg_ctx) -> None:
    cfg = ShellConfig(allowed_commands=["ls"], sandbox_root="sandbox")
    req = ToolRequest(
        name="shell",
        args={"command": "/bin/ls", "shell_config": cfg.__dict__},
    )
    res = handle_shell_request(req)
    assert not res.ok
    assert "запрещ" in (res.error or "").lower()


def test_shell_blocks_dangerous_and_chain(cfg_ctx) -> None:
    cfg = ShellConfig(allowed_commands=["ls"], sandbox_root="sandbox")
    res_rm = handle_shell_request(
        ToolRequest(
            name="shell",
            args={
                "command": "rm -rf /",
                "shell_config": cfg.__dict__,
            },
        )
    )
    res_chain = handle_shell_request(
        ToolRequest(
            name="shell",
            args={
                "command": "ls; whoami",
                "shell_config": cfg.__dict__,
            },
        )
    )
    assert not res_rm.ok and not res_chain.ok
    assert "блок" in (res_rm.error or "").lower() or "опасн" in (res_rm.error or "").lower()
    assert "цепоч" in (res_chain.error or "").lower() or "запрещ" in (res_chain.error or "").lower()


def test_shell_timeout(cfg_ctx) -> None:
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
            },
        )
    )
    assert not res.ok
    assert "лимит" in (res.error or "").lower() or "timeout" in (res.error or "").lower()


def test_shell_rejects_absolute_sandbox_root(cfg_ctx) -> None:
    outside_dir = cfg_ctx / "outside_dir"
    assert not outside_dir.exists()

    cfg = ShellConfig(
        allowed_commands=["echo"],
        timeout_seconds=2,
        max_output_chars=100,
        sandbox_root=str(outside_dir),
    )
    req = ToolRequest(
        name="shell",
        args={
            "command": "echo hi",
            "shell_config": cfg.__dict__,
        },
    )
    res = handle_shell_request(req)
    assert not res.ok
    assert not outside_dir.exists()


def test_shell_rejects_parent_reference_sandbox_root(cfg_ctx) -> None:
    outside_dir = cfg_ctx / "outside_dir"
    assert not outside_dir.exists()

    cfg = ShellConfig(
        allowed_commands=["echo"],
        timeout_seconds=2,
        max_output_chars=100,
        sandbox_root="../outside_dir",
    )
    req = ToolRequest(
        name="shell",
        args={
            "command": "echo hi",
            "shell_config": cfg.__dict__,
        },
    )
    res = handle_shell_request(req)
    assert not res.ok
    assert not outside_dir.exists()


def test_shell_uses_defaults_when_config_missing(cfg_ctx) -> None:
    assert not (cfg_ctx / "shell_config.json").exists()

    res = handle_shell_request(ToolRequest(name="shell", args={"command": "echo hi"}))
    assert res.ok
    assert "hi" in str(res.data.get("output"))
    assert not (cfg_ctx / "shell_config.json").exists()


def test_normalize_shell_sandbox_root_empty_uses_root(cfg_ctx) -> None:
    from shared.sandbox import normalize_shell_sandbox_root

    assert normalize_shell_sandbox_root("") == (cfg_ctx / "sandbox").resolve()


def test_normalize_shell_sandbox_root_rejects_symlink_escape(cfg_ctx) -> None:
    sandbox_root = (cfg_ctx / "sandbox").resolve()
    sandbox_root.mkdir(parents=True)
    outside = (cfg_ctx / "outside").resolve()
    outside.mkdir()
    link = sandbox_root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink недоступен в этом окружении.")

    from shared.sandbox import SandboxViolationError, normalize_shell_sandbox_root

    with pytest.raises(SandboxViolationError):
        normalize_shell_sandbox_root("escape")


def test_shell_rejects_symlink_sandbox_root(cfg_ctx) -> None:
    sandbox_root = (cfg_ctx / "sandbox").resolve()
    sandbox_root.mkdir(parents=True)
    outside = (cfg_ctx / "outside").resolve()
    outside.mkdir()
    link = sandbox_root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink недоступен в этом окружении.")

    cfg = ShellConfig(
        allowed_commands=["echo"],
        timeout_seconds=2,
        max_output_chars=100,
        sandbox_root="escape",
    )
    req = ToolRequest(
        name="shell",
        args={
            "command": "echo hi",
            "shell_config": cfg.__dict__,
        },
    )
    res = handle_shell_request(req)
    assert not res.ok
    assert "sandbox violation" in (res.error or "").lower()
