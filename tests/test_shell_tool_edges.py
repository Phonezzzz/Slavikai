from __future__ import annotations

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


def _request(command: str) -> ToolRequest:
    return ToolRequest(name="shell", args={"command": command})


def test_shell_allowed_command(cfg_ctx) -> None:
    save_shell_config(
        ShellConfig(
            allowed_commands=["echo"],
            timeout_seconds=2,
            max_output_chars=100,
            sandbox_root="sandbox",
        )
    )
    res = handle_shell_request(_request("echo hello"))
    assert res.ok
    assert "hello" in str(res.data.get("output"))


def test_shell_blocks_abs_path(cfg_ctx) -> None:
    save_shell_config(ShellConfig(allowed_commands=["ls"], sandbox_root="sandbox"))
    res = handle_shell_request(_request("/bin/ls"))
    assert not res.ok
    assert "запрещ" in (res.error or "").lower()


def test_shell_blocks_dangerous_and_chain(cfg_ctx) -> None:
    save_shell_config(ShellConfig(allowed_commands=["ls"], sandbox_root="sandbox"))
    res_rm = handle_shell_request(_request("rm -rf /"))
    res_chain = handle_shell_request(_request("ls; whoami"))
    assert not res_rm.ok and not res_chain.ok
    assert "блок" in (res_rm.error or "").lower() or "опасн" in (res_rm.error or "").lower()
    assert "цепоч" in (res_chain.error or "").lower() or "запрещ" in (res_chain.error or "").lower()


def test_shell_timeout(cfg_ctx) -> None:
    save_shell_config(
        ShellConfig(
            allowed_commands=["sleep"],
            timeout_seconds=1,
            max_output_chars=100,
            sandbox_root="sandbox",
        )
    )
    res = handle_shell_request(_request("sleep 2"))
    assert not res.ok
    assert "лимит" in (res.error or "").lower() or "timeout" in (res.error or "").lower()


def test_shell_rejects_absolute_sandbox_root(cfg_ctx) -> None:
    outside_dir = cfg_ctx / "outside_dir"
    assert not outside_dir.exists()

    canonical = cfg_ctx / "shell_config.json"
    canonical.write_text(
        '{"allowed_commands":["echo"],"timeout_seconds":2,'
        f'"max_output_chars":100,"sandbox_root":"{outside_dir}"}}',
        encoding="utf-8",
    )
    res = handle_shell_request(_request("echo hi"))
    assert not res.ok
    assert not outside_dir.exists()


def test_shell_rejects_parent_reference_sandbox_root(cfg_ctx) -> None:
    outside_dir = cfg_ctx / "outside_dir"
    assert not outside_dir.exists()

    canonical = cfg_ctx / "shell_config.json"
    canonical.write_text(
        '{"allowed_commands":["echo"],"timeout_seconds":2,'
        '"max_output_chars":100,"sandbox_root":"../outside_dir"}',
        encoding="utf-8",
    )
    res = handle_shell_request(_request("echo hi"))
    assert not res.ok
    assert not outside_dir.exists()


def test_shell_uses_defaults_when_config_missing(cfg_ctx) -> None:
    assert not (cfg_ctx / "shell_config.json").exists()

    res = handle_shell_request(_request("echo hi"))
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

    canonical = cfg_ctx / "shell_config.json"
    canonical.write_text(
        '{"allowed_commands":["echo"],"timeout_seconds":2,'
        '"max_output_chars":100,"sandbox_root":"escape"}',
        encoding="utf-8",
    )
    res = handle_shell_request(_request("echo hi"))
    assert not res.ok
    assert "некорректный sandbox_root" in (res.error or "").lower()
