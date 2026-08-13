from __future__ import annotations

from pathlib import Path

import pytest

import config.shell_config as shell_config_module
from shared.models import ToolRequest
from tools.shell_tool import ShellConfig, handle_shell_request

CFG = {
    "allowed_commands": ["echo"],
    "timeout_seconds": 2,
    "max_output_chars": 100,
    "sandbox_root": "sandbox",
}


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    config_dir = tmp_path / "cfg"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(shell_config_module, "SHELL_CONFIG_DIR", config_dir)
    return config_dir


def _request(config_path: str | None) -> ToolRequest:
    args: dict[str, object] = {
        "command": "echo hi",
        "shell_config": dict(CFG),
    }
    if config_path is not None:
        args["config_path"] = config_path
    return ToolRequest(name="shell", args=args)


def test_config_path_absolute_rejected(tmp_path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)
    outside = tmp_path / "outside.json"
    res = handle_shell_request(_request(str(outside)))
    assert not res.ok
    assert "config/" in (res.error or "")
    assert not outside.exists()


def test_config_path_parent_reference_rejected(tmp_path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)
    res = handle_shell_request(_request("../escaped.json"))
    assert not res.ok
    assert "config/" in (res.error or "") or ".." in (res.error or "")
    assert not (tmp_path / "escaped.json").exists()


def test_config_path_tilde_rejected(tmp_path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)
    res = handle_shell_request(_request("~/evil.json"))
    assert not res.ok
    assert "config/" in (res.error or "")


def test_config_path_symlink_escape_rejected(tmp_path, monkeypatch) -> None:
    config_dir = _prepare(tmp_path, monkeypatch)
    outside = tmp_path / "outside_dir"
    outside.mkdir(parents=True, exist_ok=True)
    try:
        (config_dir / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink недоступен в этом окружении.")
    res = handle_shell_request(_request("link/evil.json"))
    assert not res.ok
    assert not (outside / "evil.json").exists()


def test_config_path_within_config_dir_applies(tmp_path, monkeypatch) -> None:
    config_dir = _prepare(tmp_path, monkeypatch)
    res = handle_shell_request(_request("nested/shell_config.json"))
    assert res.ok
    assert (config_dir / "nested" / "shell_config.json").exists()


def test_save_shell_config_rejects_outside_path(tmp_path, monkeypatch) -> None:
    _prepare(tmp_path, monkeypatch)

    outside = tmp_path / "outside.json"
    with pytest.raises(ValueError):
        shell_config_module.save_shell_config(
            ShellConfig(allowed_commands=["echo"]),
            outside,
        )
    assert not outside.exists()
