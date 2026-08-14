from __future__ import annotations

from pathlib import Path

import pytest

import config.shell_config as shell_config_module
from config.shell_config import ShellConfig, load_shell_config, save_shell_config
from shared.models import ToolRequest
from tools.shell_tool import handle_shell_request


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(shell_config_module, "SHELL_CONFIG_DIR", tmp_path)
    monkeypatch.setattr("shared.sandbox.SANDBOX_ROOT", tmp_path / "sandbox")
    return tmp_path


def _request(config_path: str | None) -> ToolRequest:
    args: dict[str, object] = {"command": "echo hi"}
    if config_path is not None:
        args["config_path"] = config_path
    return ToolRequest(name="shell", args=args)


def test_config_path_absolute_rejected(config_dir: Path) -> None:
    res = handle_shell_request(_request(str(config_dir / "outside.json")))
    assert not res.ok
    assert "относительным" in (res.error or "").lower()
    assert not (config_dir / "outside.json").exists()


def test_config_path_parent_reference_rejected(config_dir: Path) -> None:
    res = handle_shell_request(_request("../outside.json"))
    assert not res.ok
    assert ".." in (res.error or "")
    assert not (config_dir.parent / "outside.json").exists()


def test_config_path_tilde_rejected(config_dir: Path) -> None:
    res = handle_shell_request(_request("~/outside.json"))
    assert not res.ok
    assert not (config_dir / "outside.json").exists()


def test_config_path_symlink_escape_rejected(config_dir: Path) -> None:
    outside = config_dir.parent / "escape_target"
    outside.mkdir(parents=True, exist_ok=True)
    (config_dir / "link").symlink_to(outside, target_is_directory=True)
    res = handle_shell_request(_request("link/escape.json"))
    assert not res.ok
    assert not (outside / "escape.json").exists()


def test_config_path_within_config_dir_applies(config_dir: Path) -> None:
    res = handle_shell_request(
        ToolRequest(
            name="shell",
            args={
                "command": "echo hi",
                "config_path": "nested/shell_config.json",
                "shell_config": {
                    "allowed_commands": ["echo"],
                    "timeout_seconds": 1,
                    "max_output_chars": 10,
                    "sandbox_root": "sandbox",
                },
            },
        )
    )
    assert res.ok
    assert (config_dir / "nested" / "shell_config.json").exists()


def test_save_shell_config_rejects_outside_path(config_dir: Path) -> None:
    with pytest.raises(ValueError):
        save_shell_config(ShellConfig(allowed_commands=["echo"]), str(config_dir.parent / "x.json"))
    with pytest.raises(ValueError):
        save_shell_config(ShellConfig(allowed_commands=["echo"]), "../x.json")
    with pytest.raises(ValueError):
        load_shell_config(str(config_dir.parent / "missing.json"))


def test_default_invocation_uses_canonical_shell_config(config_dir: Path) -> None:
    canonical = config_dir / "shell_config.json"
    canonical.write_text(
        '{"allowed_commands":["echo"],"timeout_seconds":1,'
        '"max_output_chars":100,"sandbox_root":"sandbox"}',
        encoding="utf-8",
    )
    res = handle_shell_request(_request(None))
    assert res.ok
    assert "hi" in str(res.data.get("output"))
    assert not (config_dir / "config" / "shell_config.json").exists()
    assert not (config_dir / "config").exists()


def test_default_invocation_does_not_double_prefix_path(config_dir: Path) -> None:
    save_shell_config(ShellConfig(allowed_commands=["echo"]), None)
    assert (config_dir / "shell_config.json").exists()
    assert not (config_dir / "config").exists()
    loaded = load_shell_config(None)
    assert loaded.allowed_commands == ["echo"]
