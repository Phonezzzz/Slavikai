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


def _request(args: dict[str, object]) -> ToolRequest:
    return ToolRequest(name="shell", args=args)


def _seed_python_file(config_dir: Path) -> Path:
    target = config_dir / "settings.py"
    target.write_text("SECRET_KEY = 'original'", encoding="utf-8")
    return target


def test_shell_config_path_arg_is_ignored_and_cannot_overwrite_py_file(config_dir: Path) -> None:
    target = _seed_python_file(config_dir)
    res = handle_shell_request(
        _request(
            {
                "command": "echo hi",
                "config_path": "settings.py",
                "shell_config": {
                    "allowed_commands": ["echo"],
                    "timeout_seconds": 1,
                    "max_output_chars": 10,
                    "sandbox_root": "sandbox",
                },
            }
        )
    )
    assert res.ok
    assert target.read_text(encoding="utf-8") == "SECRET_KEY = 'original'"
    assert not (config_dir / "shell_config.json").exists()


def test_shell_config_path_arg_absolute_cannot_overwrite_other_file(config_dir: Path) -> None:
    target = _seed_python_file(config_dir)
    res = handle_shell_request(
        _request(
            {
                "command": "echo hi",
                "config_path": str(target),
                "shell_config": {
                    "allowed_commands": ["echo"],
                    "timeout_seconds": 1,
                    "max_output_chars": 10,
                    "sandbox_root": "sandbox",
                },
            }
        )
    )
    assert res.ok
    assert target.read_text(encoding="utf-8") == "SECRET_KEY = 'original'"
    assert not (config_dir / "shell_config.json").exists()


def test_shell_config_path_arg_nested_cannot_overwrite_other_file(config_dir: Path) -> None:
    target = _seed_python_file(config_dir)
    res = handle_shell_request(
        _request(
            {
                "command": "echo hi",
                "config_path": "nested/../settings.py",
                "shell_config": {
                    "allowed_commands": ["echo"],
                    "timeout_seconds": 1,
                    "max_output_chars": 10,
                    "sandbox_root": "sandbox",
                },
            }
        )
    )
    assert res.ok
    assert target.read_text(encoding="utf-8") == "SECRET_KEY = 'original'"
    assert not (config_dir / "shell_config.json").exists()


def test_shell_config_payload_cannot_overwrite_other_config_files(config_dir: Path) -> None:
    target = _seed_python_file(config_dir)
    other = config_dir / "api_keys.py"
    other.write_text("API_KEY = 'x'", encoding="utf-8")
    res = handle_shell_request(
        _request(
            {
                "command": "echo hi",
                "shell_config": {
                    "allowed_commands": ["echo"],
                    "timeout_seconds": 1,
                    "max_output_chars": 10,
                    "sandbox_root": "sandbox",
                },
            }
        )
    )
    assert res.ok
    assert target.read_text(encoding="utf-8") == "SECRET_KEY = 'original'"
    assert other.read_text(encoding="utf-8") == "API_KEY = 'x'"
    assert not (config_dir / "shell_config.json").exists()


def test_shell_config_payload_cannot_persist_modified_policy(config_dir: Path) -> None:
    canonical = config_dir / "shell_config.json"
    canonical.write_text(
        '{"allowed_commands":["echo"],"timeout_seconds":10,'
        '"max_output_chars":6000,"sandbox_root":"sandbox"}',
        encoding="utf-8",
    )
    original = canonical.read_text(encoding="utf-8")
    res = handle_shell_request(
        _request(
            {
                "command": "echo hi",
                "shell_config": {
                    "allowed_commands": ["rm", "shutdown"],
                    "timeout_seconds": 999,
                    "max_output_chars": 999999,
                    "sandbox_root": "sandbox",
                },
            }
        )
    )
    assert res.ok
    assert canonical.read_text(encoding="utf-8") == original
    blocked = handle_shell_request(_request({"command": "rm x"}))
    assert not blocked.ok
    assert "запрещена политикой" in (blocked.error or "")


def test_save_shell_config_has_no_selectable_path(config_dir: Path) -> None:
    save_shell_config(ShellConfig(allowed_commands=["echo"]))
    assert (config_dir / "shell_config.json").exists()
    assert (config_dir / "shell_config.json").read_text(encoding="utf-8").startswith("{")
    assert not (config_dir / "config").exists()


def test_load_shell_config_has_no_selectable_path(config_dir: Path) -> None:
    (config_dir / "shell_config.json").write_text(
        '{"allowed_commands":["echo"],"timeout_seconds":1,'
        '"max_output_chars":100,"sandbox_root":"sandbox"}',
        encoding="utf-8",
    )
    loaded = load_shell_config()
    assert loaded.allowed_commands == ["echo"]


def test_default_invocation_uses_canonical_shell_config(config_dir: Path) -> None:
    (config_dir / "shell_config.json").write_text(
        '{"allowed_commands":["echo"],"timeout_seconds":1,'
        '"max_output_chars":100,"sandbox_root":"sandbox"}',
        encoding="utf-8",
    )
    res = handle_shell_request(_request({"command": "echo hi"}))
    assert res.ok
    assert "hi" in str(res.data.get("output"))
    assert not (config_dir / "config").exists()


def test_default_invocation_does_not_double_prefix_path(config_dir: Path) -> None:
    save_shell_config(ShellConfig(allowed_commands=["echo"]))
    assert (config_dir / "shell_config.json").exists()
    assert not (config_dir / "config").exists()
    loaded = load_shell_config()
    assert loaded.allowed_commands == ["echo"]


def test_config_dir_other_files_untouched_by_default_flow(config_dir: Path) -> None:
    target = _seed_python_file(config_dir)
    res = handle_shell_request(_request({"command": "echo hi"}))
    assert res.ok
    assert target.read_text(encoding="utf-8") == "SECRET_KEY = 'original'"
