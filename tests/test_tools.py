from __future__ import annotations

import uuid
from pathlib import Path

from shared.models import ToolRequest
from tools.filesystem_tool import handle_filesystem
from tools.http_client import HttpResult
from tools.shell_tool import handle_shell
from tools.web_search_tool import WebSearchTool


def test_filesystem_write_and_read_roundtrip() -> None:
    filename = f"test_{uuid.uuid4().hex}.txt"
    content = "hello world"
    write_result = handle_filesystem(
        ToolRequest(name="write", args={"path": filename, "content": content})
    )
    assert write_result.ok, write_result.error

    read_result = handle_filesystem(ToolRequest(name="read", args={"path": filename}))
    assert read_result.ok, read_result.error
    assert read_result.data.get("output") == content


def test_filesystem_denies_escape() -> None:
    result = handle_filesystem(ToolRequest(name="read", args={"path": "../etc/passwd"}))
    assert not result.ok
    assert "sandbox violation" in (result.error or "").lower()


def test_fs_rejects_sibling_prefix_escape(tmp_path: Path, monkeypatch) -> None:
    sandbox_root = (tmp_path / "sandbox").resolve()
    sandbox_root.mkdir(parents=True)
    evil_dir = tmp_path / "sandbox_evil"
    evil_dir.mkdir(parents=True)
    (evil_dir / "secret.txt").write_text("evil", encoding="utf-8")

    monkeypatch.setattr("tools.filesystem_tool.SANDBOX_ROOT", sandbox_root)

    result = handle_filesystem(
        ToolRequest(name="read", args={"path": "../sandbox_evil/secret.txt"})
    )
    assert not result.ok
    assert "sandbox violation" in (result.error or "").lower()


def test_fs_rejects_absolute_path(tmp_path: Path, monkeypatch) -> None:
    sandbox_root = (tmp_path / "sandbox").resolve()
    sandbox_root.mkdir(parents=True)

    monkeypatch.setattr("tools.filesystem_tool.SANDBOX_ROOT", sandbox_root)

    result = handle_filesystem(ToolRequest(name="read", args={"path": "/etc/passwd"}))
    assert not result.ok
    assert "sandbox violation" in (result.error or "").lower()


def test_fs_allows_normal_file(tmp_path: Path, monkeypatch) -> None:
    sandbox_root = (tmp_path / "sandbox").resolve()
    sandbox_root.mkdir(parents=True)
    (sandbox_root / "hello.txt").write_text("world", encoding="utf-8")

    monkeypatch.setattr("tools.filesystem_tool.SANDBOX_ROOT", sandbox_root)

    result = handle_filesystem(ToolRequest(name="read", args={"path": "hello.txt"}))
    assert result.ok, result.error
    assert result.data.get("output") == "world"


def test_shell_blocks_dangerous_command() -> None:
    result = handle_shell("rm -rf /")
    assert not result.ok


def test_shell_allows_simple_command() -> None:
    from tools.shell_tool import ShellConfig, handle_shell_request

    cfg = ShellConfig(allowed_commands=["echo"])
    result = handle_shell_request(
        ToolRequest(name="shell", args={"command": "echo hi", "shell_config": cfg.__dict__})
    )
    assert result.ok
    assert "hi" in str(result.data.get("output"))


def test_web_search_simulation() -> None:
    class DummyHttp:
        def get_text(self, url: str, **kwargs):
            return HttpResult(
                ok=True,
                data="stub content",
                status_code=200,
                error=None,
                headers={},
                meta={},
            )

    web = WebSearchTool(http_client=DummyHttp())
    result = web.handle(ToolRequest(name="web", args={"query": "https://example.com"}))
    assert result.ok
    assert "stub content" in str(result.data.get("output"))


def test_web_invalid_url() -> None:
    web = WebSearchTool()
    result = web.handle(ToolRequest(name="web", args={"query": "ftp://example.com"}))
    assert not result.ok


def test_shell_blocks_absolute_path() -> None:
    result = handle_shell("cat /etc/hosts")
    assert not result.ok


def test_shell_blocks_command_chaining() -> None:
    result = handle_shell("ls && whoami")
    assert not result.ok
