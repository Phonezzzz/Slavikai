from __future__ import annotations

import asyncio

import pytest

from shared.models import ToolRequest
from shared.sandbox_execution import SandboxExecutionUnavailable
from tools.terminal_tool import TerminalTool, _TerminalState


def test_terminal_create_input_resize_refuse_before_os_actions(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: calls.append(a))
    monkeypatch.setattr("os.write", lambda *a: calls.append(a))

    async def run():
        manager = TerminalTool()
        # Retained pre-fence state не становится разрешённой environment.
        manager._states["old"] = _TerminalState(
            terminal_id="old-id",
            session_id="old",
            workspace_root=str(tmp_path),
            spawn_cwd=str(tmp_path),
            rows=24,
            cols=80,
            master_fd=999,
        )
        for session in ("new", "old"):
            with pytest.raises(SandboxExecutionUnavailable):
                await manager.create_or_get(session, workspace_root=str(tmp_path))
            with pytest.raises(SandboxExecutionUnavailable):
                await manager.write_input(session, "echo private\n")
            with pytest.raises(SandboxExecutionUnavailable):
                await manager.resize(session, rows=40, cols=120)
        assert await manager.get_snapshot("new") is None
        assert (await manager.get_snapshot("old"))["terminal_id"] == "old-id"
        assert calls == []
        manager._states["old"].master_fd = None
        await manager.shutdown()

    asyncio.run(run())


def test_terminal_handle_cannot_turn_workspace_root_into_host_authority(tmp_path):
    result = TerminalTool().handle(
        ToolRequest(
            name="workspace_terminal_run",
            args={
                "command": "echo hello",
                "workspace_root": str(tmp_path),
                "sandbox_root": str(tmp_path),
            },
        )
    )
    assert not result.ok
    assert result.meta["reason_code"] == "sandbox_execution_unavailable"
