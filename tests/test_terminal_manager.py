from __future__ import annotations

import asyncio
from pathlib import Path

from server.terminal_manager import TerminalManager


async def _wait_for_snapshot(
    manager: TerminalManager,
    session_id: str,
    *,
    predicate,
    attempts: int = 60,
) -> dict[str, object]:
    for _ in range(attempts):
        snapshot = await manager.require_snapshot(session_id)
        if predicate(snapshot):
            return snapshot
        await asyncio.sleep(0.05)
    raise AssertionError("terminal snapshot predicate was not satisfied")


def test_terminal_manager_reuses_running_terminal_and_uses_workspace_root(tmp_path: Path) -> None:
    async def run() -> None:
        manager = TerminalManager()
        session_id = "session-a"
        first = await manager.create_or_get(session_id, workspace_root=str(tmp_path))
        second = await manager.create_or_get(session_id, workspace_root=str(tmp_path))
        assert first["terminal_id"] == second["terminal_id"]
        assert first["workspace_root"] == str(tmp_path)
        assert first["spawn_cwd"] == str(tmp_path)

        await manager.write_input(session_id, "pwd\n")
        snapshot = await _wait_for_snapshot(
            manager,
            session_id,
            predicate=lambda item: str(tmp_path) in str(item.get("output") or ""),
        )
        assert str(tmp_path) in str(snapshot.get("output") or "")
        await manager.shutdown()

    asyncio.run(run())


def test_terminal_manager_resize_close_and_new_terminal_id(tmp_path: Path) -> None:
    async def run() -> None:
        manager = TerminalManager()
        session_id = "session-b"
        first = await manager.create_or_get(session_id, workspace_root=str(tmp_path))
        resized = await manager.resize(session_id, rows=40, cols=120)
        assert resized["rows"] == 40
        assert resized["cols"] == 120

        closed = await manager.close(session_id)
        assert closed["status"] == "closed"

        second = await manager.create_or_get(session_id, workspace_root=str(tmp_path))
        assert second["terminal_id"] != first["terminal_id"]
        await manager.shutdown()

    asyncio.run(run())


def test_terminal_manager_tracks_exit_code_and_shutdown(tmp_path: Path) -> None:
    async def run() -> None:
        manager = TerminalManager()
        session_id = "session-c"
        await manager.create_or_get(session_id, workspace_root=str(tmp_path))
        await manager.write_input(session_id, "exit 7\n")
        exited = await _wait_for_snapshot(
            manager,
            session_id,
            predicate=lambda item: item.get("status") == "exited",
        )
        assert exited["exit_code"] == 7
        await manager.shutdown()

    asyncio.run(run())
