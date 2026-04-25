from __future__ import annotations

import asyncio
from pathlib import Path

from server.terminal_manager import TerminalManager
from server.ui_hub import UIHub

# ruff: noqa: F403,F405
from .fakes import *


async def _wait_for_terminal_snapshot(
    client,
    session_id: str,
    *,
    predicate,
    attempts: int = 80,
) -> dict[str, object]:
    for _ in range(attempts):
        response = await client.get("/ui/api/terminal", headers={"X-Slavik-Session": session_id})
        assert response.status == 200
        payload = await response.json()
        terminal = payload.get("terminal")
        assert isinstance(terminal, dict)
        if predicate(terminal):
            return terminal
        await asyncio.sleep(0.05)
    raise AssertionError("terminal snapshot predicate was not satisfied")


def test_ui_terminal_snapshot_create_policy_and_recreate() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            snapshot_resp = await client.get(
                "/ui/api/terminal",
                headers={"X-Slavik-Session": session_id},
            )
            assert snapshot_resp.status == 200
            snapshot_payload = await snapshot_resp.json()
            terminal = snapshot_payload.get("terminal")
            assert isinstance(terminal, dict)
            assert terminal.get("status") == "not_started"

            create_forbidden = await client.post(
                "/ui/api/terminal",
                headers={"X-Slavik-Session": session_id},
                json={},
            )
            assert create_forbidden.status == 403

            status_code, _ = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="yolo",
                confirm_yolo=True,
            )
            assert status_code == 200

            create_resp = await client.post(
                "/ui/api/terminal",
                headers={"X-Slavik-Session": session_id},
                json={"rows": 33, "cols": 101},
            )
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            created_terminal = create_payload.get("terminal")
            assert isinstance(created_terminal, dict)
            first_terminal_id = created_terminal.get("terminal_id")
            assert isinstance(first_terminal_id, str)
            assert created_terminal.get("rows") == 33
            assert created_terminal.get("cols") == 101

            create_again = await client.post(
                "/ui/api/terminal",
                headers={"X-Slavik-Session": session_id},
                json={},
            )
            assert create_again.status == 200
            create_again_payload = await create_again.json()
            same_terminal = create_again_payload.get("terminal")
            assert isinstance(same_terminal, dict)
            assert same_terminal.get("terminal_id") == first_terminal_id

            status_code, _ = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="sandbox",
            )
            assert status_code == 200

            snapshot_after_policy = await client.get(
                "/ui/api/terminal",
                headers={"X-Slavik-Session": session_id},
            )
            assert snapshot_after_policy.status == 200

            input_forbidden = await client.post(
                "/ui/api/terminal/input",
                headers={"X-Slavik-Session": session_id},
                json={"input": "pwd\n"},
            )
            assert input_forbidden.status == 403

            resize_forbidden = await client.post(
                "/ui/api/terminal/resize",
                headers={"X-Slavik-Session": session_id},
                json={"rows": 40, "cols": 120},
            )
            assert resize_forbidden.status == 403

            close_resp = await client.post(
                "/ui/api/terminal/close",
                headers={"X-Slavik-Session": session_id},
            )
            assert close_resp.status == 200
            close_payload = await close_resp.json()
            closed_terminal = close_payload.get("terminal")
            assert isinstance(closed_terminal, dict)
            assert closed_terminal.get("status") == "closed"

            status_code, _ = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="yolo",
                confirm_yolo=True,
            )
            assert status_code == 200
            recreate_resp = await client.post(
                "/ui/api/terminal",
                headers={"X-Slavik-Session": session_id},
                json={},
            )
            assert recreate_resp.status == 200
            recreate_payload = await recreate_resp.json()
            recreated_terminal = recreate_payload.get("terminal")
            assert isinstance(recreated_terminal, dict)
            second_terminal_id = recreated_terminal.get("terminal_id")
            assert isinstance(second_terminal_id, str)
            assert second_terminal_id != first_terminal_id
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_terminal_stream_replay_and_workspace_root_lifecycle(tmp_path: Path) -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]

            first_root = tmp_path / "root-one"
            first_root.mkdir(parents=True, exist_ok=True)
            second_root = tmp_path / "root-two"
            second_root.mkdir(parents=True, exist_ok=True)
            await hub.set_workspace_root(session_id, str(first_root))

            status_code, _ = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="yolo",
                confirm_yolo=True,
            )
            assert status_code == 200

            create_resp = await client.post(
                "/ui/api/terminal",
                headers={"X-Slavik-Session": session_id},
                json={},
            )
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            created_terminal = create_payload.get("terminal")
            assert isinstance(created_terminal, dict)
            assert created_terminal.get("workspace_root") == str(first_root)
            assert created_terminal.get("spawn_cwd") == str(first_root)

            stream_resp = await client.get(
                "/ui/api/terminal/stream",
                headers={"X-Slavik-Session": session_id},
                timeout=5,
            )
            assert stream_resp.status == 200

            first_input = await client.post(
                "/ui/api/terminal/input",
                headers={"X-Slavik-Session": session_id},
                json={"input": "printf '__ONE__\\n'\n"},
            )
            assert first_input.status == 200
            first_events = await _read_sse_events(stream_resp, max_events=8)
            output_events = [
                event
                for event in first_events
                if event.get("type") == "terminal.output"
                and isinstance(event.get("payload"), dict)
                and "__ONE__" in str(event["payload"].get("data") or "")
            ]
            assert output_events
            anchor_id = output_events[0].get("id")
            assert isinstance(anchor_id, str)

            await hub.set_workspace_root(session_id, str(second_root))

            second_input = await client.post(
                "/ui/api/terminal/input",
                headers={"X-Slavik-Session": session_id},
                json={"input": f"cd {second_root}\npwd\nprintf '__TWO__\\n'\n"},
            )
            assert second_input.status == 200
            second_snapshot = await _wait_for_terminal_snapshot(
                client,
                session_id,
                predicate=lambda terminal: "__TWO__" in str(terminal.get("output") or ""),
            )
            assert str(second_root) in str(second_snapshot.get("output") or "")

            stream_resp.close()
            replay_resp = await client.get(
                "/ui/api/terminal/stream",
                headers={"X-Slavik-Session": session_id, "Last-Event-ID": anchor_id},
                timeout=5,
            )
            assert replay_resp.status == 200
            replay_events = await _read_sse_events(replay_resp, max_events=12)
            replay_output = [
                event
                for event in replay_events
                if event.get("type") == "terminal.output" and isinstance(event.get("payload"), dict)
            ]
            assert any(
                "__TWO__" in str(event["payload"].get("data") or "") for event in replay_output
            )
            replay_resp.close()

            close_resp = await client.post(
                "/ui/api/terminal/close",
                headers={"X-Slavik-Session": session_id},
            )
            assert close_resp.status == 200

            recreate_resp = await client.post(
                "/ui/api/terminal",
                headers={"X-Slavik-Session": session_id},
                json={},
            )
            assert recreate_resp.status == 200

            third_input = await client.post(
                "/ui/api/terminal/input",
                headers={"X-Slavik-Session": session_id},
                json={"input": "pwd\nprintf '__THREE__\\n'\n"},
            )
            assert third_input.status == 200
            recreated_snapshot = await _wait_for_terminal_snapshot(
                client,
                session_id,
                predicate=lambda terminal: "__THREE__" in str(terminal.get("output") or ""),
            )
            output_text = str(recreated_snapshot.get("output") or "")
            assert str(second_root) in output_text
            assert str(first_root) not in output_text
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_terminal_session_delete_cleans_terminal_state() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            status_code, _ = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="yolo",
                confirm_yolo=True,
            )
            assert status_code == 200

            create_resp = await client.post(
                "/ui/api/terminal",
                headers={"X-Slavik-Session": session_id},
                json={},
            )
            assert create_resp.status == 200

            delete_resp = await client.delete(f"/ui/api/sessions/{session_id}")
            assert delete_resp.status == 200

            manager = client.server.app["terminal_manager"]
            assert isinstance(manager, TerminalManager)
            assert await manager.get_snapshot(session_id) is None
        finally:
            await client.close()

    asyncio.run(run())
