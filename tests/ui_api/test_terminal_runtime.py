from __future__ import annotations

# ruff: noqa: F811
import asyncio

import pytest

from .fakes import _create_client, _set_session_policy_via_api
from .test_durable_submission import isolated_agent  # noqa: F401


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/ui/api/terminal", {}),
        ("/ui/api/terminal/input", {"input": "echo private\n"}),
        ("/ui/api/terminal/resize", {"rows": 30, "cols": 90}),
    ],
)
def test_terminal_yolo_is_not_isolation_authority(
    isolated_agent,
    monkeypatch,
    endpoint,
    payload,  # noqa: F811
):
    agent, _ = isolated_agent

    async def run():
        client = await _create_client(agent)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            headers = {"X-Slavik-Session": session}
            denied = await client.post(endpoint, headers=headers, json=payload)
            assert denied.status == 403
            status, _ = await _set_session_policy_via_api(
                client,
                session_id=session,
                policy_profile="yolo",
                confirm_yolo=True,
            )
            assert status == 200
            calls = []
            monkeypatch.setattr("subprocess.Popen", lambda *a, **k: calls.append(a))
            unavailable = await client.post(endpoint, headers=headers, json=payload)
            assert unavailable.status == 503
            body = await unavailable.json()
            assert body["error"]["code"] == "sandbox_execution_unavailable"
            assert calls == []
            snapshot = await client.get("/ui/api/terminal", headers=headers)
            assert snapshot.status == 200
            assert (await snapshot.json())["terminal"]["status"] == "not_started"
        finally:
            await client.close()

    asyncio.run(run())
