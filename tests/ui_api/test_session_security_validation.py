import asyncio

import pytest

from shared.identity import principal_id_for_email
from tests.ui_api.test_memory_management import _isolate, _start


@pytest.mark.parametrize("who", ["owner", "member"])
@pytest.mark.parametrize(
    "tools",
    [
        None,
        [],
        {},
        {"state": None},
        {"state": {"fs": "invalid"}},
        {"state": {"unknown": True}},
        {"state": {"safe_mode": False}},
    ],
)
def test_invalid_combined_session_security_never_persists_policy(monkeypatch, tmp_path, who, tools):
    _isolate(monkeypatch, tmp_path)

    async def run():
        client = await _start()
        try:
            hub = client.app["ui_hub"]
            session = await hub.create_session(principal_id_for_email(who + "@example.com"))
            await hub.set_workspace_root(session, str(tmp_path))
            await hub.set_session_tools_state(session, tools_state={"fs": False})
            before = await hub.get_session(session)
            response = await client.post(
                "/ui/api/session/security",
                headers={"Cf-Access-Jwt-Assertion": who, "X-Slavik-Session": session},
                json={"policy": {"profile": "index"}, "tools": tools},
            )
            assert response.status == 400
            assert await hub.get_session(session) == before
            persisted = client.app["ui_hub"]._storage
            assert persisted is not None
            # Проверяем также снимок storage, а не только ответ endpoint.
            records = persisted.load_sessions()
            record = next(item for item in records if item.session_id == session)
            assert record.policy_profile == "sandbox"
            assert record.tools_state == {"fs": False}
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("who", ["owner", "member"])
def test_valid_combined_security_and_rejected_policy_preserve_existing_guards(
    monkeypatch, tmp_path, who
):
    _isolate(monkeypatch, tmp_path)

    async def run():
        client = await _start()
        try:
            hub = client.app["ui_hub"]
            session = await hub.create_session(principal_id_for_email(who + "@example.com"))
            await hub.set_workspace_root(session, str(tmp_path))
            headers = {"Cf-Access-Jwt-Assertion": who, "X-Slavik-Session": session}
            before = await hub.get_session(session)
            for policy in [
                {"profile": "yolo"},
                {"profile": "unexpected"},
                {"profile": "sandbox", "yolo_armed": True},
            ]:
                rejected = await client.post(
                    "/ui/api/session/security",
                    headers=headers,
                    json={"policy": policy, "tools": {"state": {"fs": False}}},
                )
                assert rejected.status == 400
                assert await hub.get_session(session) == before
            saved = await client.post(
                "/ui/api/session/security",
                headers=headers,
                json={
                    "policy": {"profile": "index"},
                    "tools": {"state": {"fs": False, "web": True}},
                },
            )
            assert saved.status == 200
            payload = await saved.json()
            assert payload["policy"] == {"profile": "index"}
            assert payload["tools_state"]["fs"] is False
            assert payload["tools_state"]["web"] is True
            persisted = await hub.get_session(session)
            foreign = "member" if who == "owner" else "owner"
            denied = await client.post(
                "/ui/api/session/security",
                headers={**headers, "Cf-Access-Jwt-Assertion": foreign},
                json={"policy": {"profile": "sandbox"}, "tools": {"state": {"fs": True}}},
            )
            assert denied.status == 403
            assert await hub.get_session(session) == persisted
        finally:
            await client.close()

    asyncio.run(run())
