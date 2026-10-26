import asyncio

from config.tools_config import DEFAULT_TOOLS_STATE
from shared.identity import principal_id_for_email
from tests.ui_api.test_memory_management import _isolate, _start


def test_owned_session_tool_sources_use_captured_override_not_value_difference(
    monkeypatch, tmp_path
):
    _isolate(monkeypatch, tmp_path)
    defaults = dict(DEFAULT_TOOLS_STATE)
    monkeypatch.setattr("server.http_api._load_tools_state", lambda: dict(defaults))

    async def run():
        client = await _start()
        try:
            hub = client.app["ui_hub"]
            for who in ["owner", "member"]:
                session = await hub.create_session(principal_id_for_email(who + "@example.com"))
                await hub.set_workspace_root(session, str(tmp_path))
                await hub.set_session_tools_state(
                    session, tools_state={"fs": defaults["fs"], "web": False}
                )
                before = await hub.get_session(session)
                response = await client.get(
                    "/ui/api/session/security",
                    headers={"Cf-Access-Jwt-Assertion": who, "X-Slavik-Session": session},
                )
                assert response.status == 200
                data = await response.json()
                assert data["tool_sources"]["basis"] == "configuration_at_refresh"
                sources = data["tool_sources"]["fields"]
                assert set(sources) == set(DEFAULT_TOOLS_STATE) - {"safe_mode"}
                assert sources["fs"] == sources["web"] == "session_override"
                assert sources["shell"] == "installation_configuration"
                assert data["tools_state"]["web"] is False
                assert data["tools_state"]["fs"] == defaults["fs"]
                assert response.headers["Cache-Control"] == "no-store"
                assert await hub.get_session(session) == before
                rejected = await client.post(
                    "/ui/api/session/security",
                    headers={"Cf-Access-Jwt-Assertion": who, "X-Slavik-Session": session},
                    json={"tool_sources": data["tool_sources"]},
                )
                assert rejected.status == 400
                assert await hub.get_session(session) == before
                foreign = "member" if who == "owner" else "owner"
                denied = await client.get(
                    "/ui/api/session/security",
                    headers={"Cf-Access-Jwt-Assertion": foreign, "X-Slavik-Session": session},
                )
                assert denied.status == 403
                assert "tool_sources" not in await denied.json()
                # A later read follows current configuration, never creates an override.
                defaults["shell"] = not defaults["shell"]
                again = await client.get(
                    "/ui/api/session/security",
                    headers={"Cf-Access-Jwt-Assertion": who, "X-Slavik-Session": session},
                )
                updated = await again.json()
                assert updated["tools_state"]["shell"] == defaults["shell"]
                assert updated["tool_sources"]["fields"]["shell"] == "installation_configuration"
        finally:
            await client.close()

    asyncio.run(run())


def test_session_security_configuration_read_is_detached_and_never_creates_state():
    from server.ui_hub import UIHub
    from server.ui_session_storage import InMemoryUISessionStorage

    async def run():
        hub = UIHub(storage=InMemoryUISessionStorage())
        assert await hub.get_session_security_configuration("missing") is None
        assert await hub.get_session("missing") is None
        session = await hub.create_session("owner")
        await hub.set_session_policy(session, profile="index")
        await hub.set_session_tools_state(session, tools_state={"fs": False})
        captured = await hub.get_session_security_configuration(session)
        assert captured == ("index", {"fs": False})
        captured[1]["fs"] = True
        await hub.set_session_policy(session, profile="sandbox")
        assert await hub.get_session_security_configuration(session) == ("sandbox", {"fs": False})

    asyncio.run(run())
