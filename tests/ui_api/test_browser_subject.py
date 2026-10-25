import asyncio

from tests.ui_api.test_memory_management import _isolate, _start


def test_auth_status_binds_opaque_hint_subject_to_verified_settings_principal(
    monkeypatch, tmp_path
):
    _isolate(monkeypatch, tmp_path)

    async def run():
        client = await _start()
        try:
            subjects = []
            for who in ["owner", "member"]:
                h = {"Cf-Access-Jwt-Assertion": who}
                response = await client.get("/ui/api/auth/status", headers=h)
                assert response.status == 200
                status = await response.json()
                settings = await (await client.get("/ui/api/settings", headers=h)).json()
                assert status["browser_subject"] == settings["settings"]["preferences_subject"]
                assert response.headers["Cache-Control"] == "no-store"
                assert len(status["browser_subject"]) == 64
                assert "@" not in status["browser_subject"]
                subjects.append(status["browser_subject"])
            assert subjects[0] != subjects[1]
        finally:
            await client.close()

    asyncio.run(run())


def test_auth_subject_keeps_existing_local_and_token_lanes_separate(monkeypatch, tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    from config.http_server_config import HttpAuthConfig
    from server.http.app import create_app
    from server.ui_session_storage import InMemoryUISessionStorage

    _isolate(monkeypatch, tmp_path)

    async def run():
        for local in [True, False]:
            app = create_app(
                ui_storage=InMemoryUISessionStorage(),
                auth_config=HttpAuthConfig(
                    api_token="fixture-token",
                    browser_auth_mode="token",
                    allow_unauth_local=local,
                ),
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                response = await client.get("/ui/api/auth/status")
                status = await response.json()
                if not local:
                    assert status["authenticated"] is False
                    assert status["browser_subject"] is None
                    assert (
                        await client.post("/ui/api/auth/login", json={"token": "fixture-token"})
                    ).status == 200
                    response = await client.get("/ui/api/auth/status")
                    status = await response.json()
                assert status["authenticated"] is True or (
                    local and status["auth_required"] is False
                )
                assert response.headers["Cache-Control"] == "no-store"
                settings = await (await client.get("/ui/api/settings")).json()
                assert status["browser_subject"] == settings["settings"]["preferences_subject"]
            finally:
                await client.close()

    asyncio.run(run())
