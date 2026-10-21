import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from config.http_server_config import HttpAuthConfig
from server.http.app import create_app
from server.http.common.request_identity import VerifiedAccessClaims
from server.ui_session_storage import InMemoryUISessionStorage
from tests.test_cloudflare_access import StubAccessVerifier
from tests.ui_api.fakes import DummyAgent


def test_settings_are_principal_scoped_and_subject_bound(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    ui = tmp_path / "ui.json"
    ui.write_text(
        '{"personalization":{"system_prompt":"OLD_GLOBAL_PRIVATE"},"appearance":{"theme":"oled"}}'
    )
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui)
    monkeypatch.setattr("server.http_api.API_KEYS_PATH", tmp_path / "keys.json")
    monkeypatch.setattr("server.http_api.MODEL_CONFIG_PATH", tmp_path / "models.json")

    async def start():
        app = create_app(
            agent=DummyAgent(),
            ui_storage=InMemoryUISessionStorage(),
            auth_config=HttpAuthConfig(
                api_token="automation",
                browser_auth_mode="cloudflare",
                cloudflare_access_issuer="https://example.cloudflareaccess.com",
                cloudflare_access_aud="test",
                owner_email="owner@example.com",
            ),
            cloudflare_access_verifier=StubAccessVerifier(
                {
                    "owner": VerifiedAccessClaims(email="owner@example.com"),
                    "member": VerifiedAccessClaims(email="member@example.com"),
                }
            ),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        return client

    async def read(client, who):
        response = await client.get("/ui/api/settings", headers={"Cf-Access-Jwt-Assertion": who})
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        return (await response.json())["settings"]

    async def run():
        client = await start()
        try:
            owner, member = await read(client, "owner"), await read(client, "member")
            assert owner["personalization"]["system_prompt"] == ""
            assert member["appearance"]["theme"] == "default"
            for who in ["owner", "member"]:
                for _ in range(2):
                    created = await client.post(
                        "/ui/api/sessions", headers={"Cf-Access-Jwt-Assertion": who}
                    )
                    assert created.status == 200
                    await created.read()
            for who, current in [("owner", owner), ("member", member)]:
                response = await client.post(
                    "/ui/api/settings",
                    headers={
                        "Cf-Access-Jwt-Assertion": who,
                        "If-Match": current["preferences_etag"],
                    },
                    json={"personalization": {"system_prompt": who + "_PRIVATE"}},
                )
                assert response.status == 200
                await response.read()
            assert (await read(client, "owner"))["personalization"][
                "system_prompt"
            ] == "owner_PRIVATE"
            assert (await read(client, "member"))["personalization"][
                "system_prompt"
            ] == "member_PRIVATE"
            wrong_subject = await client.post(
                "/ui/api/settings",
                headers={
                    "Cf-Access-Jwt-Assertion": "member",
                    "If-Match": owner["preferences_etag"],
                },
                json={"appearance": {"theme": "oled"}},
            )
            assert wrong_subject.status == 409
            forbidden = await client.post(
                "/ui/api/settings",
                headers={"Cf-Access-Jwt-Assertion": "member"},
                json={"memory": {"inbox_max_items": 20}},
            )
            assert forbidden.status == 403
            fresh_member = await read(client, "member")
            forged = await client.post(
                "/ui/api/settings",
                headers={
                    "Cf-Access-Jwt-Assertion": "member",
                    "If-Match": fresh_member["preferences_etag"],
                },
                json={"personalization": {"principal_id": "owner@example.com"}},
            )
            assert forged.status == 400
            bearer = await client.get(
                "/ui/api/settings", headers={"Authorization": "Bearer automation"}
            )
            assert bearer.status == 401
        finally:
            await client.close()
        restarted = await start()
        try:
            assert (await read(restarted, "member"))["personalization"][
                "system_prompt"
            ] == "member_PRIVATE"
            assert (await read(restarted, "owner"))["personalization"][
                "system_prompt"
            ] == "owner_PRIVATE"
        finally:
            await restarted.close()

    asyncio.run(run())
    assert "owner_PRIVATE" not in caplog.text
    assert "member_PRIVATE" not in caplog.text
    assert "owner_PRIVATE" not in repr([vars(record) for record in caplog.records])
    assert "member_PRIVATE" not in repr([vars(record) for record in caplog.records])
