import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer

from config.http_server_config import HttpAuthConfig
from core.desktop_policy import DesktopApprovalRule, DesktopApprovalScope, DesktopPolicyStore
from server.http.app import create_app
from server.http.common.request_identity import VerifiedAccessClaims
from server.ui_session_storage import InMemoryUISessionStorage
from shared.identity import principal_id_for_email, principal_subject
from tests.test_cloudflare_access import StubAccessVerifier
from tests.ui_api.test_memory_management import _isolate


def test_owner_desktop_read_view_is_private_sanitized_and_never_rewrites_store(
    monkeypatch, tmp_path
):
    _isolate(monkeypatch, tmp_path)
    owner = principal_id_for_email("owner@example.com")
    path = tmp_path / "rules.json"
    own = DesktopApprovalRule.create(
        effect="deny",
        scope=DesktopApprovalScope(
            tool="desktop_shell",
            action="run",
            command_exact="SECRET_COMMAND",
            target_pattern="https://user:SECRET_URL@example.invalid/path?token=SECRET_QUERY",
        ),
        source="persistent",
        subject_principal_id=owner,
        description="SECRET_DESCRIPTION",
    ).to_dict()
    own["rule_id"] = "SECRET_ID"
    legacy = DesktopApprovalRule.create(
        effect="allow",
        scope=DesktopApprovalScope(
            tool="desktop_file_delete", action="delete", target_pattern="/home/owner/one.iso"
        ),
        source="persistent",
    ).to_dict()
    foreign = DesktopApprovalRule.create(
        effect="deny",
        scope=DesktopApprovalScope(
            tool="desktop_shell", action="run", command_exact="SECRET_FOREIGN"
        ),
        source="persistent",
        subject_principal_id=principal_id_for_email("member@example.com"),
    ).to_dict()
    path.write_text(json.dumps([own, legacy, foreign]))
    before = path.read_bytes()
    store = DesktopPolicyStore(path, legacy_subject_principal_id=owner)

    async def run():
        client = TestClient(
            TestServer(
                create_app(
                    ui_storage=InMemoryUISessionStorage(),
                    desktop_policy_store=store,
                    auth_config=HttpAuthConfig(
                        api_token="automation",
                        browser_auth_mode="cloudflare",
                        cloudflare_access_issuer="https://fixture.cloudflareaccess.com",
                        cloudflare_access_aud="fixture",
                        owner_email="owner@example.com",
                    ),
                    cloudflare_access_verifier=StubAccessVerifier(
                        {
                            who: VerifiedAccessClaims(email=who + "@example.com")
                            for who in ["owner", "member"]
                        }
                    ),
                )
            )
        )
        await client.start_server()
        try:
            response = await client.get(
                "/ui/api/desktop/approvals?view=settings",
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert response.status == 200
            text = await response.text()
            assert "SECRET_" not in text and "email:" not in text
            view = json.loads(text)
            assert view["preferences_subject"] == principal_subject(owner)
            assert view["basis"] == "saved_desktop_rules_at_refresh"
            assert len(view["rules"]) == 2
            assert view["rules"][0]["command_restriction_present"] is True
            assert view["rules"][0]["target_kind"] == "hidden"
            assert view["rules"][0]["target_path"] is None
            assert view["rules"][1]["target_path"] == "/home/owner/one.iso"
            assert response.headers.get("Cache-Control") == "no-store"
            assert path.read_bytes() == before
            for headers in [
                {"Cf-Access-Jwt-Assertion": "member"},
                {"Authorization": "Bearer automation"},
                {"Cf-Access-Jwt-Assertion": "forged"},
            ]:
                denial = await client.get(
                    "/ui/api/desktop/approvals?view=settings", headers=headers
                )
                assert denial.status in {401, 403}
                assert "SECRET_" not in await denial.text()
            invalid = await client.get(
                "/ui/api/desktop/approvals?view=unknown",
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert invalid.status == 400
            path.write_text("SECRET_INVALID_JSON")
            response = await client.get(
                "/ui/api/desktop/approvals?view=settings",
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert response.status == 409
            assert "SECRET_" not in await response.text()
            assert path.read_text() == "SECRET_INVALID_JSON"
        finally:
            await client.close()

    asyncio.run(run())
