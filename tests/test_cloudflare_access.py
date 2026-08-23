from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from config.http_server_config import HttpAuthConfig
from core.desktop_policy import DesktopPolicyStore
from server.http.app import create_app
from server.http.common.request_identity import (
    CloudflareAccessJWTVerifier,
    CloudflareAccessTokenError,
    VerifiedAccessClaims,
)
from server.ui_session_storage import InMemoryUISessionStorage, PersistedSession
from tests.ui_api.fakes import DummyAgent


class StubAccessVerifier:
    def __init__(self, claims_by_token: Mapping[str, VerifiedAccessClaims]) -> None:
        self.claims_by_token = dict(claims_by_token)

    async def verify(self, token: str) -> VerifiedAccessClaims:
        claims = self.claims_by_token.get(token)
        if claims is None:
            raise CloudflareAccessTokenError("invalid")
        return claims


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _jwt(
    private_key: rsa.RSAPrivateKey,
    payload: dict[str, object],
    *,
    kid: str = "test-key",
) -> str:
    header = {"alg": "RS256", "kid": kid, "typ": "JWT"}
    header_segment = _b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_segment = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header_segment}.{payload_segment}.{_b64url(signature)}"


def _jwk(private_key: rsa.RSAPrivateKey) -> dict[str, object]:
    numbers = private_key.public_key().public_numbers()
    exponent = numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")
    modulus = numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")
    return {
        "kty": "RSA",
        "alg": "RS256",
        "kid": "test-key",
        "e": _b64url(exponent),
        "n": _b64url(modulus),
    }


def test_cloudflare_verifier_validates_signature_and_registered_claims() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    async def fetch_jwks() -> object:
        return {"keys": [_jwk(private_key)]}

    verifier = CloudflareAccessJWTVerifier(
        issuer="https://example.cloudflareaccess.com",
        audience="application-aud",
        jwks_fetcher=fetch_jwks,
        now=lambda: 1_000.0,
    )
    token = _jwt(
        private_key,
        {
            "iss": "https://example.cloudflareaccess.com",
            "aud": ["application-aud"],
            "exp": 1_100,
            "nbf": 900,
            "email": "Owner@Example.COM",
            "sub": "access-user",
        },
    )

    claims = asyncio.run(verifier.verify(token))

    assert claims == VerifiedAccessClaims(email="owner@example.com", subject="access-user")


@pytest.mark.parametrize(
    ("claim", "value", "message"),
    [
        ("iss", "https://wrong.example", "issuer"),
        ("aud", ["wrong-aud"], "audience"),
        ("exp", 999, "expired"),
        ("nbf", 1_001, "not active"),
        ("email", "", "email"),
    ],
)
def test_cloudflare_verifier_rejects_invalid_claims(
    claim: str,
    value: object,
    message: str,
) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    async def fetch_jwks() -> object:
        return {"keys": [_jwk(private_key)]}

    verifier = CloudflareAccessJWTVerifier(
        issuer="https://example.cloudflareaccess.com",
        audience="application-aud",
        jwks_fetcher=fetch_jwks,
        now=lambda: 1_000.0,
    )
    payload: dict[str, object] = {
        "iss": "https://example.cloudflareaccess.com",
        "aud": "application-aud",
        "exp": 1_100,
        "nbf": 900,
        "email": "member@example.com",
    }
    payload[claim] = value

    with pytest.raises(CloudflareAccessTokenError, match=message):
        asyncio.run(verifier.verify(_jwt(private_key, payload)))


def test_cloudflare_verifier_does_not_refetch_unknown_kids_within_cache_ttl() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    fetch_count = 0

    async def fetch_jwks() -> object:
        nonlocal fetch_count
        fetch_count += 1
        return {"keys": [_jwk(private_key)]}

    verifier = CloudflareAccessJWTVerifier(
        issuer="https://example.cloudflareaccess.com",
        audience="application-aud",
        jwks_fetcher=fetch_jwks,
        now=lambda: 1_000.0,
    )

    async def resolve_unknown_kids() -> None:
        for kid in ("unknown-one", "unknown-two"):
            with pytest.raises(CloudflareAccessTokenError, match="kid is unknown"):
                await verifier._key_for_kid(kid)

    asyncio.run(resolve_unknown_kids())

    assert fetch_count == 1


def test_cloudflare_browser_lane_is_separate_from_bearer_automation(tmp_path: Path) -> None:
    async def run() -> None:
        auth_config = HttpAuthConfig(
            api_token="automation-token",
            browser_auth_mode="cloudflare",
            cloudflare_access_issuer="https://example.cloudflareaccess.com",
            cloudflare_access_aud="application-aud",
            owner_email="owner@example.com",
        )
        verifier = StubAccessVerifier(
            {
                "owner-token": VerifiedAccessClaims(email="owner@example.com"),
                "member-token": VerifiedAccessClaims(email="member@example.com"),
            }
        )
        app = create_app(
            agent=DummyAgent(),
            ui_storage=InMemoryUISessionStorage(),
            auth_config=auth_config,
            cloudflare_access_verifier=verifier,
            desktop_policy_store=DesktopPolicyStore(tmp_path / "desktop-approvals.json"),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            assert (await client.get("/ui/api/auth/status")).status == 401
            bearer_on_ui = await client.get(
                "/ui/api/status",
                headers={"Authorization": "Bearer automation-token"},
            )
            assert bearer_on_ui.status == 401

            owner_status = await client.get(
                "/ui/api/auth/status",
                headers={"Cf-Access-Jwt-Assertion": "owner-token"},
            )
            owner_payload = await owner_status.json()
            assert owner_payload["principal_id"] == "email:owner@example.com"
            assert owner_payload["role"] == "owner"

            owner_runtime_status = await client.get(
                "/ui/api/status",
                headers={"Cf-Access-Jwt-Assertion": "owner-token"},
            )
            owner_session_id = (await owner_runtime_status.json())["session_id"]
            automation_runtime = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer automation-token"},
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "read owner state"}],
                    "slavik_meta": {
                        "runtime_mode": "auto",
                        "runtime_session_id": owner_session_id,
                    },
                },
            )
            assert automation_runtime.status == 403
            assert (await automation_runtime.json())["error"]["code"] == "session_forbidden"

            member_persistent = await client.get(
                "/ui/api/desktop/approvals",
                headers={"Cf-Access-Jwt-Assertion": "member-token"},
            )
            assert member_persistent.status == 403
            assert (await member_persistent.json())["error"]["code"] == "owner_required"

            member_settings_update = await client.post(
                "/ui/api/settings",
                headers={"Cf-Access-Jwt-Assertion": "member-token"},
                json={"appearance": {"theme": "oled"}},
            )
            assert member_settings_update.status == 403
            assert (await member_settings_update.json())["error"]["code"] == "owner_required"

            member_status = await client.get(
                "/ui/api/status",
                headers={"Cf-Access-Jwt-Assertion": "member-token"},
            )
            member_session_id = (await member_status.json())["session_id"]
            hub = app["ui_hub"]
            await hub.set_session_decision(
                member_session_id,
                {
                    "id": "member-always-allow",
                    "kind": "approval",
                    "decision_type": "tool_approval",
                    "status": "pending",
                    "blocking": True,
                    "reason": "destructive_action",
                    "summary": "Delete one file",
                    "proposed_action": {
                        "required_categories": ["FS_DELETE_OVERWRITE"],
                        "scope": {
                            "tool": "desktop_file_delete",
                            "action": "delete",
                            "target_pattern": str(tmp_path / "member.txt"),
                            "risk_class": "destructive",
                            "execution_target": "desktop",
                        },
                    },
                    "options": [],
                    "default_option_id": None,
                    "context": {"session_id": member_session_id},
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "resolved_at": None,
                },
            )
            member_always_allow = await client.post(
                "/ui/api/decision/respond",
                headers={
                    "Cf-Access-Jwt-Assertion": "member-token",
                    "X-Slavik-Session": member_session_id,
                },
                json={
                    "session_id": member_session_id,
                    "decision_id": "member-always-allow",
                    "choice": "always_allow",
                },
            )
            assert member_always_allow.status == 403
            assert (await member_always_allow.json())["error"]["code"] == "owner_required"

            cloudflare_on_v1 = await client.get(
                "/v1/models",
                headers={"Cf-Access-Jwt-Assertion": "owner-token"},
            )
            assert cloudflare_on_v1.status == 401
            bearer_on_v1 = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer automation-token"},
            )
            assert bearer_on_v1.status == 200
        finally:
            await client.close()

    asyncio.run(run())


def test_cloudflare_owner_receives_legacy_sessions(tmp_path: Path) -> None:
    storage = InMemoryUISessionStorage()
    storage.save_session(
        PersistedSession(
            session_id="legacy-session",
            principal_id="legacy",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            status="ok",
            decision=None,
            messages=[],
        )
    )
    app = create_app(
        agent=DummyAgent(),
        ui_storage=storage,
        auth_config=HttpAuthConfig(
            api_token="automation-token",
            browser_auth_mode="cloudflare",
            cloudflare_access_issuer="https://example.cloudflareaccess.com",
            cloudflare_access_aud="application-aud",
            owner_email="owner@example.com",
        ),
        cloudflare_access_verifier=StubAccessVerifier({}),
        desktop_policy_store=DesktopPolicyStore(tmp_path / "desktop-approvals.json"),
    )

    sessions = storage.load_sessions()

    assert sessions[0].principal_id == "email:owner@example.com"
    assert app["ui_hub"] is not None
