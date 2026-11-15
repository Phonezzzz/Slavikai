from __future__ import annotations

import base64
import hashlib
import json
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from llm import chatgpt_plan as plan_module
from llm.chatgpt_plan import ChatGPTPlan, ChatGPTPlanError

PRINCIPAL = "owner@example.test"
OTHER = "other@example.test"


def _b64(value):
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _sign(key, claims):
    header = _b64(json.dumps({"alg": "RS256", "kid": "test-key"}).encode())
    payload = _b64(json.dumps(claims).encode())
    data = f"{header}.{payload}"
    signature = key.sign(data.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{data}.{_b64(signature)}"


@pytest.fixture
def simulated(tmp_path, monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": "test-key",
        "e": _b64(public.e.to_bytes(3)),
        "n": _b64(public.n.to_bytes(256)),
    }
    pages = {}
    calls = []
    options = {}

    def fake_request(method, url, *, data=None, headers=None, empty=False):
        calls.append((method, url, data, headers))
        if url == plan_module.TOKEN_URL:
            if data["grant_type"] == "refresh_token":
                if "refresh_error" in options:
                    raise options["refresh_error"]
                refreshed = {
                    "token_type": "Bearer",
                    "access_token": "refreshed-secret",
                    "refresh_token": "rotated-refresh",
                    "expires_in": 3600,
                }
                if "refresh_scope" in options:
                    refreshed["scope"] = options["refresh_scope"]
                return refreshed
            query = pages[data["code"]]
            assert data["redirect_uri"] == query["redirect_uri"]
            digest = hashlib.sha256(data["code_verifier"].encode()).digest()
            assert _b64(digest) == query["code_challenge"]
            claims = {
                "iss": plan_module.ISSUER,
                "aud": data["client_id"],
                "sub": "test-subject",
                "nonce": query["nonce"],
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
                "email": "chatgpt@example.test",
                "email_verified": True,
            }
            claims.update(options.get("claims", {}))
            id_token = _sign(key, claims)
            if options.get("bad_signature"):
                id_token += "wrong-signature"
            return {
                "id_token": id_token,
                "token_type": "Bearer",
                "access_token": "access-secret",
                "refresh_token": "refresh-secret",
                "scope": options.get("scope", " ".join(plan_module.SCOPES)),
                "expires_in": options.get("duration", 3600),
            }
        if url == plan_module.JWKS_URL:
            return {"keys": [jwk]}
        if url == plan_module.DISCOVERY_URL:
            return {
                "revocation_endpoint": options.get("revoke_url", f"{plan_module.ISSUER}/revoke")
            }
        if url == f"{plan_module.ISSUER}/revoke":
            assert empty
            if options.get("revoke_error"):
                raise options["revoke_error"]
            return {}
        assert url == f"{plan_module.RESOURCE}/models"
        return {
            "models": [
                {"slug": "gpt-b", "display_name": "Second", "visibility": "list"},
                {"slug": "hidden", "display_name": "Hidden", "visibility": "hide"},
                {"slug": "gpt-a", "display_name": "First", "visibility": "list"},
            ]
        }

    monkeypatch.setattr(plan_module, "_request", fake_request)
    service = ChatGPTPlan(tmp_path / "credentials")
    try:
        yield service, pages, calls, options
    finally:
        service.close()


def _authorization(service, pages, *, account_id=None, code="test-auth-code", principal=PRINCIPAL):
    parsed = urlsplit(service.begin(principal, account_id))
    assert (parsed.scheme, parsed.hostname, parsed.path) == (
        "https",
        "auth.openai.com",
        "/api/accounts/authorize",
    )
    query = {key: value[0] for key, value in parse_qs(parsed.query).items()}
    pages[code] = query
    assert "id_token_hint" not in query
    assert query["redirect_uri"].startswith("http://127.0.0.1:")
    assert query["redirect_uri"].endswith("/auth/callback")
    return query


def _callback(query, *, code="test-auth-code", client="registration-client", **extra):
    params = {"state": query["state"], "code": code}
    if client is not None:
        params["client_id"] = client
    params.update(extra)
    with requests.Session() as browser:
        browser.trust_env = False
        return browser.get(query["redirect_uri"], params=params, timeout=5)


def test_registration_pkce_and_catalog_are_principal_scoped(simulated, capsys):
    service, pages, calls, _ = simulated
    query = _authorization(service, pages)
    assert query["client_id"] == "dynamic_agent_client"
    assert query["agent_name_hint"] == "SlavikAI"
    assert service.authorization_status(PRINCIPAL) == "pending"

    assert _callback(query, state="wrong-state").status_code == 400
    assert calls == []
    result = _callback(query)
    assert result.status_code == 200
    assert result.headers["Cache-Control"] == "no-store"
    assert "test-auth-code" not in result.text
    try:
        replay = _callback(query)
    except requests.ConnectionError:
        pass
    else:
        assert replay.status_code == 400
    assert sum(call[1] == plan_module.TOKEN_URL for call in calls) == 1
    (row,) = service.accounts(PRINCIPAL)
    assert row["status"] == "connected"
    assert row["label"].startswith("chatgpt@example.test")
    assert len(row["provider"]) == len("chatgpt-plan-") + 32
    assert service.authorization_status(PRINCIPAL) == "connected"
    assert service.accounts(OTHER) == []
    with pytest.raises(ChatGPTPlanError):
        service.access(OTHER, row["provider"])
    assert list(service.catalog(PRINCIPAL, row["provider"]).items()) == [
        ("gpt-b", "Second"),
        ("gpt-a", "First"),
    ]
    assert calls[-1][-1] == {"Authorization": "Bearer access-secret"}
    assert "test-auth-code" not in capsys.readouterr().err


def test_reauth_rotates_provider_and_preserves_two_registrations(simulated):
    service, pages, _, _ = simulated
    first = _authorization(service, pages)
    assert _callback(first).status_code == 200
    (original,) = service.accounts(PRINCIPAL)
    other = _authorization(service, pages, code="second")
    assert _callback(other, client="second-registration", code="second").status_code == 200
    before = {row["id"]: row for row in service.accounts(PRINCIPAL)}
    assert len(before) == 2
    assert len({row["label"] for row in before.values()}) == 2
    reauth = _authorization(service, pages, account_id=original["id"], code="reauth")
    assert reauth["client_id"] == "registration-client"
    assert reauth["ext_agent_host_id"] == first["ext_agent_host_id"]
    assert "agent_name_hint" not in reauth
    assert _callback(reauth, client=None, code="reauth").status_code == 200
    after = {row["id"]: row for row in service.accounts(PRINCIPAL)}
    assert list(after) == list(before)
    assert after[original["id"]]["provider"] != original["provider"]
    with pytest.raises(ChatGPTPlanError):
        service.validate_grant(PRINCIPAL, original["provider"])
    assert all(after[key] == row for key, row in before.items() if key != original["id"])


@pytest.mark.parametrize(
    "claims",
    [
        {"nonce": "bad"},
        {"aud": "wrong"},
        {"iss": "https://bad.invalid"},
        {"exp": 1},
    ],
)
def test_invalid_signed_identity_cannot_register(simulated, claims):
    service, pages, _, options = simulated
    query = _authorization(service, pages)
    options["claims"] = claims
    result = _callback(query)
    assert result.status_code == 400
    assert service.accounts(PRINCIPAL) == []
    assert service.authorization_status(PRINCIPAL) == "failed"


def test_no_plan_scope_registers_account_without_provider(simulated):
    service, pages, _, options = simulated
    query = _authorization(service, pages)
    options["scope"] = "openid profile email"
    _callback(query)
    (row,) = service.accounts(PRINCIPAL)
    assert row["status"] == "disabled"
    assert row["provider"] is None
    assert service.authorization_status(PRINCIPAL) == "disabled"


def test_newer_attempt_invalidates_callback_and_redacts_errors(simulated):
    service, pages, calls, _ = simulated
    old = _authorization(service, pages)
    new = _authorization(service, pages, code="new")
    try:
        result = _callback(old)
    except requests.ConnectionError:
        pass
    else:
        assert result.status_code == 400
    assert calls == []
    result = requests.get(
        new["redirect_uri"] + "?" + urlencode({"state": new["state"], "error": "access_denied"}),
        timeout=5,
    )
    assert result.status_code == 400
    assert calls == []
    assert service.authorization_status(PRINCIPAL) == "failed"


def test_untrusted_signature_cannot_register(simulated):
    service, pages, _, options = simulated
    query = _authorization(service, pages)
    options["bad_signature"] = True
    assert _callback(query).status_code == 400
    assert service.accounts(PRINCIPAL) == []


def test_signed_access_token_hash_must_match(simulated):
    service, pages, _, options = simulated
    query = _authorization(service, pages)
    options["claims"] = {"at_hash": "unpaired-access-token"}
    assert _callback(query).status_code == 400
    assert service.accounts(PRINCIPAL) == []

    query = _authorization(service, pages)
    options["claims"] = {
        "at_hash": _b64(hashlib.sha256(b"access-secret").digest()[:16]),
    }
    assert _callback(query).status_code == 200


def test_callback_requires_exact_path_and_authority(simulated):
    service, pages, calls, _ = simulated
    query = _authorization(service, pages)
    parsed = urlsplit(query["redirect_uri"])
    parameters = urlencode({"state": query["state"], "code": "secret"})
    connection = HTTPConnection(parsed.hostname, parsed.port, timeout=3)
    try:
        connection.request("GET", f"/{parsed.path}?{parameters}")
        response = connection.getresponse()
        assert response.status == 400
        assert b"secret" not in response.read()
        connection.request("GET", f"{parsed.path}?{parameters}", headers={"Host": "localhost"})
        response = connection.getresponse()
        assert response.status == 400
        response.read()
    finally:
        connection.close()
    assert calls == []
    assert _callback(query).status_code == 200


def test_mismatched_reauthorization_keeps_current_grant(simulated):
    service, pages, _, options = simulated
    query = _authorization(service, pages)
    assert _callback(query).status_code == 200
    (row,) = service.accounts(PRINCIPAL)
    reauth = _authorization(service, pages, account_id=row["id"], code="reauth")
    options["claims"] = {"sub": "other-subject"}
    assert _callback(reauth, code="reauth").status_code == 400
    assert service.accounts(PRINCIPAL) == [row]
    assert service.access(PRINCIPAL, row["provider"]) == "access-secret"


def test_second_instance_cannot_complete_superseded_attempt(simulated):
    service, pages, calls, _ = simulated
    other = ChatGPTPlan(service.directory)
    try:
        old = _authorization(service, pages)
        new = _authorization(other, pages, code="new")
        assert _callback(old).status_code == 400
        assert calls == []
        assert _callback(new, code="new").status_code == 200
        assert service.accounts(PRINCIPAL) == other.accounts(PRINCIPAL)
    finally:
        other.close()


def test_credential_permissions_and_cross_instance_refresh_rotation(simulated):
    service, pages, calls, options = simulated
    options["duration"] = 1
    query = _authorization(service, pages)
    assert _callback(query).status_code == 200
    (row,) = service.accounts(PRINCIPAL)
    assert stat.S_IMODE(service.directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in service.directory.iterdir())
    other = ChatGPTPlan(service.directory)
    try:
        other.validate_grant(PRINCIPAL, row["provider"])
        assert not any((call[2] or {}).get("grant_type") == "refresh_token" for call in calls)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(service.access, PRINCIPAL, row["provider"])
            second = pool.submit(other.access, PRINCIPAL, row["provider"])
            assert first.result(timeout=5) == second.result(timeout=5) == "refreshed-secret"
        (refresh,) = [
            call for call in calls if (call[2] or {}).get("grant_type") == "refresh_token"
        ]
        assert refresh[2] == {
            "grant_type": "refresh_token",
            "client_id": "registration-client",
            "refresh_token": "refresh-secret",
            "resource": plan_module.RESOURCE,
        }
        assert other.accounts(PRINCIPAL) == [row]
        assert service.disconnect(PRINCIPAL, row["id"])
        assert calls[-1][2] == {
            "token": "rotated-refresh",
            "token_type_hint": "refresh_token",
            "client_id": "registration-client",
        }
        assert other.accounts(PRINCIPAL) == [{**row, "status": "disconnected", "provider": None}]
        with pytest.raises(ChatGPTPlanError, match="отсутствует или изменился"):
            other.access(PRINCIPAL, row["provider"])
    finally:
        other.close()


def test_definitive_refresh_failure_invalidates_grant_and_retains_registration(simulated):
    service, pages, _, options = simulated
    options["duration"] = 1
    query = _authorization(service, pages)
    assert _callback(query).status_code == 200
    (row,) = service.accounts(PRINCIPAL)
    options["refresh_error"] = ChatGPTPlanError("chatgpt_plan_transport", "temporary", status=503)
    with pytest.raises(ChatGPTPlanError, match="temporary"):
        service.access(PRINCIPAL, row["provider"])
    assert service.accounts(PRINCIPAL) == [row]
    options["refresh_error"] = ChatGPTPlanError("chatgpt_plan_revoked", "revoked", status=409)
    with pytest.raises(ChatGPTPlanError, match="revoked"):
        service.access(PRINCIPAL, row["provider"])
    assert service.accounts(PRINCIPAL) == [{**row, "status": "expired", "provider": None}]
    parsed = urlsplit(service.begin(PRINCIPAL, row["id"], reconsent=True))
    reauth = parse_qs(parsed.query)
    assert reauth["client_id"] == ["registration-client"]
    assert reauth["prompt"] == ["consent"]


def test_refresh_withdrawal_disables_grant_and_revokes_new_token_on_signout(simulated):
    service, pages, calls, options = simulated
    options["duration"] = 1
    query = _authorization(service, pages)
    assert _callback(query).status_code == 200
    (row,) = service.accounts(PRINCIPAL)
    options["refresh_scope"] = "openid profile email offline_access resource.invoke"
    with pytest.raises(ChatGPTPlanError) as caught:
        service.access(PRINCIPAL, row["provider"])
    assert caught.value.code == "chatgpt_plan_permission"
    assert service.accounts(PRINCIPAL) == [{**row, "status": "disabled", "provider": None}]
    assert service.disconnect(PRINCIPAL, row["id"])
    assert calls[-1][2]["token"] == "rotated-refresh"


def test_revoke_unconfirmed_still_disconnects_locally(simulated, monkeypatch):
    service, pages, calls, options = simulated
    query = _authorization(service, pages)
    assert _callback(query).status_code == 200
    (row,) = service.accounts(PRINCIPAL)
    options["revoke_error"] = ChatGPTPlanError("chatgpt_plan_transport", "temporary", status=503)
    monkeypatch.setattr(plan_module.time, "sleep", lambda _: None)
    assert not service.disconnect(PRINCIPAL, row["id"])
    assert len([call for call in calls if call[1].endswith("/revoke")]) == 3
    assert service.accounts(PRINCIPAL) == [{**row, "status": "disconnected", "provider": None}]
    with pytest.raises(ChatGPTPlanError):
        service.access(PRINCIPAL, row["provider"])


def test_discovery_cannot_send_refresh_token_to_untrusted_url(simulated):
    service, pages, calls, options = simulated
    query = _authorization(service, pages)
    assert _callback(query).status_code == 200
    (row,) = service.accounts(PRINCIPAL)
    options["revoke_url"] = f"{plan_module.ISSUER}@bad.invalid/revoke"
    assert not service.disconnect(PRINCIPAL, row["id"])
    assert calls[-1][1] == plan_module.DISCOVERY_URL


def test_storage_refuses_group_readable_record(simulated):
    service, pages, _, _ = simulated
    query = _authorization(service, pages)
    assert _callback(query).status_code == 200
    record = service.directory / (hashlib.sha256(PRINCIPAL.encode()).hexdigest() + ".json")
    record.chmod(0o640)
    with pytest.raises(ChatGPTPlanError, match="Хранилище"):
        service.accounts(PRINCIPAL)


def test_catalog_rejection_does_not_disclose_raw_upstream_details(monkeypatch):
    session_factory = MagicMock()
    session = session_factory.return_value.__enter__.return_value
    response = session.request.return_value
    response.status_code = 403
    response.headers = {"x-request-id": "req_42"}
    response.json.return_value = {"detail": "redact-upstream-secret"}
    monkeypatch.setattr(plan_module.requests, "Session", session_factory)
    with pytest.raises(ChatGPTPlanError) as caught:
        plan_module._request("GET", f"{plan_module.RESOURCE}/models")
    assert caught.value.status == caught.value.upstream_status == 403
    assert caught.value.body_shape == "detail"
    assert caught.value.request_id == "req_42"
    assert "redact-upstream-secret" not in str(caught.value)
    assert not session.trust_env
    assert session.request.call_args.kwargs["allow_redirects"] is False
