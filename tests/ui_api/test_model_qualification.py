from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from config.api_keys import save_api_keys
from config.http_server_config import HttpAuthConfig
from server import http_api as api
from server.agent_provider import AgentScope
from server.http.app import create_app
from server.http.common import ui_settings
from server.http.common.request_identity import VerifiedAccessClaims
from server.ui_session_storage import InMemoryUISessionStorage
from tests.fake_provider_http import FakeHttpResponse


class Verifier:
    async def verify(self, token):
        return VerifiedAccessClaims(
            email="owner@test.invalid" if token == "owner" else "member@test.invalid"
        )


def test_corrupt_key_store_revokes_context(monkeypatch, tmp_path):
    from server.http.handlers.qualification import _context_validator

    config, calls = setup(monkeypatch, tmp_path)
    current = _context_validator(config, "fixture-key")
    assert current()
    api.API_KEYS_PATH.write_text("{broken", encoding="utf-8")
    assert not current()
    assert calls == []


def setup(monkeypatch, tmp_path, *, malformed=False, rotate=False):
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(api, "UI_SETTINGS_PATH", tmp_path / "ui.json")
    monkeypatch.setattr(ui_settings, "UI_SETTINGS_PATH", tmp_path / "ui.json")
    monkeypatch.setattr(api, "API_KEYS_PATH", tmp_path / "keys.json")
    monkeypatch.setattr("server.http.app.load_model_configs", lambda: None)
    monkeypatch.setattr(
        "server.http.common.runtime_contract._resolve_provider_api_key",
        lambda provider: api._resolve_provider_api_key(provider),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "")
    provider = ui_settings._save_provider_instance(
        display_name="fixture", base_url="https://provider.test/v1"
    )
    save_api_keys({provider: "fixture-key"}, path=api.API_KEYS_PATH)
    config = ui_settings._build_model_config(provider, "opaque")
    calls = []
    original = aiohttp.ClientSession._request

    class SSE(FakeHttpResponse):
        async def _lines(self):
            choice = self.payload["choices"][0]
            message = choice["message"]
            if message.get("tool_calls"):
                message["tool_calls"][0]["index"] = 0
            yield (
                "data: "
                + json.dumps(
                    {"choices": [{"delta": message, "finish_reason": choice["finish_reason"]}]}
                )
                + "\n"
            ).encode()
            yield b"data: [DONE]\n"

    async def request(session, method, url, **kwargs):
        if "provider.test" not in str(url):
            return await original(session, method, url, **kwargs)
        body = kwargs["json"]
        calls.append(body)
        if body["messages"][-1]["role"] == "user":
            nonce = body["tools"][0]["function"]["parameters"]["properties"]["nonce"]["enum"][0]
            message = {
                "content": "",
                "tool_calls": [
                    {
                        "id": "fixture-call",
                        "type": "function",
                        "function": {
                            "name": "inert_fixture",
                            "arguments": json.dumps({"nonce": "wrong" if malformed else nonce}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            assert body["messages"][-1]["tool_call_id"] == "fixture-call"
            message, finish = {"content": body["messages"][-1]["content"]}, "stop"
        if rotate and len(calls) == 1:
            save_api_keys({provider: "rotated-fixture-key"}, path=api.API_KEYS_PATH)
        response_type = SSE if body.get("stream") else FakeHttpResponse
        return response_type(payload={"choices": [{"message": message, "finish_reason": finish}]})

    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)
    return config, calls


async def client_and_session(config):
    app = create_app(
        ui_storage=InMemoryUISessionStorage(),
        auth_config=HttpAuthConfig(
            api_token="fixture-automation",
            browser_auth_mode="cloudflare",
            owner_email="owner@test.invalid",
            cloudflare_access_issuer="https://fixture.cloudflareaccess.com",
            cloudflare_access_aud="fixture",
        ),
        cloudflare_access_verifier=Verifier(),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    principal = "email:owner@test.invalid"
    session = await app["ui_hub"].create_session(principal)
    await app["runtime_model_state"].set_session_override(session, config)
    return client, app, session, principal


def test_owner_inert_qualification_is_scoped_and_revoked(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)

    async def run():
        client, app, session, principal = await client_and_session(config)
        owner = {"Cf-Access-Jwt-Assertion": "owner"}
        url = f"/ui/api/sessions/{session}/model/qualify"
        try:
            before = await client.get(url.replace("qualify", "capabilities"), headers=owner)
            assert (await before.json())["state"] == "unknown"
            no_confirm = await client.post(url, json={}, headers=owner)
            assert no_confirm.status == 400
            member = await client.post(
                url, json={"confirm": True}, headers={"Cf-Access-Jwt-Assertion": "member"}
            )
            assert member.status == 403
            assert not calls
            response = await client.post(url, json={"confirm": True}, headers=owner)
            assert response.status == 200, await response.text()
            assert (await response.json())["state"] == "supported"
            assert len(calls) == 4
            assert [bool(body.get("stream")) for body in calls] == [False, False, True, True]
            provider = app["agent_provider"]
            agent = await provider.get_existing_for_current_task(AgentScope(principal, session))
            assert agent.brain.native_tool_capability().allowed
            other = await app["ui_hub"].create_session(principal)
            await app["runtime_model_state"].set_session_override(other, config)
            other_agent = await provider.get_for_current_task(AgentScope(principal, other), config)
            assert not other_agent.brain.native_tool_capability().allowed
            agent.reconfigure_models(config, main_api_key="fixture-key", persist=False)
            assert agent.brain.native_tool_capability().allowed
            save_api_keys({config.provider: "different-fixture"}, path=api.API_KEYS_PATH)
            assert not agent.brain.native_tool_capability().allowed
            save_api_keys({config.provider: "fixture-key"}, path=api.API_KEYS_PATH)
            assert not agent.brain.native_tool_capability().allowed
        finally:
            await client.close()

    asyncio.run(run())


def test_protocol_failure_is_unsupported_without_host_dispatch(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path, malformed=True)

    async def run():
        client, app, session, principal = await client_and_session(config)
        try:
            response = await client.post(
                f"/ui/api/sessions/{session}/model/qualify",
                json={"confirm": True},
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert response.status == 200, await response.text()
            assert (await response.json())["state"] == "unsupported"
            assert len(calls) == 1
            agent = await app["agent_provider"].get_existing_for_current_task(
                AgentScope(principal, session)
            )
            assert not agent.brain.native_tool_capability().allowed
            assert agent.brain.native_tool_capability().model.state == "unsupported"
        finally:
            await client.close()

    asyncio.run(run())


def test_model_selection_away_and_back_revokes_without_chat(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_settings, "_probe_openai_models", lambda *_: (["opaque", "other"], "ready", None)
    )

    async def run():
        client, app, session, principal = await client_and_session(config)
        headers = {"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session}
        try:
            checked = await client.post(
                f"/ui/api/sessions/{session}/model/qualify",
                json={"confirm": True},
                headers=headers,
            )
            assert checked.status == 200
            agent = await app["agent_provider"].get_existing_for_current_task(
                AgentScope(principal, session)
            )
            assert agent.brain.native_tool_capability().allowed
            for model in ("other", "opaque"):
                response = await client.post(
                    "/ui/api/session-model",
                    json={"provider": config.provider, "model": model},
                    headers=headers,
                )
                assert response.status == 200, await response.text()
                assert not agent.brain.native_tool_capability().allowed
            assert len(calls) == 4
        finally:
            await client.close()

    asyncio.run(run())


def test_rotation_during_probe_stops_before_continuation(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path, rotate=True)

    async def run():
        client, app, session, principal = await client_and_session(config)
        try:
            response = await client.post(
                f"/ui/api/sessions/{session}/model/qualify",
                json={"confirm": True},
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert response.status == 409, await response.text()
            assert len(calls) == 1
            agent = await app["agent_provider"].get_existing_for_current_task(
                AgentScope(principal, session)
            )
            assert not agent.brain.native_tool_capability().allowed
        finally:
            await client.close()

    asyncio.run(run())


def test_disconnect_cancels_probe_and_never_admits_tools(monkeypatch, tmp_path):
    from contextlib import suppress

    from tests.fake_provider_http import HoldingProvider

    config, _ = setup(monkeypatch, tmp_path)
    upstream = HoldingProvider(monkeypatch, "body")

    async def run():
        client, app, session, principal = await client_and_session(config)
        pending = None
        try:
            pending = asyncio.create_task(
                client.post(
                    f"/ui/api/sessions/{session}/model/qualify",
                    json={"confirm": True},
                    headers={"Cf-Access-Jwt-Assertion": "owner"},
                )
            )
            assert await asyncio.to_thread(upstream.started.wait, 2)
            lock = app["agent_provider"].lock_for(AgentScope(principal, session))
            assert lock.locked()
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
            for _ in range(30):
                if not lock.locked():
                    break
                await asyncio.sleep(0.05)
            assert not lock.locked()
            assert upstream.calls == 1
            assert upstream.active == 0
            assert all(response.closed for response in upstream.responses)
            agent = await app["agent_provider"].get_existing_for_current_task(
                AgentScope(principal, session)
            )
            assert not agent.brain.native_tool_capability().allowed
        finally:
            if pending is not None:
                pending.cancel()
            await client.close()

    asyncio.run(run())


def test_request_only_config_keeps_qualification_and_streaming(monkeypatch, tmp_path):
    from dataclasses import replace

    config, _ = setup(monkeypatch, tmp_path)

    async def run():
        client, app, session, principal = await client_and_session(config)
        try:
            response = await client.post(
                f"/ui/api/sessions/{session}/model/qualify",
                json={"confirm": True},
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert response.status == 200
            agent = await app["agent_provider"].get_existing_for_current_task(
                AgentScope(principal, session)
            )
            assert agent.brain.can_stream_native_tools()
            for changed in (replace(config, web_search_enabled=True), config):
                agent.reconfigure_models(changed, main_api_key="fixture-key", persist=False)
                assert agent.brain.native_tool_capability().allowed
                assert agent.brain.can_stream_native_tools()
            from shared.models import LLMMessage

            sent = []
            original = aiohttp.ClientSession._request

            class ChatSSE(FakeHttpResponse):
                async def _lines(self):
                    yield (
                        b'data: {"choices":[{"delta":{"content":"fixture answer"},'
                        b'"finish_reason":"stop"}]}\n'
                    )
                    yield b"data: [DONE]\n"

            async def chat_request(client_session, method, url, **kwargs):
                if "provider.test" in str(url):
                    sent.append(kwargs["json"])
                    return ChatSSE()
                return await original(client_session, method, url, **kwargs)

            monkeypatch.setattr(aiohttp.ClientSession, "_request", chat_request)
            expected = {spec.name for spec in agent._chat_read_tool_specs()}
            assert expected
            await asyncio.to_thread(
                lambda: list(
                    agent.respond_stream([LLMMessage(role="user", content="fixture query")])
                )
            )
            assert len(sent) == 1
            assert {tool["function"]["name"] for tool in sent[0]["tools"]} == expected
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": True},
        {"choices": []},
        {"choices": [{}]},
        ["invalid"],
        {"choices": [{"message": {"content": "fixture"}}], "usage": {"total_tokens": None}},
    ],
)
def test_invalid_success_is_protocol_failure(monkeypatch, tmp_path, payload):
    config, calls = setup(monkeypatch, tmp_path)
    original = aiohttp.ClientSession._request

    async def malformed(session, method, url, **kwargs):
        if "provider.test" in str(url):
            calls.append(kwargs["json"])
            return FakeHttpResponse(payload=payload)
        return await original(session, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", malformed)

    async def run():
        client, _, session, _ = await client_and_session(config)
        try:
            response = await client.post(
                f"/ui/api/sessions/{session}/model/qualify",
                json={"confirm": True},
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert response.status == 200
            assert (await response.json())["state"] == "unsupported"
            assert len(calls) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_inert_probe_does_not_retry_paid_requests(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)
    original = aiohttp.ClientSession._request

    async def unavailable(session, method, url, **kwargs):
        if "provider.test" in str(url):
            calls.append(kwargs["json"])
            return FakeHttpResponse(status=503)
        return await original(session, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", unavailable)

    async def run():
        client, _, session, _ = await client_and_session(config)
        try:
            response = await client.post(
                f"/ui/api/sessions/{session}/model/qualify",
                json={"confirm": True},
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert response.status == 200
            assert (await response.json())["state"] == "unknown"
            assert len(calls) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_approval_arriving_at_lock_keeps_brain_and_proof(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)

    async def run():
        client, app, session, principal = await client_and_session(config)
        headers = {"Cf-Access-Jwt-Assertion": "owner"}
        url = f"/ui/api/sessions/{session}/model/qualify"
        try:
            assert (await client.post(url, json={"confirm": True}, headers=headers)).status == 200
            provider = app["agent_provider"]
            agent = await provider.get_existing_for_current_task(AgentScope(principal, session))
            armed = [False]
            captured = []
            pending = object()

            class HandoffLock(asyncio.Lock):
                async def __aenter__(self):
                    await super().__aenter__()
                    if armed[0]:
                        agent.last_approval_request = pending
                        armed[0] = False
                    return self

            lock = HandoffLock()
            monkeypatch.setattr(provider, "lock_for", lambda _: lock)
            resolve = provider.get_for_current_task

            async def handoff(*args):
                result = await resolve(*args)
                captured.append(agent.brain)
                armed[0] = True
                return result

            monkeypatch.setattr(provider, "get_for_current_task", handoff)
            response = await client.post(url, json={"confirm": True}, headers=headers)
            assert response.status == 409
            assert agent.brain is captured[0]
            assert agent.brain.native_tool_capability().allowed
            assert agent.last_approval_request is pending
            assert len(calls) == 4
            # Уже существующий approval также нельзя разрушить при resolve.
            previous_brain = agent.brain
            response = await client.post(url, json={"confirm": True}, headers=headers)
            assert response.status == 409
            assert agent.brain is previous_brain
            assert len(calls) == 4
        finally:
            await client.close()

    asyncio.run(run())


def test_invalid_utf8_stream_is_protocol_failure(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)
    original = aiohttp.ClientSession._request

    class InvalidSSE(FakeHttpResponse):
        async def _lines(self):
            yield b"data: \xff\n"

    async def request(session, method, url, **kwargs):
        if "provider.test" in str(url) and kwargs["json"].get("stream"):
            calls.append(kwargs["json"])
            return InvalidSSE()
        return await original(session, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)

    async def run():
        client, app, session, principal = await client_and_session(config)
        try:
            response = await client.post(
                f"/ui/api/sessions/{session}/model/qualify",
                json={"confirm": True},
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert response.status == 200
            assert (await response.json())["state"] == "unsupported"
            assert len(calls) == 3
            agent = await app["agent_provider"].get_existing_for_current_task(
                AgentScope(principal, session)
            )
            assert not agent.brain.native_tool_capability().allowed
        finally:
            await client.close()

    asyncio.run(run())


def test_same_model_selection_preserves_pending_brain(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_settings, "_probe_openai_models", lambda *_: (["opaque", "other"], "ready", None)
    )

    async def run():
        client, app, session, principal = await client_and_session(config)
        headers = {"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session}
        try:
            agent = await app["agent_provider"].get_for_current_task(
                AgentScope(principal, session), config
            )
            pending = object()
            agent.last_approval_request = pending
            before = agent.brain
            for model, expected_status in (("opaque", 200), ("other", 409)):
                response = await client.post(
                    "/ui/api/session-model",
                    json={"provider": config.provider, "model": model},
                    headers=headers,
                )
                assert response.status == expected_status
                assert agent.brain is before
                assert agent.last_approval_request is pending
            assert calls == []
        finally:
            await client.close()

    asyncio.run(run())


def test_overall_timeout_is_unknown_and_settles_worker(monkeypatch, tmp_path):
    from server.http.handlers import qualification
    from tests.fake_provider_http import HoldingProvider

    config, _ = setup(monkeypatch, tmp_path)
    upstream = HoldingProvider(monkeypatch, "body")
    monkeypatch.setattr(qualification, "QUALIFICATION_TIMEOUT_SECONDS", 0.1)

    async def run():
        client, app, session, principal = await client_and_session(config)
        try:
            response = await client.post(
                f"/ui/api/sessions/{session}/model/qualify",
                json={"confirm": True},
                headers={"Cf-Access-Jwt-Assertion": "owner"},
            )
            assert response.status == 200, await response.text()
            assert (await response.json())["state"] == "unknown"
            assert upstream.calls == 1
            assert upstream.active == 0
            assert all(item.closed for item in upstream.responses)
            provider = app["agent_provider"]
            assert not provider.lock_for(AgentScope(principal, session)).locked()
            agent = await provider.get_existing_for_current_task(AgentScope(principal, session))
            assert not agent.brain.native_tool_capability().allowed
        finally:
            await client.close()

    asyncio.run(run())
