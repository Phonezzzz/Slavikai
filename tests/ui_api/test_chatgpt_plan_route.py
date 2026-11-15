from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from config.http_server_config import HttpAuthConfig
from llm.chatgpt_plan import ChatGPTPlanError
from server.http.app import create_app
from server.http.common.request_identity import CloudflareAccessTokenError, VerifiedAccessClaims
from server.http.common.workspace_runtime import _resolve_workspace_file
from server.ui_session_storage import InMemoryUISessionStorage
from shared.models import ToolRequest
from tools.workspace_tools import (
    ListFilesTool,
    ReadFileTool,
    RenameFileTool,
    workspace_root_context,
)

PROVIDER = "chatgpt-plan-" + "a" * 32
OWNER = "email:owner@example.com"


def test_plan_state_is_private_even_when_workspace_contains_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / ".local" / "state" / "slavikai" / "chatgpt-plan"
    state.mkdir(parents=True)
    private_record = state / "example.json"
    private_record.write_text("fixture", encoding="utf-8")
    alias = tmp_path / "browse"
    alias.symlink_to(state, target_is_directory=True)
    monkeypatch.setattr("shared.sandbox.CHATGPT_PLAN_STORAGE", state)
    with workspace_root_context(tmp_path):
        listing = ListFilesTool().handle(ToolRequest("workspace_list", {"recursive": True}))
        assert listing.ok and "browse" not in json.dumps(listing.data["tree"])
        for target in [private_record, alias / "example.json"]:
            rel = str(target.relative_to(tmp_path))
            assert not ReadFileTool().handle(ToolRequest("workspace_read", {"path": rel})).ok
            with pytest.raises(ValueError, match="outside workspace"):
                _resolve_workspace_file(rel, workspace_root=tmp_path, max_download_bytes=100)
        rename = RenameFileTool().handle(
            ToolRequest(
                "workspace_rename",
                {"old_path": ".local/state/slavikai", "new_path": "relocated"},
            )
        )
        assert not rename.ok
        assert private_record.is_file()
        source = tmp_path / "work"
        source.mkdir()
        monkeypatch.setattr(
            "tools.workspace_tools.read_mount_bindings",
            lambda: [(Path("/"), Path("/")), (source / "mounted", Path("/volume"))],
        )
        rename_with_mount = RenameFileTool().handle(
            ToolRequest("workspace_rename", {"old_path": "work", "new_path": "relocated"})
        )
        assert not rename_with_mount.ok
        assert source.is_dir()


class Verifier:
    async def verify(self, token: str) -> VerifiedAccessClaims:
        if token not in {"owner", "member"}:
            raise CloudflareAccessTokenError("invalid")
        return VerifiedAccessClaims(email=f"{token}@example.com")


class Plan:
    def __init__(self) -> None:
        self.connected = True

    def authorization_status(self, principal: str) -> None:
        assert principal == OWNER

    def accounts(self, principal: str) -> list[dict[str, str]]:
        assert principal == OWNER
        return [
            {
                "id": "account",
                "label": "Personal",
                "provider": PROVIDER if self.connected else "",
                "status": "connected" if self.connected else "disconnected",
            }
        ]

    def catalog(self, principal: str, provider: str) -> dict[str, str]:
        assert (principal, provider) == (OWNER, PROVIDER)
        return {"gpt-accessible": "Account model"}

    def validate_grant(self, principal: str, provider: str) -> None:
        assert (principal, provider) == (OWNER, PROVIDER)
        if not self.connected:
            raise ChatGPTPlanError("chatgpt_plan_grant_unavailable", "Disconnected.", status=409)

    def access(self, principal: str, provider: str) -> str:
        self.validate_grant(principal, provider)
        return "test-token"

    def disconnect(self, principal: str, account: str) -> bool:
        assert (principal, account) == (OWNER, "account")
        self.connected = False
        return True

    def begin(
        self, principal: str, account_id: str | None = None, *, reconsent: bool = False
    ) -> str:
        assert (principal, account_id, reconsent) == (OWNER, None, False)
        return "https://auth.openai.com/api/accounts/authorize"

    def close(self) -> None:
        pass


def test_owner_plan_is_explicit_and_uses_account_catalog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("server.http_api.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("SLAVIK_CHATGPT_PLAN_LOCAL", "true")
    monkeypatch.setattr("server.http.app.ChatGPTPlan", lambda _directory: Plan())
    requests: list[dict] = []

    def stream(*, payload: dict, **kwargs: object) -> Iterator[str]:
        assert kwargs["headers"] == {
            "Authorization": "Bearer test-token",
            "Accept": "text/event-stream",
        }
        assert payload["model"] == "gpt-accessible"
        assert payload["store"] is False and payload["stream"] is True
        requests.append(payload)
        if len(requests) == 1:
            assert "workspace_list" in json.dumps(payload["tools"])
            output = {
                "type": "function_call",
                "status": "completed",
                "call_id": "call_list",
                "name": "workspace_list",
                "namespace": "slavikai",
                "arguments": "{}",
            }
        else:
            observations = [
                item for item in payload["input"] if item.get("type") == "function_call_output"
            ]
            assert observations[0]["call_id"] == "call_list"
            assert "fixture.txt" in observations[0]["output"]
            output = {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "fixture.txt is present."}],
            }
        event = {
            "type": "response.completed",
            "response": {"status": "completed", "output": [output]},
        }
        yield "data: " + json.dumps(event)
        yield ""

    monkeypatch.setattr("llm.chatgpt_plan_brain.post_stream_lines", stream)
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "fixture.txt").write_text("data\n", encoding="utf-8")
    asyncio.run(_assert_plan_routes(workspace))
    assert len(requests) == 2


async def _assert_plan_routes(workspace: Path) -> None:
    app = create_app(
        ui_storage=InMemoryUISessionStorage(),
        auth_config=HttpAuthConfig(
            api_token="automation-token",
            browser_auth_mode="cloudflare",
            cloudflare_access_issuer="https://example.cloudflareaccess.com",
            cloudflare_access_aud="aud",
            owner_email="owner@example.com",
        ),
        cloudflare_access_verifier=Verifier(),
    )
    async with TestClient(TestServer(app)) as client:
        owner = {"Cf-Access-Jwt-Assertion": "owner"}
        member = {"Cf-Access-Jwt-Assertion": "member"}
        path = "/ui/api/chatgpt-plan"
        status = await client.get(path, headers=owner)
        assert status.status == 200
        assert status.headers["Cache-Control"] == "no-store"
        assert (await status.json())["accounts"][0]["provider"] == PROVIDER
        assert (await client.get(path, headers=member)).status == 403
        assert (
            await client.get(f"/ui/api/models?provider={PROVIDER}", headers=member)
        ).status == 403
        denied = await client.post(
            f"{path}/authorize", headers={**owner, "Origin": "https://elsewhere.test"}, json={}
        )
        assert denied.status == 403
        same_origin = str(client.make_url("/").origin())
        begin = await client.post(
            f"{path}/authorize", headers={**owner, "Origin": same_origin}, json={}
        )
        assert begin.status == 200
        assert (await begin.json())["authorization_url"].startswith("https://auth.openai.com/")

        discovery = await client.get(f"/ui/api/models?provider={PROVIDER}", headers=owner)
        assert discovery.status == 200
        assert (await discovery.json())["providers"][0]["model_display_names"] == {
            "gpt-accessible": "Account model"
        }
        session = await client.post("/ui/api/sessions", headers=owner)
        session_id = (await session.json())["session"]["session_id"]
        selection_headers = {**owner, "X-Slavik-Session": session_id, "Origin": same_origin}
        select = await client.post(
            "/ui/api/session-model",
            headers=selection_headers,
            json={"provider": PROVIDER, "model": "unlisted"},
        )
        assert select.status == 404
        select = await client.post(
            "/ui/api/session-model",
            headers=selection_headers,
            json={"provider": PROVIDER, "model": "gpt-accessible"},
        )
        assert select.status == 200
        await app["ui_hub"].set_workspace_root(session_id, str(workspace))
        sent = await client.post(
            "/ui/api/chat/send",
            headers=selection_headers,
            json={"content": "List files in the workspace", "stream": True},
        )
        assert sent.status == 200
        assert "fixture.txt is present." in await sent.text()
        disconnected = await client.post(
            f"{path}/disconnect",
            headers={**owner, "Origin": same_origin},
            json={"account_id": "account"},
        )
        assert (await disconnected.json())["revoked"] is True
        stale = await client.post(
            "/ui/api/chat/send", headers=selection_headers, json={"content": "Try again"}
        )
        assert stale.status == 409
        assert (await stale.json())["error"]["code"] == "chatgpt_plan_grant_unavailable"
        hub = app["ui_hub"]
        await hub.set_session_decision(
            session_id,
            {
                "id": "pending",
                "status": "pending",
                "blocking": True,
                "context": {
                    "source_endpoint": "chat.tool_continue",
                    "resume_payload": {"continuation_id": "test"},
                },
            },
        )
        cancelled = await client.post("/ui/api/chat/cancel", headers=selection_headers)
        assert cancelled.status == 200
        assert (await cancelled.json())["cancelled"] is True
        assert (await hub.get_session_decision(session_id))["status"] == "rejected"
        automation = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer automation-token", "X-Slavik-Session": session_id},
            json={"model": "slavik", "messages": [{"role": "user", "content": "test"}]},
        )
        assert automation.status == 403
