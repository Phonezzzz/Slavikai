"""E02-T04: ручной вход через UI/API (``NEEDS_USER_AUTH``).

Владелец run видит решение «Войди в окне Chromium на ПК → Готово» (в том
числе с телефона через session API), «Готово» продолжает run ровно один
раз, чужое и повторное нажатие отклоняются, просроченное «Готово» не
возобновляет run, а явный retry открывает новую попытку. Headed Chromium в
тестах не запускается: Playwright заменён fake-окном.
"""

from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio
import time
from pathlib import Path

import pytest

from core.agent import Agent
from core.browser_service import BrowserProfile
from core.desktop_policy import DesktopPolicyStore
from core.desktop_security import DesktopPathSecurity
from llm.brain_base import Brain
from llm.types import LLMResult, ToolCall
from tools import desktop_browser_tool
from tools.desktop_browser_tool import DesktopBrowserTool

from ..test_login_handoff import LOGIN_URL, OTP, PASSWORD, _Playwright
from .fakes import *


class LoginBrain(Brain):
    supports_native_tools = True

    def __init__(self) -> None:
        self.seen: list[list[LLMMessage]] = []

    def generate(self, messages, config=None, tools=None):  # type: ignore[override]
        del config, tools
        self.seen.append(list(messages))
        if messages[-1].role == "tool":
            return LLMResult(text="login-page-rechecked")
        return LLMResult(
            text="need login",
            tool_calls=[
                ToolCall(
                    id="login-1",
                    name="desktop_browser",
                    arguments={"operation": "login_handoff", "url": LOGIN_URL},
                )
            ],
        )


class LoginAgent(Agent):
    def __init__(self, home: Path) -> None:
        super().__init__(
            brain=LoginBrain(),
            desktop_home=home,
            desktop_policy_store=DesktopPolicyStore(home / ".config/slavik/policy.json"),
        )
        self.browser = DesktopBrowserTool(
            DesktopPathSecurity(home=home, policy_store_path=home / ".run" / "policy.json"),
            profile=BrowserProfile(profile_id="default", owner_id="local"),
            profile_root=home / "profiles",
        )
        self.tool_registry.register(
            "desktop_browser",
            self.browser,
            enabled=True,
            capability="exec",
            risk_classes=["network", "external_side_effect"],
            execution_targets={"desktop"},
            description="Desktop browser",
            parameters_schema={"type": "object", "properties": {}},
        )

    def close_desktop_resources(self) -> None:
        self.browser.close()


def _fake_chromium(monkeypatch: pytest.MonkeyPatch) -> _Playwright:
    fake = _Playwright()
    monkeypatch.setattr(desktop_browser_tool, "sync_playwright", lambda: fake)
    return fake


async def _start_login(client, session_id: str) -> dict:
    headers = {"X-Slavik-Session": session_id}
    await _select_local_model(client, session_id)
    await client.post("/ui/api/mode", headers=headers, json={"mode": "desktop"})
    response = await client.post(
        "/ui/api/chat/send", headers=headers, json={"content": "зайди в мой аккаунт"}
    )
    assert response.status == 200
    return await response.json()


async def _respond(client, session_id: str, decision_id: str, choice: str, headers=None):
    return await client.post(
        "/ui/api/decision/respond",
        headers=headers or {"X-Slavik-Session": session_id},
        json={"session_id": session_id, "decision_id": decision_id, "choice": choice},
    )


def test_login_handoff_waits_then_done_resumes_once(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLAVIK_ADMIN_TOKEN", "secondary-principal-token")
    playwright = _fake_chromium(monkeypatch)
    agent = LoginAgent(tmp_path)

    async def run() -> None:
        client = await _create_client(agent)
        try:
            session_id = (await (await client.get("/ui/api/status")).json())["session_id"]
            payload = await _start_login(client, session_id)
            decision = payload["decision"]
            assert decision["decision_type"] == "user_auth"
            assert decision["status"] == "pending" and decision["blocking"] is True
            assert decision["reason"] == "NEEDS_USER_AUTH"
            assert decision["summary"] == "Войди в окне Chromium на ПК → Готово"
            assert decision["context"]["run_phase"] == "WAITING_USER_INPUT"
            assert [option["id"] for option in decision["options"]] == [
                "confirm",
                "retry",
                "reject",
            ]
            assert isinstance(decision["proposed_action"]["attempt_expires_at"], str)
            assert decision["proposed_action"]["url"] == LOGIN_URL

            # Статус ожидания виден с другого устройства через session API.
            session = await (await client.get(f"/ui/api/sessions/{session_id}")).json()
            assert session["session"]["decision"]["reason"] == "NEEDS_USER_AUTH"
            assert session["session"]["decision"]["status"] == "pending"

            # approval-ответ не продолжает паузу на ручной вход.
            wrong = await _respond(client, session_id, decision["id"], "approve_once")
            assert wrong.status == 400

            # Чужой владелец не может нажать «Готово».
            foreign = await _respond(
                client,
                session_id,
                decision["id"],
                "confirm",
                headers={
                    "Cookie": (
                        f"{UI_AUTH_COOKIE}={_ui_auth_cookie_value('secondary-principal-token')}"
                    ),
                    "X-Slavik-Session": session_id,
                },
            )
            assert foreign.status == 403

            # Пользователь ввёл пароль и MFA сам, в окне Chromium.
            playwright.chromium.contexts[0].pages[0].type_credentials()
            done = await _respond(client, session_id, decision["id"], "confirm")
            assert done.status == 200
            done_payload = await done.json()
            assert done_payload["resume"]["ok"] is True
            assert "login-page-rechecked" in done_payload["resume"]["data"]["output"]
            # После «Готово» модель получила свежий observation, без секретов.
            tool_messages = [m for m in agent.brain.seen[-1] if m.role == "tool"]
            assert len(tool_messages) == 1 and "Welcome back" in tool_messages[0].content
            assert PASSWORD not in repr(agent.brain.seen) and OTP not in repr(agent.brain.seen)

            again = await _respond(client, session_id, decision["id"], "confirm")
            assert again.status == 409
            assert (await again.json())["error"]["code"] == "decision_already_resolved"
            assert agent.desktop_runtime.run_coordinator.try_acquire()
            agent.desktop_runtime.run_coordinator.release()
        finally:
            agent.desktop_runtime.cancel_pending()
            await client.close()

    asyncio.run(run())


def test_login_handoff_timeout_refuses_done_and_allows_retry(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    playwright = _fake_chromium(monkeypatch)
    agent = LoginAgent(tmp_path)
    agent.desktop_runtime.user_auth_timeout_seconds = 0.2

    async def run() -> None:
        client = await _create_client(agent)
        try:
            session_id = (await (await client.get("/ui/api/status")).json())["session_id"]
            decision = (await _start_login(client, session_id))["decision"]
            pending = agent._pending_chat_approval
            assert pending is not None
            continuation = pending.continuation
            deadline = time.monotonic() + 5
            while (
                not agent.desktop_runtime.user_auth_expired(continuation)
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert playwright.chromium.contexts[0].closed

            late = await _respond(client, session_id, decision["id"], "confirm")
            assert late.status == 409
            assert (await late.json())["error"]["code"] == "user_auth_attempt_expired"
            # Run остаётся в NEEDS_USER_AUTH: decision снова pending, run на паузе.
            session = await (await client.get(f"/ui/api/sessions/{session_id}")).json()
            assert session["session"]["decision"]["status"] == "pending"
            assert agent.desktop_runtime.has_continuation(continuation)

            retry = await _respond(client, session_id, decision["id"], "retry")
            assert retry.status == 200
            next_decision = (await retry.json())["decision"]
            assert next_decision["decision_type"] == "user_auth"
            assert next_decision["status"] == "pending"
            assert next_decision["id"] != decision["id"]
            assert [call["headless"] for call in playwright.chromium.persistent_calls] == [
                False,
                False,
            ]
            playwright.chromium.contexts[1].pages[0].type_credentials()
            agent.desktop_runtime.user_auth_timeout_seconds = 300
            done = await _respond(client, session_id, next_decision["id"], "confirm")
            assert done.status == 200
            assert "login-page-rechecked" in (await done.json())["resume"]["data"]["output"]
        finally:
            agent.desktop_runtime.cancel_pending()
            await client.close()

    asyncio.run(run())


def test_login_handoff_reject_closes_window_and_releases(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    playwright = _fake_chromium(monkeypatch)
    agent = LoginAgent(tmp_path)

    async def run() -> None:
        client = await _create_client(agent)
        try:
            session_id = (await (await client.get("/ui/api/status")).json())["session_id"]
            decision = (await _start_login(client, session_id))["decision"]
            assert not agent.desktop_runtime.run_coordinator.try_acquire()
            rejected = await _respond(client, session_id, decision["id"], "reject")
            assert rejected.status == 200
            assert playwright.chromium.contexts[0].closed
            assert agent.desktop_runtime.run_coordinator.try_acquire()
            agent.desktop_runtime.run_coordinator.release()
        finally:
            agent.desktop_runtime.cancel_pending()
            await client.close()

    asyncio.run(run())
