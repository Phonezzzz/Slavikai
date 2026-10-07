from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio
import threading
from pathlib import Path

import pytest

from core.agent import Agent
from core.desktop_policy import (
    DesktopApprovalRule,
    DesktopApprovalScope,
    DesktopPolicyStore,
)
from llm.brain_base import Brain
from llm.types import LLMResult, ToolCall
from server.agent_provider import AgentScope
from shared.models import LLMMessage
from tools.desktop_tools import DesktopFileDeleteTool

from .fakes import *


class CapturingTracer:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, JSONValue] | None]] = []

    def log(
        self,
        event_type: str,
        message: str,
        meta: dict[str, JSONValue] | None = None,
    ) -> None:
        self.events.append((event_type, message, meta))


class DesktopApprovalBrain(Brain):
    supports_native_tools = True

    def __init__(self, target: str) -> None:
        self.target = target
        self.seen = []
        self.owner_threads = []

    def generate(self, messages, config=None, tools=None):
        self.owner_threads.append(threading.get_ident())
        self.seen.append(list(messages))
        if messages[-1].role == "tool":
            if messages[-1].tool_call_id == "delete-original":
                return LLMResult(
                    text="verify",
                    tool_calls=[
                        ToolCall(
                            id="verify-original",
                            name="desktop_verify",
                            arguments={"path": self.target, "check": "path_missing"},
                        )
                    ],
                )
            return LLMResult(text="desktop-sensitive-action-completed")
        return LLMResult(
            text="delete",
            tool_calls=[
                ToolCall(
                    id="delete-original",
                    name="desktop_file_delete",
                    arguments={"path": self.target},
                )
            ],
        )


class DesktopApprovalAgent(Agent):
    def __init__(self, target: str) -> None:
        self.target = target
        home = Path(target).parent.parent
        super().__init__(
            brain=DesktopApprovalBrain(target),
            desktop_home=home,
            desktop_policy_store=DesktopPolicyStore(home / ".config/slavik/policy.json"),
        )
        self.completed = 0
        self.desktop_rule_snapshots = []
        self.desktop_clear_count = 0
        self.tracer = CapturingTracer()
        tool = DesktopFileDeleteTool(self.desktop_security)

        def execute(request):
            result = tool.handle(request)
            if result.ok:
                self.completed += 1
            return result

        self.tool_registry.register(
            "desktop_file_delete",
            execute,
            enabled=True,
            capability="write",
            risk_classes=["write", "destructive"],
            execution_targets={"desktop"},
            description="Delete one exact file",
            parameters_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

    def reconfigure_models(self, main_config, main_api_key=None, *, persist=True):
        # Only the external provider is stubbed; the execution mechanism is production.
        pass

    def set_desktop_policy_context(self, rules, principal_id="legacy"):
        self.desktop_rule_snapshots.append(list(rules))
        super().set_desktop_policy_context(rules, principal_id)

    def clear_desktop_policy_context(self):
        self.desktop_clear_count += 1
        super().clear_desktop_policy_context()


def test_desktop_mode_transition_and_session_approval_lifecycle(tmp_path: Path) -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status = await client.get("/ui/api/status")
            status_payload = await status.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            enter = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "desktop"},
            )
            assert enter.status == 200
            enter_payload = await enter.json()
            assert enter_payload.get("mode") == "desktop"

            session_store = client.server.app["session_store"]
            principal_id = await client.server.app["ui_hub"].get_session_principal_id(session_id)
            assert isinstance(principal_id, str)
            scope = AgentScope(principal_id=principal_id, session_id=session_id)
            rule = DesktopApprovalRule.create(
                effect="allow",
                source="session",
                scope=DesktopApprovalScope(
                    tool="desktop_file_delete",
                    action="delete",
                    target_pattern=str(tmp_path / "one.txt"),
                    risk_class="destructive",
                ),
            )
            await session_store.add_desktop_rule(scope, rule)
            await session_store.approve(scope, {"NETWORK_RISK"})
            assert await session_store.get_desktop_rules(scope) == [rule]

            leave = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "ask"},
            )
            assert leave.status == 200
            assert await session_store.get_desktop_rules(scope) == []
            assert await session_store.get_categories(scope) == {"NETWORK_RISK"}
        finally:
            await client.close()

    asyncio.run(run())


def test_desktop_persistent_approval_crud_api(tmp_path: Path) -> None:
    async def run() -> None:
        client = await _create_client(
            DummyAgent(),
            desktop_policy_store=DesktopPolicyStore(tmp_path / "desktop-approvals.json"),
        )
        try:
            create = await client.post(
                "/ui/api/desktop/approvals",
                json={
                    "effect": "allow",
                    "description": "one exact file",
                    "scope": {
                        "tool": "desktop_file_delete",
                        "action": "delete",
                        "target_pattern": str(tmp_path / "Downloads" / "one.iso"),
                        "risk_class": "destructive",
                        "execution_target": "desktop",
                    },
                },
            )
            assert create.status == 201
            created = await create.json()
            rule = created.get("rule")
            assert isinstance(rule, dict)
            rule_id = rule.get("rule_id")
            assert isinstance(rule_id, str)

            listing = await client.get("/ui/api/desktop/approvals")
            listing_payload = await listing.json()
            rules = listing_payload.get("rules")
            assert isinstance(rules, list) and len(rules) == 1

            update = await client.patch(
                f"/ui/api/desktop/approvals/{rule_id}",
                json={"effect": "deny"},
            )
            assert update.status == 200
            updated = await update.json()
            assert updated["rule"]["effect"] == "deny"

            remove = await client.delete(f"/ui/api/desktop/approvals/{rule_id}")
            assert remove.status == 200
            empty = await client.get("/ui/api/desktop/approvals")
            assert (await empty.json()).get("rules") == []
        finally:
            await client.close()

    asyncio.run(run())


def test_desktop_invalid_approval_store_has_explicit_owner_recovery(tmp_path: Path) -> None:
    async def run() -> None:
        store_path = tmp_path / "desktop-approvals.json"
        store_path.write_text("{not-json", encoding="utf-8")
        client = await _create_client(
            DummyAgent(),
            desktop_policy_store=DesktopPolicyStore(store_path),
        )
        try:
            listing = await client.get("/ui/api/desktop/approvals")
            assert listing.status == 409
            listing_error = (await listing.json()).get("error")
            assert isinstance(listing_error, dict)
            assert listing_error.get("code") == "desktop_approval_store_invalid"

            missing_confirmation = await client.post(
                "/ui/api/desktop/approvals/reset-invalid",
                json={"confirm": False},
            )
            assert missing_confirmation.status == 400

            reset = await client.post(
                "/ui/api/desktop/approvals/reset-invalid",
                json={"confirm": True},
            )
            assert reset.status == 200
            reset_payload = await reset.json()
            assert reset_payload.get("rules") == []
            assert reset_payload.get("discarded_load_errors")

            recovered = await client.get("/ui/api/desktop/approvals")
            assert recovered.status == 200
            assert (await recovered.json()).get("rules") == []
        finally:
            await client.close()

    asyncio.run(run())


def test_desktop_always_allow_decision_persists_exact_scope(tmp_path: Path) -> None:
    async def run() -> None:
        store = DesktopPolicyStore(tmp_path / "desktop-approvals.json")
        client = await _create_client(DummyAgent(), desktop_policy_store=store)
        try:
            status = await client.get("/ui/api/status")
            session_id = (await status.json()).get("session_id")
            assert isinstance(session_id, str)
            hub = client.server.app["ui_hub"]
            target = str(tmp_path / "Downloads" / "one.iso")
            await hub.set_session_decision(
                session_id,
                {
                    "id": "desktop-approval-1",
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
                            "target_pattern": target,
                            "risk_class": "destructive",
                            "execution_target": "desktop",
                        },
                    },
                    "options": [],
                    "default_option_id": None,
                    "context": {
                        "session_id": session_id,
                        "source_endpoint": "workspace.tool",
                        "resume_payload": {
                            "tool_name": "workspace_write",
                            "args": {"path": "approval-probe.txt", "content": "ok"},
                        },
                    },
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "resolved_at": None,
                },
            )

            response = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "desktop-approval-1",
                    "choice": "always_allow",
                },
            )

            assert response.status == 200
            rules = store.list_rules()
            assert len(rules) == 1
            assert rules[0].scope.target_pattern == target
            assert rules[0].scope.tool == "desktop_file_delete"
        finally:
            await client.close()

    asyncio.run(run())


def test_desktop_chat_approval_resumes_same_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    async def run() -> None:
        target = str(tmp_path / "Downloads" / "sensitive.iso")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("sensitive fixture")
        agent = DesktopApprovalAgent(target)
        client = await _create_client(agent)
        try:
            status = await client.get("/ui/api/status")
            session_id = (await status.json()).get("session_id")
            assert isinstance(session_id, str)
            principal_id = await client.server.app["ui_hub"].get_session_principal_id(session_id)
            assert isinstance(principal_id, str)
            scope = AgentScope(principal_id=principal_id, session_id=session_id)
            await _select_local_model(client, session_id)
            enter = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "desktop"},
            )
            assert enter.status == 200

            first = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "delete the sensitive file"},
            )
            assert first.status == 200
            first_payload = await first.json()
            decision = first_payload.get("decision")
            assert isinstance(decision, dict), [
                event for event in agent.tracer.events if event[0] == "policy_denied"
            ]
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)
            current_decision = await client.server.app["ui_hub"].get_session_decision(session_id)
            assert isinstance(current_decision, dict)
            assert current_decision.get("id") == decision_id, (decision, current_decision)

            approve = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "approve_once",
                },
            )

            payload = await approve.json()
            assert approve.status == 200, (
                payload,
                agent.desktop_rule_snapshots,
                await client.server.app["session_store"].get_desktop_rules(scope),
            )
            resume = payload.get("resume")
            assert isinstance(resume, dict) and resume.get("ok") is True
            assert agent.completed == 1, (
                payload,
                agent.desktop_rule_snapshots,
                agent.desktop_clear_count,
                await client.server.app["ui_hub"].get_session_workflow(session_id),
            )
            assert await client.server.app["session_store"].get_desktop_rules(scope) == []
            assert not Path(target).exists()
            assert agent.brain.seen[1][-1].tool_call_id == "delete-original"
            assert agent.brain.seen[2][-1].tool_call_id == "verify-original"
            assert len(agent.brain.seen) == 3
            assert len(set(agent.brain.owner_threads)) == 1
            chat_messages = await client.server.app["ui_hub"].get_messages(session_id, lane="chat")
            assert payload["messages"] == chat_messages
            assert agent.short_term[-1].content.startswith(chat_messages[-1]["content"])
            assert "desktop-sensitive-action-completed" in agent.short_term[-1].content
            interaction = agent._interaction_store.get_interaction(agent.last_chat_interaction_id)
            assert interaction is not None
            assert interaction.response_text == agent.short_term[-1].content
            assert payload["output"] == await client.server.app["ui_hub"].get_session_output(
                session_id
            )
            assert len([m for m in chat_messages if m.get("role") == "user"]) == 1
            assert any(
                event == "desktop_approval_decision"
                and message == "approve_once"
                and isinstance(meta, dict)
                and meta.get("session_id") == session_id
                for event, message, meta in agent.tracer.events
            )

            Path(target).write_text("second fixture")
            again = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "delete the sensitive file again"},
            )
            assert again.status == 200
            again_decision = (await again.json()).get("decision")
            assert isinstance(again_decision, dict)
            assert again_decision.get("status") == "pending"
            assert agent.completed == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_desktop_session_approval_reuses_then_expires_on_mode_exit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    async def run() -> None:
        target = str(tmp_path / "Downloads" / "session.iso")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("sensitive fixture")
        agent = DesktopApprovalAgent(target)
        client = await _create_client(agent)
        try:
            status = await client.get("/ui/api/status")
            session_id = (await status.json()).get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)
            await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "desktop"},
            )
            first = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "delete it"},
            )
            first_payload = await first.json()
            decision = first_payload.get("decision")
            assert isinstance(decision, dict) and isinstance(decision.get("id"), str), first_payload
            approved = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision["id"],
                    "choice": "approve_session",
                },
            )
            assert approved.status == 200
            assert agent.completed == 1

            Path(target).write_text("second fixture")
            reused = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "delete it again"},
            )
            assert reused.status == 200
            assert (await reused.json()).get("decision") is None
            assert agent.completed == 2

            await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "ask"},
            )
            await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "desktop"},
            )
            Path(target).write_text("third fixture")
            expired = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "delete it after re-enter"},
            )
            expired_decision = (await expired.json()).get("decision")
            assert isinstance(expired_decision, dict)
            assert expired_decision.get("status") == "pending"
            assert agent.completed == 2
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("termination", ["reject", "cancel", "mode_exit", "timeout", "new_turn"])
def test_desktop_paused_run_owns_lease_until_cleanup(tmp_path, monkeypatch, termination):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "Downloads" / "pending.iso"
    target.parent.mkdir()
    target.write_text("pending")
    agent = DesktopApprovalAgent(str(target))
    generate = agent.brain.generate

    def respond_new_turn(messages, config=None, tools=None):
        if next((m.content for m in reversed(messages) if m.role == "user"), None) == (
            "найди в интернете погоду"
        ):
            agent.brain.owner_threads.append(threading.get_ident())
            return LLMResult(text="new turn response")
        return generate(messages, config, tools)

    monkeypatch.setattr(agent.brain, "generate", respond_new_turn)
    closed = threading.Event()
    closes = []

    def close():
        closes.append(threading.get_ident())
        closed.set()

    monkeypatch.setattr(agent, "close_desktop_resources", close)
    if termination == "timeout":
        agent.desktop_runtime.approval_timeout_seconds = 0.3

    async def run():
        client = await _create_client(agent)
        try:
            session_id = (await (await client.get("/ui/api/status")).json())["session_id"]
            headers = {"X-Slavik-Session": session_id}
            await _select_local_model(client, session_id)
            await client.post("/ui/api/mode", headers=headers, json={"mode": "desktop"})
            payload = await (
                await client.post(
                    "/ui/api/chat/send", headers=headers, json={"content": "delete the file"}
                )
            ).json()
            decision = payload["decision"]
            assert decision["context"]["source_endpoint"] == "chat.tool_continue"
            identity = agent.last_approval_resume_payload["continuation_id"]
            assert agent.completed == 0 and target.exists()
            assert not closed.is_set()
            assert not agent.desktop_runtime.run_coordinator.try_acquire()
            if termination == "reject":
                response = await client.post(
                    "/ui/api/decision/respond",
                    headers=headers,
                    json={
                        "session_id": session_id,
                        "decision_id": decision["id"],
                        "choice": "reject",
                    },
                )
                assert response.status == 200
            elif termination == "cancel":
                response = await client.post("/ui/api/chat/cancel", headers=headers)
                assert response.status == 200
                assert (await response.json())["cancelled"] is True
            elif termination == "new_turn":
                response = await client.post(
                    "/ui/api/chat/send",
                    headers=headers,
                    json={"content": "найди в интернете погоду"},
                )
                assert response.status == 200
            elif termination == "mode_exit":
                response = await client.post("/ui/api/mode", headers=headers, json={"mode": "ask"})
                assert response.status == 200
            assert await asyncio.to_thread(closed.wait, 2)
            assert closes == (
                agent.brain.owner_threads[:2]
                if termination == "new_turn"
                else [agent.brain.owner_threads[0]]
            )
            assert agent.desktop_runtime.run_coordinator.try_acquire()
            agent.desktop_runtime.run_coordinator.release()
            from core.agent_tools import ChatApprovalUnavailable

            with pytest.raises(ChatApprovalUnavailable):
                agent.resume_chat_approval(identity)
            agent.desktop_runtime.cancel_pending()
            assert closes == (
                agent.brain.owner_threads[:2]
                if termination == "new_turn"
                else [agent.brain.owner_threads[0]]
            )
            assert agent.completed == 0 and target.exists()
            if termination == "timeout":
                response = await client.post(
                    "/ui/api/decision/respond",
                    headers=headers,
                    json={
                        "session_id": session_id,
                        "decision_id": decision["id"],
                        "choice": "approve_once",
                    },
                )
                assert response.status == 409
                principal = await client.server.app["ui_hub"].get_session_principal_id(session_id)
                assert (
                    await client.server.app["session_store"].get_desktop_rules(
                        AgentScope(principal, session_id)
                    )
                    == []
                )
        finally:
            agent.desktop_runtime.cancel_pending()
            await client.close()

    asyncio.run(run())


def test_desktop_resume_preserves_result_on_provider_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "Downloads" / "failure.iso"
    target.parent.mkdir()
    target.write_text("fixture")
    agent = DesktopApprovalAgent(str(target))
    agent.set_session_context("session", set())
    agent.set_runtime_state(
        mode="desktop", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    try:
        agent.respond([LLMMessage(role="user", content="delete file")])
        assert agent.last_approval_request is not None
        identity = agent.last_approval_resume_payload["continuation_id"]
        rule = DesktopApprovalRule.create(
            effect="allow", source="once", scope=agent.last_approval_request.scope
        )
        agent.set_desktop_policy_context([rule], "local")

        def fail(*args, **kwargs):
            raise RuntimeError("provider unavailable after dispatch")

        monkeypatch.setattr(agent.brain, "generate", fail)
        response = agent.resume_chat_approval(identity)
        assert agent.completed == 1 and not target.exists()
        assert response.runtime_result.loop_result.error == "provider_generation_failed"
        assert len(response.runtime_result.loop_result.tool_calls) == 1
        assert response.runtime_result.loop_result.tool_calls[0].result.ok is True
        assert response.runtime_result.loop_result.tool_calls[0].call.id == "delete-original"
        from core.agent_tools import ChatApprovalUnavailable

        with pytest.raises(ChatApprovalUnavailable):
            agent.resume_chat_approval(identity)
        assert agent.completed == 1
        assert agent.desktop_runtime.run_coordinator.try_acquire()
        agent.desktop_runtime.run_coordinator.release()
    finally:
        agent.close()


def test_desktop_cancel_at_approval_does_not_retain_resources(tmp_path, monkeypatch):
    from core.approval_policy import ApprovalRequired

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "Downloads" / "cancel.iso"
    target.parent.mkdir()
    target.write_text("fixture")
    agent = DesktopApprovalAgent(str(target))
    agent.set_session_context("session", set())
    agent.set_runtime_state(
        mode="desktop", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    token = asyncio.Event()
    original_builder = agent._build_tool_gateway

    def builder(*args, **kwargs):
        gateway = original_builder(*args, **kwargs)
        call = gateway.call

        def dispatch(request):
            try:
                return call(request)
            except ApprovalRequired:
                token.set()
                raise

        gateway.call = dispatch
        return gateway

    monkeypatch.setattr(agent, "_build_tool_gateway", builder)
    try:
        outcome = agent.desktop_runtime.run("delete file", cancellation_token=token)
        assert outcome.loop_result.cancelled
        assert agent.completed == 0 and target.exists()
        assert agent.desktop_runtime.run_coordinator.try_acquire()
        agent.desktop_runtime.run_coordinator.release()
    finally:
        agent.close()


@pytest.mark.parametrize("finish", ["approve", "reject", "new_turn"])
def test_desktop_compound_request_retains_exact_once_scopes(tmp_path, monkeypatch, finish):
    from shared.models import ToolResult

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "Downloads" / "compound.iso"
    target.parent.mkdir()
    target.write_text("original")
    agent = DesktopApprovalAgent(str(target))
    calls = []
    args = {
        "operation": "download",
        "url": "https://example.test/download",
        "destination": str(target),
        "overwrite": True,
    }

    def generate(messages, config=None, tools=None):
        if next((m.content for m in reversed(messages) if m.role == "user"), None) == (
            "найди в интернете погоду"
        ):
            return LLMResult(text="new turn response")
        if not any(message.role == "tool" for message in messages):
            return LLMResult(
                text="",
                tool_calls=[ToolCall(id="compound", name="desktop_browser", arguments=args)],
            )
        return LLMResult(text="browser offline")

    monkeypatch.setattr(agent.brain, "generate", generate)

    def browser(request):
        calls.append(request)
        return ToolResult.failure("fixture browser offline")

    agent.tool_registry.register(
        "desktop_browser",
        browser,
        enabled=True,
        capability="exec",
        execution_targets={"desktop"},
        description="Browser download",
        parameters_schema={"type": "object"},
    )

    async def run():
        client = await _create_client(agent)
        try:
            session_id = (await (await client.get("/ui/api/status")).json())["session_id"]
            headers = {"X-Slavik-Session": session_id}
            await _select_local_model(client, session_id)
            await client.post("/ui/api/mode", headers=headers, json={"mode": "desktop"})
            payload = await (
                await client.post(
                    "/ui/api/chat/send", headers=headers, json={"content": "download file"}
                )
            ).json()
            first = payload["decision"]

            async def respond(decision, choice):
                return await client.post(
                    "/ui/api/decision/respond",
                    headers=headers,
                    json={
                        "session_id": session_id,
                        "decision_id": decision["id"],
                        "choice": choice,
                    },
                )

            response = await respond(first, "approve_once")
            assert response.status == 200
            second = (await response.json())["decision"]
            assert second["id"] != first["id"] and second["status"] == "pending"
            assert calls == []
            assert len(agent._pending_chat_approval.desktop_once_rules) == 1
            if finish == "approve":
                completed = await respond(second, "approve_once")
                assert completed.status == 200
                assert (await completed.json())["status"] == "resolved"
                assert len(calls) == 1 and calls[0].args == args
            elif finish == "reject":
                assert (await respond(second, "reject")).status == 200
                assert calls == []
            else:
                superseded = await client.post(
                    "/ui/api/chat/send",
                    headers=headers,
                    json={"content": "найди в интернете погоду"},
                )
                assert superseded.status == 200
                assert calls == []
            assert agent._pending_chat_approval is None
            assert target.read_text() == "original"
            assert agent.desktop_runtime.run_coordinator.try_acquire()
            agent.desktop_runtime.run_coordinator.release()
        finally:
            await client.close()

    asyncio.run(run())
