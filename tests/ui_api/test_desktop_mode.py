from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio
from pathlib import Path

from core.approval_policy import ApprovalPrompt, ApprovalRequest
from core.desktop_policy import (
    DesktopApprovalRule,
    DesktopApprovalScope,
    DesktopPolicyStore,
)
from server.agent_provider import AgentScope

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


class DesktopApprovalAgent(DummyAgent):
    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target
        self.desktop_rules: list[DesktopApprovalRule] = []
        self.desktop_rule_snapshots: list[list[DesktopApprovalRule]] = []
        self.desktop_clear_count = 0
        self.consumed_rule_ids: list[str] = []
        self.last_approval_request: ApprovalRequest | None = None
        self.last_approval_source_endpoint: str | None = None
        self.last_approval_resume_payload: dict[str, JSONValue] | None = None
        self.last_chat_interaction_id: str | None = None
        self.completed = 0
        self.tracer = CapturingTracer()

    def set_desktop_policy_context(
        self,
        rules: list[DesktopApprovalRule],
        principal_id: str = "legacy",
    ) -> None:
        del principal_id
        self.desktop_rules = list(rules)
        self.desktop_rule_snapshots.append(list(rules))

    def clear_desktop_policy_context(self) -> None:
        self.desktop_rules = []
        self.desktop_clear_count += 1

    def drain_consumed_desktop_rule_ids(self) -> list[str]:
        consumed = list(self.consumed_rule_ids)
        self.consumed_rule_ids.clear()
        return consumed

    def respond(self, messages) -> str:  # noqa: ANN001
        del messages
        matching = [
            rule
            for rule in self.desktop_rules
            if rule.effect == "allow" and rule.scope.target_pattern == self.target
        ]
        if matching:
            self.last_approval_request = None
            self.consumed_rule_ids.extend(
                rule.rule_id for rule in matching if rule.source == "once"
            )
            self.completed += 1
            return "desktop-sensitive-action-completed"
        scope = DesktopApprovalScope(
            tool="desktop_file_delete",
            action="delete",
            target_pattern=self.target,
            risk_class="destructive",
        )
        self.last_approval_request = ApprovalRequest(
            category="FS_DELETE_OVERWRITE",
            required_categories=["FS_DELETE_OVERWRITE"],
            prompt=ApprovalPrompt(
                what="Delete one exact file",
                why="Requested by the user",
                risk="Recoverable destructive action",
                changes=[self.target],
            ),
            tool="desktop_file_delete",
            details={"path": self.target},
            session_id=self._session_id,
            scope=scope,
            reason="destructive_action",
        )
        return "approval required"


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


def test_desktop_chat_approval_resumes_same_pipeline(tmp_path: Path) -> None:
    async def run() -> None:
        target = str(tmp_path / "Downloads" / "sensitive.iso")
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
            decision = (await first.json()).get("decision")
            assert isinstance(decision, dict)
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
            assert any(
                event == "desktop_approval_decision"
                and message == "approve_once"
                and isinstance(meta, dict)
                and meta.get("session_id") == session_id
                for event, message, meta in agent.tracer.events
            )

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


def test_desktop_session_approval_reuses_then_expires_on_mode_exit(tmp_path: Path) -> None:
    async def run() -> None:
        target = str(tmp_path / "Downloads" / "session.iso")
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
            decision = (await first.json()).get("decision")
            assert isinstance(decision, dict) and isinstance(decision.get("id"), str)
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
