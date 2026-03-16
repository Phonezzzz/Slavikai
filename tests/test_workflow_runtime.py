from __future__ import annotations

import asyncio
from pathlib import Path

from server.http.common import workflow_runtime


class DummyHub:
    async def get_session_policy(self, session_id: str) -> dict[str, object]:
        assert session_id == "session-1"
        return {"profile": "sandbox", "yolo_armed": False}

    async def get_session_tools_state(self, session_id: str) -> dict[str, bool] | None:
        assert session_id == "session-1"
        return {"safe_mode": True}

    async def get_session_workflow(self, session_id: str) -> dict[str, object]:
        assert session_id == "session-1"
        return {
            "mode": "act",
            "active_plan": {"plan_id": "p1"},
            "active_task": {"task_id": "t1"},
            "auto_state": {"status": "idle"},
        }

    async def set_session_workflow(
        self,
        session_id: str,
        *,
        mode: str | None = None,
        active_plan: dict[str, object] | None | object = None,
        active_task: dict[str, object] | None | object = None,
        auto_state: dict[str, object] | None | object = None,
    ) -> dict[str, object]:
        _ = (session_id, mode, active_plan, active_task, auto_state)
        return {}


class DummyAgent:
    def __init__(self) -> None:
        self.runtime_tools_state: dict[str, bool] | None = None
        self.runtime_workspace_root: str | None = None
        self.runtime_state: dict[str, object] | None = None

    def apply_runtime_tools_enabled(self, state: dict[str, bool]) -> None:
        self.runtime_tools_state = dict(state)

    def apply_runtime_workspace_root(self, workspace_root: str | None) -> None:
        self.runtime_workspace_root = workspace_root

    def set_runtime_state(
        self,
        *,
        mode: str,
        active_plan: dict[str, object] | None,
        active_task: dict[str, object] | None,
        auto_state: dict[str, object] | None = None,
        enforce_plan_guard: bool,
    ) -> None:
        self.runtime_state = {
            "mode": mode,
            "active_plan": active_plan,
            "active_task": active_task,
            "auto_state": auto_state,
            "enforce_plan_guard": enforce_plan_guard,
        }


def test_apply_agent_runtime_state_propagates_workspace_root(tmp_path: Path) -> None:
    async def run() -> None:
        hub = DummyHub()
        agent = DummyAgent()

        async def _security_loader(
            _hub: workflow_runtime.WorkflowHubProtocol,
            _session_id: str,
        ) -> tuple[dict[str, bool], dict[str, object]]:
            return {"safe_mode": True}, {"profile": "sandbox"}

        async def _workspace_root_loader(
            _hub: workflow_runtime.WorkflowHubProtocol,
            _session_id: str,
        ) -> Path:
            return tmp_path

        result = await workflow_runtime.apply_agent_runtime_state(
            agent=agent,
            hub=hub,
            session_id="session-1",
            load_effective_session_security_fn=_security_loader,
            resolve_workspace_root_fn=_workspace_root_loader,
            normalize_mode_value_fn=lambda value: str(value),
            normalize_plan_payload_fn=lambda value: value if isinstance(value, dict) else None,
            normalize_task_payload_fn=lambda value: value if isinstance(value, dict) else None,
            normalize_auto_state_fn=lambda value: value if isinstance(value, dict) else None,
        )

        assert result[0] == "act"
        assert agent.runtime_tools_state == {"safe_mode": True}
        assert agent.runtime_workspace_root == str(tmp_path)
        assert agent.runtime_state == {
            "mode": "act",
            "active_plan": {"plan_id": "p1"},
            "active_task": {"task_id": "t1"},
            "auto_state": {"status": "idle"},
            "enforce_plan_guard": True,
        }

    asyncio.run(run())


def test_compile_plan_to_task_packet(tmp_path: Path) -> None:
    plan = workflow_runtime.build_plan_draft(
        goal="Исправить файл",
        audit_log=[],
        utc_now_iso_fn=lambda: "2026-01-01T00:00:00+00:00",
        plan_hash_payload_fn=lambda payload: "plan-hash",
    )

    packet = workflow_runtime.compile_plan_to_task_packet(
        plan=plan,
        session_id="session-1",
        trace_id="trace-1",
        workspace_root=str(tmp_path),
        approved_categories=["EXEC_ARBITRARY"],
    )

    assert packet.scope["workspace_root"] == str(tmp_path)
    assert packet.approvals["approved_categories"] == ["EXEC_ARBITRARY"]
    assert packet.packet_hash
    assert len(packet.steps) == 3
