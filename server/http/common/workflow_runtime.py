from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Protocol

from aiohttp import web

from shared.models import JSONValue


class WorkflowHubProtocol(Protocol):
    async def get_session_policy(self, session_id: str) -> dict[str, JSONValue]: ...

    async def get_session_tools_state(self, session_id: str) -> dict[str, bool] | None: ...

    async def get_session_workflow(self, session_id: str) -> dict[str, JSONValue]: ...

    async def set_session_workflow(
        self,
        session_id: str,
        *,
        mode: str | None = None,
        active_plan: dict[str, JSONValue] | None | object = None,
        active_task: dict[str, JSONValue] | None | object = None,
    ) -> dict[str, JSONValue]: ...


def normalize_tools_state_payload(
    raw: object,
    *,
    default_tools_state_keys: set[str],
) -> dict[str, bool]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, bool] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if key not in default_tools_state_keys:
            continue
        if isinstance(value, bool):
            normalized[key] = value
    return normalized


def build_effective_tools_state(
    *,
    session_override: dict[str, bool] | None,
    load_tools_state_fn: Callable[[], dict[str, bool]],
    default_tools_state_keys: set[str],
) -> dict[str, bool]:
    defaults = load_tools_state_fn()
    effective = dict(defaults)
    if isinstance(session_override, dict):
        for key, value in session_override.items():
            if key in default_tools_state_keys and isinstance(value, bool):
                effective[key] = value
    return effective


async def load_effective_session_security(
    *,
    hub: WorkflowHubProtocol,
    session_id: str,
    normalize_policy_profile_fn: Callable[[object], str],
    build_effective_tools_state_fn: Callable[[dict[str, bool] | None], dict[str, bool]],
) -> tuple[dict[str, bool], dict[str, JSONValue]]:
    policy = await hub.get_session_policy(session_id)
    profile_raw = policy.get("profile")
    profile = normalize_policy_profile_fn(profile_raw)
    session_tools_override = await hub.get_session_tools_state(session_id)
    effective_tools = build_effective_tools_state_fn(session_tools_override)
    yolo_armed = policy.get("yolo_armed") is True
    yolo_armed_at_raw = policy.get("yolo_armed_at")
    yolo_armed_at = (
        yolo_armed_at_raw.strip()
        if isinstance(yolo_armed_at_raw, str) and yolo_armed_at_raw.strip()
        else None
    )
    if not yolo_armed:
        yolo_armed_at = None
    effective_policy: dict[str, JSONValue] = {
        "profile": profile,
        "yolo_armed": yolo_armed,
        "yolo_armed_at": yolo_armed_at,
    }
    return effective_tools, effective_policy


async def apply_agent_runtime_state(
    *,
    agent: object,
    hub: WorkflowHubProtocol,
    session_id: str,
    load_effective_session_security_fn: Callable[
        [WorkflowHubProtocol, str],
        Coroutine[object, object, tuple[dict[str, bool], dict[str, JSONValue]]],
    ],
    normalize_mode_value_fn: Callable[[object], str],
    normalize_plan_payload_fn: Callable[[object], dict[str, JSONValue] | None],
    normalize_task_payload_fn: Callable[[object], dict[str, JSONValue] | None],
) -> tuple[str, dict[str, JSONValue] | None, dict[str, JSONValue] | None]:
    effective_tools, _ = await load_effective_session_security_fn(hub, session_id)
    runtime_tools_setter = getattr(agent, "apply_runtime_tools_enabled", None)
    if callable(runtime_tools_setter):
        runtime_tools_setter(effective_tools)

    workflow = await hub.get_session_workflow(session_id)
    mode = normalize_mode_value_fn(workflow.get("mode"))
    active_plan = normalize_plan_payload_fn(workflow.get("active_plan"))
    active_task = normalize_task_payload_fn(workflow.get("active_task"))
    runtime_setter = getattr(agent, "set_runtime_state", None)
    if callable(runtime_setter):
        runtime_setter(
            mode=mode,
            active_plan=active_plan,
            active_task=active_task,
            enforce_plan_guard=(
                mode == "act" and active_plan is not None and active_task is not None
            ),
        )
    return mode, active_plan, active_task


def build_default_plan_steps() -> list[dict[str, JSONValue]]:
    return [
        {
            "step_id": "step-1-audit",
            "title": "Аудит контекста",
            "description": "Проверить релевантные файлы и текущее состояние проекта.",
            "allowed_tool_kinds": ["workspace_list", "workspace_read", "project"],
            "acceptance_checks": ["Понять текущее поведение и ограничения"],
            "status": "todo",
            "evidence": None,
        },
        {
            "step_id": "step-2-implement",
            "title": "Изменения",
            "description": "Внести изменения по задаче и синхронизировать артефакты.",
            "allowed_tool_kinds": [
                "workspace_read",
                "workspace_write",
                "workspace_patch",
                "project",
                "shell",
                "fs",
            ],
            "acceptance_checks": ["Изменения применены в нужных файлах"],
            "status": "todo",
            "evidence": None,
        },
        {
            "step_id": "step-3-verify",
            "title": "Проверка",
            "description": "Запустить проверки и убедиться, что задача закрыта.",
            "allowed_tool_kinds": ["workspace_read", "workspace_run", "shell", "project"],
            "acceptance_checks": ["make check или эквивалентные проверки зелёные"],
            "status": "todo",
            "evidence": None,
        },
    ]


def build_plan_draft(
    *,
    goal: str,
    audit_log: list[dict[str, JSONValue]],
    utc_now_iso_fn: Callable[[], str],
    plan_hash_payload_fn: Callable[[dict[str, JSONValue]], str],
    build_default_plan_steps_fn: Callable[
        [],
        list[dict[str, JSONValue]],
    ] = build_default_plan_steps,
) -> dict[str, JSONValue]:
    now = utc_now_iso_fn()
    plan: dict[str, JSONValue] = {
        "plan_id": f"plan-{uuid.uuid4().hex}",
        "plan_hash": "",
        "plan_revision": 1,
        "status": "draft",
        "goal": goal,
        "scope_in": [],
        "scope_out": [],
        "assumptions": [],
        "inputs_needed": [],
        "audit_log": audit_log,
        "steps": build_default_plan_steps_fn(),
        "exit_criteria": [
            "Целевое изменение внедрено",
            "Регрессии не обнаружены",
            "Результат проверен",
        ],
        "created_at": now,
        "updated_at": now,
        "approved_at": None,
        "approved_by": None,
    }
    plan["plan_hash"] = plan_hash_payload_fn(plan)
    return plan


def plan_with_status(
    plan: dict[str, JSONValue],
    *,
    status: str,
    utc_now_iso_fn: Callable[[], str],
    increment_plan_revision_fn: Callable[[dict[str, JSONValue]], int],
    plan_hash_payload_fn: Callable[[dict[str, JSONValue]], str],
) -> dict[str, JSONValue]:
    updated = dict(plan)
    updated["status"] = status
    updated["updated_at"] = utc_now_iso_fn()
    if status == "approved":
        updated["approved_at"] = utc_now_iso_fn()
        updated["approved_by"] = "user"
    elif status == "draft":
        updated["approved_at"] = None
        updated["approved_by"] = None
    increment_plan_revision_fn(updated)
    updated["plan_hash"] = plan_hash_payload_fn(updated)
    return updated


def task_with_status(
    task: dict[str, JSONValue],
    *,
    status: str,
    current_step_id: str | None = None,
    utc_now_iso_fn: Callable[[], str],
) -> dict[str, JSONValue]:
    updated = dict(task)
    updated["status"] = status
    updated["current_step_id"] = current_step_id
    updated["updated_at"] = utc_now_iso_fn()
    return updated


async def run_plan_runner(
    *,
    app: web.Application,
    session_id: str,
    plan_id: str,
    task_id: str,
    normalize_plan_payload_fn: Callable[[object], dict[str, JSONValue] | None],
    normalize_task_payload_fn: Callable[[object], dict[str, JSONValue] | None],
    normalize_mode_value_fn: Callable[[object], str],
    find_next_todo_step_fn: Callable[[dict[str, JSONValue]], dict[str, JSONValue] | None],
    plan_with_status_fn: Callable[[dict[str, JSONValue], str], dict[str, JSONValue]],
    task_with_status_fn: Callable[[dict[str, JSONValue], str, str | None], dict[str, JSONValue]],
    plan_mark_step_fn: Callable[
        [dict[str, JSONValue], str, str, dict[str, JSONValue] | None],
        dict[str, JSONValue],
    ],
    utc_now_iso_fn: Callable[[], str],
) -> None:
    hub: WorkflowHubProtocol = app["ui_hub"]
    while True:
        workflow = await hub.get_session_workflow(session_id)
        plan = normalize_plan_payload_fn(workflow.get("active_plan"))
        task = normalize_task_payload_fn(workflow.get("active_task"))
        mode = normalize_mode_value_fn(workflow.get("mode"))
        if (
            plan is None
            or task is None
            or task.get("task_id") != task_id
            or plan.get("plan_id") != plan_id
            or task.get("status") != "running"
            or mode != "act"
        ):
            return

        current_step_id_raw = task.get("current_step_id")
        current_step_id = current_step_id_raw if isinstance(current_step_id_raw, str) else None
        if current_step_id is None:
            next_step = find_next_todo_step_fn(plan)
            if next_step is None:
                completed_plan = plan_with_status_fn(plan, "completed")
                completed_task = task_with_status_fn(task, "completed", None)
                await hub.set_session_workflow(
                    session_id,
                    mode="act",
                    active_plan=completed_plan,
                    active_task=completed_task,
                )
                return
            step_id_raw = next_step.get("step_id")
            step_id = step_id_raw if isinstance(step_id_raw, str) else None
            if step_id is None:
                failed_plan = plan_with_status_fn(plan, "failed")
                failed_task = task_with_status_fn(task, "failed", None)
                await hub.set_session_workflow(
                    session_id,
                    mode="act",
                    active_plan=failed_plan,
                    active_task=failed_task,
                )
                return
            plan = plan_mark_step_fn(plan, step_id, "doing", None)
            task = task_with_status_fn(task, "running", step_id)
            await hub.set_session_workflow(
                session_id,
                mode="act",
                active_plan=plan,
                active_task=task,
            )
            await asyncio.sleep(0)
            continue

        evidence: dict[str, JSONValue] = {
            "runner": "skeleton",
            "completed_at": utc_now_iso_fn(),
        }
        plan = plan_mark_step_fn(plan, current_step_id, "done", evidence)
        task = task_with_status_fn(task, "running", None)
        await hub.set_session_workflow(
            session_id,
            mode="act",
            active_plan=plan,
            active_task=task,
        )
        await asyncio.sleep(0)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
