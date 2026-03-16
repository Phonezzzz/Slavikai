from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from aiohttp import web

from core.agent_mwv import TaskPacketApprovalPending
from core.mwv.manager import MWVRunResult
from core.mwv.models import (
    MWVMessage,
    TaskPacket,
    TaskStepContract,
    VerificationStatus,
    WorkStatus,
    with_task_packet_hash,
)
from core.mwv.verifier_summary import extract_verifier_excerpt
from shared.models import JSONValue

WorkflowRuntimeState = tuple[
    str,
    dict[str, JSONValue] | None,
    dict[str, JSONValue] | None,
    dict[str, JSONValue] | None,
]


class WorkflowHubProtocol(Protocol):
    async def get_session_policy(self, session_id: str) -> dict[str, JSONValue]: ...

    async def get_session_tools_state(self, session_id: str) -> dict[str, bool] | None: ...

    async def get_session_workflow(self, session_id: str) -> dict[str, JSONValue]: ...
    async def get_session_decision(self, session_id: str) -> dict[str, JSONValue] | None: ...
    async def set_session_decision(
        self,
        session_id: str,
        decision: dict[str, JSONValue] | None,
    ) -> dict[str, JSONValue] | None: ...
    async def get_workspace_root(self, session_id: str) -> str | None: ...

    async def set_session_workflow(
        self,
        session_id: str,
        *,
        mode: str | None = None,
        active_plan: dict[str, JSONValue] | None | object = None,
        active_task: dict[str, JSONValue] | None | object = None,
        auto_state: dict[str, JSONValue] | None | object = None,
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
    safe_mode_effective = bool(effective_tools.get("safe_mode", False))
    effective_policy: dict[str, JSONValue] = {
        "profile": profile,
        "yolo_armed": yolo_armed,
        "yolo_armed_at": yolo_armed_at,
        "safe_mode_effective": safe_mode_effective,
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
    resolve_workspace_root_fn: Callable[
        [WorkflowHubProtocol, str],
        Coroutine[object, object, Path],
    ],
    normalize_mode_value_fn: Callable[[object], str],
    normalize_plan_payload_fn: Callable[[object], dict[str, JSONValue] | None],
    normalize_task_payload_fn: Callable[[object], dict[str, JSONValue] | None],
    normalize_auto_state_fn: Callable[[object], dict[str, JSONValue] | None],
) -> WorkflowRuntimeState:
    effective_tools, _ = await load_effective_session_security_fn(hub, session_id)
    runtime_tools_setter = getattr(agent, "apply_runtime_tools_enabled", None)
    if callable(runtime_tools_setter):
        runtime_tools_setter(effective_tools)
    workspace_root = await resolve_workspace_root_fn(hub, session_id)
    runtime_workspace_setter = getattr(agent, "apply_runtime_workspace_root", None)
    if callable(runtime_workspace_setter):
        runtime_workspace_setter(str(workspace_root))

    workflow = await hub.get_session_workflow(session_id)
    mode = normalize_mode_value_fn(workflow.get("mode"))
    active_plan = normalize_plan_payload_fn(workflow.get("active_plan"))
    active_task = normalize_task_payload_fn(workflow.get("active_task"))
    auto_state = normalize_auto_state_fn(workflow.get("auto_state"))
    runtime_setter = getattr(agent, "set_runtime_state", None)
    if callable(runtime_setter):
        runtime_setter(
            mode=mode,
            active_plan=active_plan,
            active_task=active_task,
            auto_state=auto_state,
            enforce_plan_guard=(
                mode == "act" and active_plan is not None and active_task is not None
            ),
        )
    return mode, active_plan, active_task, auto_state


def build_default_plan_steps() -> list[dict[str, JSONValue]]:
    return [
        {
            "step_id": "step-1-audit",
            "title": "Аудит контекста",
            "description": "Проверить релевантные файлы и текущее состояние проекта.",
            "allowed_tool_kinds": ["workspace_list", "workspace_read", "project"],
            "inputs": {},
            "expected_outputs": ["Понять текущее поведение и ограничения"],
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
            "inputs": {},
            "expected_outputs": ["Изменения применены в нужных файлах"],
            "acceptance_checks": ["Изменения применены в нужных файлах"],
            "status": "todo",
            "evidence": None,
        },
        {
            "step_id": "step-3-verify",
            "title": "Проверка",
            "description": "Запустить проверки и убедиться, что задача закрыта.",
            "allowed_tool_kinds": ["workspace_read", "workspace_run", "shell", "project"],
            "inputs": {},
            "expected_outputs": ["make check или эквивалентные проверки зелёные"],
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
        "budgets": {"max_attempts": 1},
        "approvals": {"approved_categories": []},
        "verifier": {},
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


def compile_plan_to_task_packet(
    *,
    plan: dict[str, JSONValue],
    session_id: str,
    trace_id: str,
    workspace_root: str,
    approved_categories: list[str] | None = None,
) -> TaskPacket:
    steps_raw = plan.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError("plan.steps должен содержать хотя бы один шаг.")

    task_steps: list[TaskStepContract] = []
    for index, item in enumerate(steps_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"plan.steps[{index - 1}] должен быть объектом.")
        step_id_raw = item.get("step_id")
        title_raw = item.get("title")
        description_raw = item.get("description")
        if not isinstance(step_id_raw, str) or not step_id_raw.strip():
            raise ValueError(f"plan.steps[{index - 1}].step_id обязателен.")
        if not isinstance(title_raw, str) or not title_raw.strip():
            raise ValueError(f"plan.steps[{index - 1}].title обязателен.")
        if not isinstance(description_raw, str):
            raise ValueError(f"plan.steps[{index - 1}].description должен быть строкой.")
        allowed_raw = item.get("allowed_tool_kinds")
        if not isinstance(allowed_raw, list) or not any(
            isinstance(tool_name, str) and tool_name.strip() for tool_name in allowed_raw
        ):
            raise ValueError(
                f"plan.steps[{index - 1}].allowed_tool_kinds "
                "должен содержать хотя бы один инструмент."
            )
        inputs_raw = item.get("inputs")
        inputs = inputs_raw if isinstance(inputs_raw, dict) else {}
        expected_outputs_raw = item.get("expected_outputs")
        expected_outputs = (
            [
                str(value).strip()
                for value in expected_outputs_raw
                if isinstance(value, str) and value.strip()
            ]
            if isinstance(expected_outputs_raw, list)
            else []
        )
        acceptance_checks_raw = item.get("acceptance_checks")
        acceptance_checks = (
            [
                str(value).strip()
                for value in acceptance_checks_raw
                if isinstance(value, str) and value.strip()
            ]
            if isinstance(acceptance_checks_raw, list)
            else []
        )
        task_steps.append(
            TaskStepContract(
                step_id=step_id_raw.strip(),
                title=title_raw.strip(),
                description=description_raw,
                allowed_tool_kinds=[
                    tool_name.strip()
                    for tool_name in allowed_raw
                    if isinstance(tool_name, str) and tool_name.strip()
                ],
                inputs={str(key): value for key, value in inputs.items()},
                expected_outputs=expected_outputs,
                acceptance_checks=acceptance_checks,
            )
        )

    budgets_raw = plan.get("budgets")
    budgets = dict(budgets_raw) if isinstance(budgets_raw, dict) else {"max_attempts": 1}
    approvals_raw = plan.get("approvals")
    approvals = dict(approvals_raw) if isinstance(approvals_raw, dict) else {}
    if approved_categories is not None:
        approvals["approved_categories"] = list(approved_categories)
    verifier_raw = plan.get("verifier")
    verifier = dict(verifier_raw) if isinstance(verifier_raw, dict) else {}

    packet = TaskPacket(
        task_id=f"task-{uuid.uuid4().hex}",
        session_id=session_id,
        trace_id=trace_id,
        goal=str(plan.get("goal") or ""),
        steps=task_steps,
        policy={"source": "plan_execute"},
        scope={"workspace_root": workspace_root},
        budgets=budgets,
        approvals=approvals,
        verifier=verifier,
        context={
            "plan_id": plan.get("plan_id"),
            "plan_revision": plan.get("plan_revision"),
            "source": "plan_execute",
        },
    )
    return with_task_packet_hash(packet)


def serialize_task_packet_payload(packet: TaskPacket) -> dict[str, JSONValue]:
    return {
        "task_id": packet.task_id,
        "session_id": packet.session_id,
        "trace_id": packet.trace_id,
        "goal": packet.goal,
        "packet_revision": packet.packet_revision,
        "packet_hash": packet.packet_hash,
        "messages": [
            {"role": message.role, "content": message.content} for message in packet.messages
        ],
        "steps": [
            {
                "step_id": step.step_id,
                "title": step.title,
                "description": step.description,
                "allowed_tool_kinds": list(step.allowed_tool_kinds),
                "inputs": dict(step.inputs),
                "expected_outputs": list(step.expected_outputs),
                "acceptance_checks": list(step.acceptance_checks),
            }
            for step in packet.steps
        ],
        "constraints": list(packet.constraints),
        "policy": dict(packet.policy),
        "scope": dict(packet.scope),
        "budgets": dict(packet.budgets),
        "approvals": dict(packet.approvals),
        "verifier": dict(packet.verifier),
        "context": dict(packet.context),
    }


def deserialize_task_packet_payload(payload: object) -> TaskPacket:
    if not isinstance(payload, dict):
        raise ValueError("task_packet должен быть объектом.")
    task_id_raw = payload.get("task_id")
    session_id_raw = payload.get("session_id")
    trace_id_raw = payload.get("trace_id")
    goal_raw = payload.get("goal")
    if not isinstance(task_id_raw, str) or not task_id_raw.strip():
        raise ValueError("task_packet.task_id обязателен.")
    if not isinstance(session_id_raw, str) or not session_id_raw.strip():
        raise ValueError("task_packet.session_id обязателен.")
    if not isinstance(trace_id_raw, str) or not trace_id_raw.strip():
        raise ValueError("task_packet.trace_id обязателен.")
    if not isinstance(goal_raw, str):
        raise ValueError("task_packet.goal должен быть строкой.")
    packet_revision_raw = payload.get("packet_revision")
    packet_revision = (
        packet_revision_raw
        if isinstance(packet_revision_raw, int) and packet_revision_raw > 0
        else 1
    )
    packet_hash_raw = payload.get("packet_hash")
    packet_hash = packet_hash_raw.strip() if isinstance(packet_hash_raw, str) else ""

    messages_raw = payload.get("messages")
    messages: list[dict[str, str]] = []
    if isinstance(messages_raw, list):
        for item in messages_raw:
            if not isinstance(item, dict):
                continue
            role_raw = item.get("role")
            content_raw = item.get("content")
            if isinstance(role_raw, str) and isinstance(content_raw, str):
                messages.append({"role": role_raw, "content": content_raw})

    steps_raw = payload.get("steps")
    steps: list[TaskStepContract] = []
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError("task_packet.steps должен содержать хотя бы один шаг.")
    for index, item in enumerate(steps_raw):
        if not isinstance(item, dict):
            raise ValueError(f"task_packet.steps[{index}] должен быть объектом.")
        step_id_raw = item.get("step_id")
        title_raw = item.get("title")
        description_raw = item.get("description")
        if not isinstance(step_id_raw, str) or not step_id_raw.strip():
            raise ValueError(f"task_packet.steps[{index}].step_id обязателен.")
        if not isinstance(title_raw, str) or not title_raw.strip():
            raise ValueError(f"task_packet.steps[{index}].title обязателен.")
        if not isinstance(description_raw, str):
            raise ValueError(f"task_packet.steps[{index}].description должен быть строкой.")
        allowed_raw = item.get("allowed_tool_kinds")
        allowed = (
            [value.strip() for value in allowed_raw if isinstance(value, str) and value.strip()]
            if isinstance(allowed_raw, list)
            else []
        )
        if not allowed:
            raise ValueError(
                f"task_packet.steps[{index}].allowed_tool_kinds должен содержать инструмент."
            )
        inputs_raw = item.get("inputs")
        expected_outputs_raw = item.get("expected_outputs")
        acceptance_checks_raw = item.get("acceptance_checks")
        steps.append(
            TaskStepContract(
                step_id=step_id_raw.strip(),
                title=title_raw.strip(),
                description=description_raw,
                allowed_tool_kinds=allowed,
                inputs={str(key): value for key, value in inputs_raw.items()}
                if isinstance(inputs_raw, dict)
                else {},
                expected_outputs=[
                    value.strip()
                    for value in expected_outputs_raw
                    if isinstance(value, str) and value.strip()
                ]
                if isinstance(expected_outputs_raw, list)
                else [],
                acceptance_checks=[
                    value.strip()
                    for value in acceptance_checks_raw
                    if isinstance(value, str) and value.strip()
                ]
                if isinstance(acceptance_checks_raw, list)
                else [],
            )
        )

    policy_raw = payload.get("policy")
    scope_raw = payload.get("scope")
    budgets_raw = payload.get("budgets")
    approvals_raw = payload.get("approvals")
    verifier_raw = payload.get("verifier")
    context_raw = payload.get("context")

    normalized_messages: list[MWVMessage] = []
    for item in messages:
        role_raw = item["role"]
        if role_raw not in {"system", "user", "assistant", "tool"}:
            continue
        normalized_role = cast(
            Literal["system", "user", "assistant", "tool"],
            role_raw,
        )
        normalized_messages.append(MWVMessage(role=normalized_role, content=item["content"]))

    return TaskPacket(
        task_id=task_id_raw.strip(),
        session_id=session_id_raw.strip(),
        trace_id=trace_id_raw.strip(),
        goal=goal_raw,
        packet_revision=packet_revision,
        packet_hash=packet_hash,
        messages=normalized_messages,
        steps=steps,
        constraints=[
            value.strip()
            for value in payload.get("constraints", [])
            if isinstance(value, str) and value.strip()
        ]
        if isinstance(payload.get("constraints"), list)
        else [],
        policy=dict(policy_raw) if isinstance(policy_raw, dict) else {},
        scope=dict(scope_raw) if isinstance(scope_raw, dict) else {},
        budgets=dict(budgets_raw) if isinstance(budgets_raw, dict) else {},
        approvals=(dict(approvals_raw) if isinstance(approvals_raw, dict) else {}),
        verifier=dict(verifier_raw) if isinstance(verifier_raw, dict) else {},
        context=dict(context_raw) if isinstance(context_raw, dict) else {},
    )


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


def compute_plan_completion_state(plan: dict[str, JSONValue]) -> str:
    steps_raw = plan.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        return "running"
    has_failed = False
    done_count = 0
    total = 0
    for step in steps_raw:
        if not isinstance(step, dict):
            continue
        total += 1
        status = step.get("status")
        if status == "failed":
            has_failed = True
        elif status == "done":
            done_count += 1
    if total == 0:
        return "running"
    if has_failed:
        return "failed"
    if done_count == total:
        return "completed"
    return "running"


def _task_packet_with_resume_state(
    packet: TaskPacket,
    *,
    step_results: list[dict[str, JSONValue]],
    changes: list[dict[str, JSONValue]],
    tool_calls_used: int,
    diff_size: int,
) -> TaskPacket:
    context = dict(packet.context)
    context["plan_runner_resume"] = {
        "step_results": step_results,
        "changes": changes,
        "tool_calls_used": tool_calls_used,
        "diff_size": diff_size,
    }
    return with_task_packet_hash(
        TaskPacket(
            task_id=packet.task_id,
            session_id=packet.session_id,
            trace_id=packet.trace_id,
            goal=packet.goal,
            packet_revision=packet.packet_revision,
            messages=list(packet.messages),
            steps=list(packet.steps),
            constraints=list(packet.constraints),
            policy=dict(packet.policy),
            scope=dict(packet.scope),
            budgets=dict(packet.budgets),
            approvals=dict(packet.approvals),
            verifier=dict(packet.verifier),
            context=context,
        )
    )


def _task_execution_payload(
    *,
    packet: TaskPacket,
    run_result: MWVRunResult,
    step_results: list[dict[str, JSONValue]],
    stop_reason_code: str | None,
) -> dict[str, JSONValue]:
    verification = run_result.verification_result
    excerpt = verification.excerpt or extract_verifier_excerpt(
        verification,
        max_lines=3,
        max_chars=300,
    )
    payload: dict[str, JSONValue] = {
        "runner": "mwv_packet_runner",
        "task_packet_hash": packet.packet_hash,
        "attempt": run_result.attempt,
        "max_attempts": run_result.max_attempts,
        "work_status": run_result.work_result.status.value,
        "verifier_status": verification.status.value,
        "work_summary": run_result.work_result.summary,
        "verification_excerpt": excerpt,
        "step_results": step_results,
        "elapsed_ms": run_result.work_result.elapsed_ms,
        "files_touched": run_result.work_result.files_touched,
        "tool_calls_used": run_result.work_result.tool_calls_used,
        "diff_size": run_result.work_result.diff_size,
        "root_cause_tag": run_result.work_result.root_cause_tag,
        "verifier_fail_type": verification.fail_type,
        "verifier_profile": verification.verifier_profile,
    }
    if stop_reason_code is not None:
        payload["stop_reason_code"] = stop_reason_code
    return payload


def _step_evidence(
    snapshot: dict[str, JSONValue],
    *,
    utc_now_iso_fn: Callable[[], str],
) -> dict[str, JSONValue]:
    return {
        "runner": "mwv_packet_runner",
        "completed_at": utc_now_iso_fn(),
        "operation": snapshot.get("operation"),
        "result": snapshot.get("result"),
        "tool_calls_used": snapshot.get("tool_calls_used"),
        "changes": snapshot.get("changes"),
    }


def _apply_step_results_to_plan(
    plan: dict[str, JSONValue],
    *,
    step_results: list[dict[str, JSONValue]],
    plan_mark_step_fn: Callable[
        [dict[str, JSONValue], str, str, dict[str, JSONValue] | None],
        dict[str, JSONValue],
    ],
    utc_now_iso_fn: Callable[[], str],
) -> dict[str, JSONValue]:
    updated = plan
    for snapshot in step_results:
        step_id_raw = snapshot.get("step_id")
        status_raw = snapshot.get("status")
        if not isinstance(step_id_raw, str) or not step_id_raw.strip():
            continue
        if status_raw == "done":
            step_status = "done"
        elif status_raw == "failed":
            step_status = "failed"
        elif status_raw == "waiting_approval":
            step_status = "waiting_approval"
        else:
            continue
        updated = plan_mark_step_fn(
            updated,
            step_id_raw.strip(),
            step_status,
            _step_evidence(snapshot, utc_now_iso_fn=utc_now_iso_fn),
        )
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
    build_ui_approval_decision_fn: Callable[
        [
            dict[str, JSONValue],
            str,
            str,
            dict[str, JSONValue],
            str | None,
            dict[str, JSONValue] | None,
        ],
        dict[str, JSONValue],
    ],
    decision_workflow_context_fn: Callable[
        [str, dict[str, JSONValue] | None, dict[str, JSONValue] | None],
        dict[str, JSONValue],
    ],
    serialize_approval_request_fn: Callable[[object], dict[str, JSONValue] | None],
    utc_now_iso_fn: Callable[[], str],
) -> None:
    hub: WorkflowHubProtocol = app["ui_hub"]
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

    task_packet_raw = task.get("task_packet")
    try:
        packet = deserialize_task_packet_payload(task_packet_raw)
    except ValueError:
        failed_plan = plan_with_status_fn(plan, "failed")
        failed_task = task_with_status_fn(task, "failed", None)
        failed_task["execution"] = {
            "runner": "mwv_packet_runner",
            "status": "failed",
            "error": "task_packet_invalid",
        }
        await hub.set_session_workflow(
            session_id,
            mode="act",
            active_plan=failed_plan,
            active_task=failed_task,
        )
        return

    provider = app["agent_provider"]
    session_store = app["session_store"]
    agent = await provider.get()
    approved_categories = await session_store.get_categories(session_id)
    set_session_context = getattr(agent, "set_session_context", None)
    if callable(set_session_context):
        set_session_context(session_id, approved_categories)

    async def _security_loader(
        _hub: WorkflowHubProtocol,
        _session_id: str,
    ) -> tuple[dict[str, bool], dict[str, JSONValue]]:
        policy = await _hub.get_session_policy(_session_id)
        effective_tools = dict(getattr(agent, "tools_enabled", {}))
        session_override = await _hub.get_session_tools_state(_session_id)
        if isinstance(session_override, dict):
            for key, value in session_override.items():
                if isinstance(key, str) and isinstance(value, bool):
                    effective_tools[key] = value
        return effective_tools, {
            "profile": str(policy.get("profile") or "default"),
            "safe_mode_effective": bool(effective_tools.get("safe_mode", False)),
        }

    async def _workspace_root_loader(
        _hub: WorkflowHubProtocol,
        _session_id: str,
    ) -> Path:
        scope_root_raw = packet.scope.get("workspace_root")
        if isinstance(scope_root_raw, str) and scope_root_raw.strip():
            return Path(scope_root_raw)
        session_root_raw = await _hub.get_workspace_root(_session_id)
        if isinstance(session_root_raw, str) and session_root_raw.strip():
            return Path(session_root_raw)
        return Path(".").resolve()

    await apply_agent_runtime_state(
        agent=agent,
        hub=hub,
        session_id=session_id,
        load_effective_session_security_fn=_security_loader,
        resolve_workspace_root_fn=_workspace_root_loader,
        normalize_mode_value_fn=normalize_mode_value_fn,
        normalize_plan_payload_fn=normalize_plan_payload_fn,
        normalize_task_payload_fn=normalize_task_payload_fn,
        normalize_auto_state_fn=lambda value: value if isinstance(value, dict) else None,
    )

    context_root = str(packet.scope.get("workspace_root") or "")
    from core.mwv.models import RunContext  # local import to avoid protocol layering drift

    max_attempts_raw = packet.budgets.get("max_attempts")
    max_attempts = max_attempts_raw if isinstance(max_attempts_raw, int) else 1

    run_context = RunContext(
        session_id=session_id,
        trace_id=packet.trace_id,
        workspace_root=context_root,
        safe_mode=bool(getattr(agent, "tools_enabled", {}).get("safe_mode", False)),
        approved_categories=sorted(str(item) for item in approved_categories),
        max_retries=max(0, max_attempts - 1),
    )

    try:
        run_result = await asyncio.to_thread(agent.run_task_packet, packet, run_context)
    except TaskPacketApprovalPending as exc:
        approval_payload = serialize_approval_request_fn(exc.request)
        if approval_payload is None:
            failed_plan = plan_with_status_fn(plan, "failed")
            failed_task = task_with_status_fn(task, "failed", exc.blocked_step_id)
            failed_task["execution"] = {
                "runner": "mwv_packet_runner",
                "status": "failed",
                "error": "approval_payload_missing",
            }
            await hub.set_session_workflow(
                session_id,
                mode="act",
                active_plan=failed_plan,
                active_task=failed_task,
            )
            return
        plan = _apply_step_results_to_plan(
            plan,
            step_results=exc.step_results
            + [
                {
                    "step_id": exc.blocked_step_id,
                    "description": next(
                        (
                            snapshot.get("description")
                            for snapshot in exc.step_results
                            if snapshot.get("step_id") == exc.blocked_step_id
                        ),
                        exc.blocked_step_id,
                    ),
                    "status": "waiting_approval",
                    "operation": None,
                    "result": "Требуется подтверждение",
                    "tool_calls_used": 0,
                    "changes": [],
                }
            ],
            plan_mark_step_fn=plan_mark_step_fn,
            utc_now_iso_fn=utc_now_iso_fn,
        )
        resumed_packet = _task_packet_with_resume_state(
            packet,
            step_results=exc.step_results,
            changes=exc.changes,
            tool_calls_used=exc.tool_calls_used,
            diff_size=exc.diff_size,
        )
        task = task_with_status_fn(task, "running", exc.blocked_step_id)
        task["task_packet"] = serialize_task_packet_payload(resumed_packet)
        task["execution"] = {
            "runner": "mwv_packet_runner",
            "status": "waiting_approval",
            "blocked_step_id": exc.blocked_step_id,
            "resume_state": {
                "step_results": exc.step_results,
                "changes": exc.changes,
                "tool_calls_used": exc.tool_calls_used,
                "diff_size": exc.diff_size,
            },
        }
        ui_decision = build_ui_approval_decision_fn(
            approval_payload,
            session_id,
            "plan.execute_runner",
            {
                "plan_id": plan_id,
                "task_id": task_id,
                "plan_revision": plan.get("plan_revision"),
                "blocked_step_id": exc.blocked_step_id,
                "task_packet": serialize_task_packet_payload(resumed_packet),
            },
            packet.trace_id,
            decision_workflow_context_fn("act", plan, task),
        )
        await hub.set_session_decision(session_id, ui_decision)
        await hub.set_session_workflow(
            session_id,
            mode="act",
            active_plan=plan,
            active_task=task,
        )
        return
    except Exception as exc:  # noqa: BLE001
        failed_plan = plan_with_status_fn(plan, "failed")
        failed_task = task_with_status_fn(task, "failed", None)
        failed_task["execution"] = {
            "runner": "mwv_packet_runner",
            "status": "failed",
            "error": str(exc),
        }
        await hub.set_session_workflow(
            session_id,
            mode="act",
            active_plan=failed_plan,
            active_task=failed_task,
        )
        return

    step_results_raw = run_result.work_result.diagnostics.get("step_results")
    step_results = (
        [dict(item) for item in step_results_raw if isinstance(item, dict)]
        if isinstance(step_results_raw, list)
        else []
    )
    plan = _apply_step_results_to_plan(
        plan,
        step_results=step_results,
        plan_mark_step_fn=plan_mark_step_fn,
        utc_now_iso_fn=utc_now_iso_fn,
    )
    stop_reason_code_raw = run_result.work_result.diagnostics.get("stop_reason_code")
    stop_reason_code = stop_reason_code_raw if isinstance(stop_reason_code_raw, str) else None
    if (
        run_result.work_result.status == WorkStatus.SUCCESS
        and run_result.verification_result.status == VerificationStatus.PASSED
    ):
        plan = plan_with_status_fn(plan, "completed")
        task = task_with_status_fn(task, "completed", None)
    else:
        plan = plan_with_status_fn(plan, "failed")
        task = task_with_status_fn(task, "failed", None)
    task["execution"] = _task_execution_payload(
        packet=packet,
        run_result=run_result,
        step_results=step_results,
        stop_reason_code=stop_reason_code,
    )
    await hub.set_session_workflow(
        session_id,
        mode="act",
        active_plan=plan,
        active_task=task,
    )


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
