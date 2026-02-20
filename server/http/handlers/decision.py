from __future__ import annotations

import asyncio
import uuid

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from core.approval_policy import ApprovalRequired
from server.http.common.responses import error_response, json_response
from server.http_api import (
    _apply_agent_runtime_state,
    _decision_categories,
    _decision_mismatch_response,
    _decision_type_value,
    _decision_with_status,
    _ensure_session_owned,
    _load_effective_session_security,
    _normalize_json_value,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_policy_profile,
    _normalize_task_payload,
    _normalize_ui_decision,
    _plan_apply_edit_operation,
    _plan_revision_value,
    _plan_with_status,
    _resolve_agent,
    _resolve_workspace_root_candidate,
    _run_plan_runner,
    _serialize_approval_request,
    _set_current_plan_step_status,
    _utc_now_iso,
    _workspace_root_for_session,
)
from shared.models import JSONValue
from tools.workspace_tools import set_workspace_root as set_runtime_workspace_root


async def handle_ui_decision_respond(request: web.Request) -> web.Response:
    hub = request.app["ui_hub"]
    session_store = request.app["session_store"]
    agent_lock = request.app["agent_lock"]

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    allowed_keys = {"session_id", "decision_id", "choice", "edited_action", "edited_plan"}
    forbidden_keys = sorted(
        {str(key).strip() for key in payload if str(key).strip() not in allowed_keys}
    )
    if forbidden_keys:
        return error_response(
            status=400,
            message=(
                "Допустимы session_id, decision_id, choice, edited_action, edited_plan. "
                f"Запрещённые поля: {', '.join(forbidden_keys)}."
            ),
            error_type="invalid_request_error",
            code="invalid_request_error",
            details={"forbidden_fields": forbidden_keys},
        )

    session_id_raw = payload.get("session_id")
    decision_id_raw = payload.get("decision_id")
    choice_raw = payload.get("choice")
    if not isinstance(session_id_raw, str) or not session_id_raw.strip():
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(decision_id_raw, str) or not decision_id_raw.strip():
        return error_response(
            status=400,
            message="decision_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(choice_raw, str) or not choice_raw.strip():
        return error_response(
            status=400,
            message="choice обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    session_id = session_id_raw.strip()
    decision_id = decision_id_raw.strip()
    choice = choice_raw.strip()
    edited_action_raw = payload.get("edited_action")
    edited_plan_raw = payload.get("edited_plan")
    if edited_action_raw is not None and not isinstance(edited_action_raw, dict):
        return error_response(
            status=400,
            message="edited_action должен быть объектом или null.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if edited_plan_raw is not None and not isinstance(edited_plan_raw, dict):
        return error_response(
            status=400,
            message="edited_plan должен быть объектом или null.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    edited_action = edited_action_raw if isinstance(edited_action_raw, dict) else None
    edited_plan = edited_plan_raw if isinstance(edited_plan_raw, dict) else None
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error

    current_decision = _normalize_ui_decision(
        await hub.get_session_decision(session_id),
        session_id=session_id,
    )
    if current_decision is None:
        return error_response(
            status=404,
            message="Pending decision not found.",
            error_type="invalid_request_error",
            code="decision_not_found",
        )
    current_id = current_decision.get("id")
    if not isinstance(current_id, str) or current_id != decision_id:
        return _decision_mismatch_response(
            expected_id=decision_id,
            actual_decision=current_decision,
        )
    current_status = current_decision.get("status")
    if current_status != "pending":
        status_value = current_status if isinstance(current_status, str) else "unknown"
        return json_response(
            {
                "ok": True,
                "decision": current_decision,
                "status": status_value,
                "resume_started": False,
                "already_resolved": status_value in {"resolved", "rejected"},
                "resume": None,
            }
        )
    decision_type = _decision_type_value(current_decision)
    if decision_type == "runtime_packet":
        return error_response(
            status=409,
            message=(
                "runtime_packet decision должен обрабатываться через "
                "POST /ui/api/decision/runtime/respond."
            ),
            error_type="invalid_request_error",
            code="decision_type_not_supported",
            details={"decision_type": decision_type},
        )
    if decision_type == "tool_approval":
        context_raw = current_decision.get("context")
        context = context_raw if isinstance(context_raw, dict) else {}
        source_endpoint_raw = context.get("source_endpoint")
        source_endpoint = source_endpoint_raw if isinstance(source_endpoint_raw, str) else ""
        if source_endpoint == "workspace.tool" and not _decision_categories(current_decision):
            return error_response(
                status=404,
                message="Pending decision not found.",
                error_type="invalid_request_error",
                code="decision_not_found",
            )
    if decision_type == "plan_execute":
        if choice not in {"approve_once", "edit_plan", "reject"}:
            return error_response(
                status=400,
                message="Для plan_execute доступны approve_once|edit_plan|reject.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
    elif decision_type != "tool_approval":
        return error_response(
            status=409,
            message=f"Unsupported decision_type: {decision_type}.",
            error_type="invalid_request_error",
            code="decision_type_not_supported",
            details={"decision_type": decision_type},
        )
    elif choice in {"edit_plan"}:
        return error_response(
            status=400,
            message="edit_plan доступен только для plan_execute decision.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    else:
        if choice not in {"approve_once", "approve_session", "edit_and_approve", "reject"}:
            return error_response(
                status=400,
                message=(
                    "Для tool decision доступны "
                    "approve_once|approve_session|edit_and_approve|reject."
                ),
                error_type="invalid_request_error",
                code="invalid_request_error",
            )

    async def _resolve_tool_decision() -> dict[str, JSONValue]:
        context_raw = current_decision.get("context")
        context = context_raw if isinstance(context_raw, dict) else {}
        source_endpoint_raw = context.get("source_endpoint")
        source_endpoint = source_endpoint_raw if isinstance(source_endpoint_raw, str) else ""
        resume_payload_raw = context.get("resume_payload")
        resume_payload = resume_payload_raw if isinstance(resume_payload_raw, dict) else {}

        required_categories = _decision_categories(current_decision)
        if choice == "approve_session" and required_categories:
            await session_store.approve(session_id, required_categories)
        approved_categories = await session_store.get_categories(session_id)
        one_call_categories = (
            approved_categories | required_categories
            if choice in {"approve_once", "edit_and_approve"}
            else approved_categories
        )

        if source_endpoint == "workspace.root_select":
            if choice == "edit_and_approve":
                return {
                    "ok": False,
                    "error": "edit_and_approve поддерживается только для workspace.tool.",
                    "source_endpoint": source_endpoint,
                }
            root_raw = resume_payload.get("root_path")
            if not isinstance(root_raw, str) or not root_raw.strip():
                return {
                    "ok": False,
                    "error": "resume_payload.root_path is missing.",
                    "source_endpoint": source_endpoint,
                }
            _, effective_policy = await _load_effective_session_security(
                hub=hub,
                session_id=session_id,
            )
            profile = _normalize_policy_profile(effective_policy.get("profile"))
            try:
                target_root = _resolve_workspace_root_candidate(
                    root_raw.strip(),
                    policy_profile=profile,
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "error": str(exc),
                    "source_endpoint": source_endpoint,
                }
            await hub.set_workspace_root(session_id, str(target_root))
            return {
                "ok": True,
                "source_endpoint": source_endpoint,
                "data": {"root_path": str(target_root)},
            }

        if source_endpoint != "workspace.tool":
            return {
                "ok": False,
                "error": f"decision_resume_not_supported: {source_endpoint or 'unknown'}",
                "source_endpoint": source_endpoint,
            }

        tool_name_raw = resume_payload.get("tool_name")
        if not isinstance(tool_name_raw, str) or not tool_name_raw.strip():
            return {
                "ok": False,
                "error": "resume_payload.tool_name is missing.",
                "source_endpoint": source_endpoint,
            }
        tool_name = tool_name_raw.strip()
        args_raw = resume_payload.get("args")
        args = args_raw if isinstance(args_raw, dict) else {}
        raw_input_raw = resume_payload.get("raw_input")
        raw_input = raw_input_raw if isinstance(raw_input_raw, str) else f"decision:{tool_name}"

        if choice == "edit_and_approve":
            if edited_action is None:
                return {
                    "ok": False,
                    "error": "edited_action обязателен для edit_and_approve.",
                    "source_endpoint": source_endpoint,
                }
            invalid_edit_keys = sorted({str(key) for key in edited_action if str(key) != "args"})
            if invalid_edit_keys:
                return {
                    "ok": False,
                    "error": (
                        "edit_and_approve может менять только args. "
                        f"Запрещённые поля: {', '.join(invalid_edit_keys)}."
                    ),
                    "source_endpoint": source_endpoint,
                }
            edited_args_raw = edited_action.get("args")
            if edited_args_raw is not None and not isinstance(edited_args_raw, dict):
                return {
                    "ok": False,
                    "error": "edited_action.args должен быть объектом.",
                    "source_endpoint": source_endpoint,
                }
            if isinstance(edited_args_raw, dict):
                args = edited_args_raw

        try:
            agent = await _resolve_agent(request)
        except ModelNotAllowedError as exc:
            return {
                "ok": False,
                "error": f"model_not_allowed: {exc.model_id}",
                "source_endpoint": source_endpoint,
            }
        if agent is None:
            return {
                "ok": False,
                "error": "model_not_selected",
                "source_endpoint": source_endpoint,
            }
        session_root = await _workspace_root_for_session(hub, session_id)
        async with agent_lock:
            set_runtime_workspace_root(session_root)
            try:
                await _apply_agent_runtime_state(agent=agent, hub=hub, session_id=session_id)
                agent.set_session_context(session_id, one_call_categories)
                try:
                    result = agent.call_tool(
                        tool_name,
                        args={
                            str(key): _normalize_json_value(value) for key, value in args.items()
                        },
                        raw_input=raw_input,
                    )
                except ApprovalRequired as exc:
                    approval_payload = _serialize_approval_request(exc.request)
                    return {
                        "ok": False,
                        "error": "approval_required",
                        "approval_request": approval_payload,
                        "source_endpoint": source_endpoint,
                        "tool_name": tool_name,
                    }
            finally:
                set_runtime_workspace_root(None)
        if not result.ok:
            return {
                "ok": False,
                "error": result.error or "tool failed",
                "source_endpoint": source_endpoint,
                "tool_name": tool_name,
            }
        return {
            "ok": True,
            "source_endpoint": source_endpoint,
            "tool_name": tool_name,
            "data": _normalize_json_value(result.data),
        }

    async def _resolve_plan_execute_decision() -> dict[str, JSONValue]:
        workflow = await hub.get_session_workflow(session_id)
        mode = _normalize_mode_value(workflow.get("mode"), default="ask")
        plan = _normalize_plan_payload(workflow.get("active_plan"))
        task = _normalize_task_payload(workflow.get("active_task"))
        if plan is None:
            return {"ok": False, "error": "plan_not_found"}
        if choice == "edit_plan":
            if mode != "plan":
                return {"ok": False, "error": "mode_not_plan"}
            if edited_plan is None:
                return {"ok": False, "error": "edited_plan обязателен для edit_plan."}
            revision_raw = edited_plan.get("plan_revision")
            if not isinstance(revision_raw, int) or revision_raw <= 0:
                return {"ok": False, "error": "edited_plan.plan_revision обязателен."}
            current_revision = _plan_revision_value(plan)
            if revision_raw != current_revision:
                return {
                    "ok": False,
                    "error": "plan_revision_mismatch",
                    "details": {
                        "expected_revision": current_revision,
                        "actual_revision": revision_raw,
                    },
                }
            operation_raw = edited_plan.get("operation")
            if not isinstance(operation_raw, dict):
                return {"ok": False, "error": "edited_plan.operation обязателен."}
            operation = {
                str(key): _normalize_json_value(value) for key, value in operation_raw.items()
            }
            try:
                updated_plan = _plan_apply_edit_operation(plan=plan, operation=operation)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            await hub.set_session_workflow(
                session_id,
                mode="plan",
                active_plan=updated_plan,
                active_task=None,
            )
            return {
                "ok": True,
                "action": "edit_plan",
                "plan_id": updated_plan.get("plan_id"),
                "plan_revision": _plan_revision_value(updated_plan),
            }

        if mode != "plan":
            return {"ok": False, "error": "switch_to_act_required"}
        if plan.get("status") != "approved":
            return {"ok": False, "error": "plan_not_approved"}
        decision_context_raw = current_decision.get("context")
        decision_context = decision_context_raw if isinstance(decision_context_raw, dict) else {}
        resume_payload_raw = decision_context.get("resume_payload")
        resume_payload = resume_payload_raw if isinstance(resume_payload_raw, dict) else {}
        expected_revision_raw = resume_payload.get("plan_revision")
        if isinstance(expected_revision_raw, int) and expected_revision_raw > 0:
            current_revision = _plan_revision_value(plan)
            if expected_revision_raw != current_revision:
                return {
                    "ok": False,
                    "error": "plan_revision_mismatch",
                    "details": {
                        "expected_revision": expected_revision_raw,
                        "actual_revision": current_revision,
                    },
                }
        running_plan = _plan_with_status(plan, status="running")
        task_payload: dict[str, JSONValue] = {
            "task_id": f"task-{uuid.uuid4().hex}",
            "plan_id": running_plan.get("plan_id"),
            "plan_hash": running_plan.get("plan_hash"),
            "current_step_id": None,
            "status": "running",
            "started_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        await hub.set_session_workflow(
            session_id,
            mode="act",
            active_plan=running_plan,
            active_task=task_payload,
        )
        plan_id_raw = task_payload.get("plan_id")
        task_id_raw = task_payload.get("task_id")
        if isinstance(plan_id_raw, str) and isinstance(task_id_raw, str):
            asyncio.create_task(
                _run_plan_runner(
                    app=request.app,
                    session_id=session_id,
                    plan_id=plan_id_raw,
                    task_id=task_id_raw,
                )
            )
        return {
            "ok": True,
            "action": "approve_once",
            "plan_id": running_plan.get("plan_id"),
            "task_id": task_payload.get("task_id"),
            "mode": "act",
            "previous_task": task,
        }

    if choice == "reject":
        rejected = _decision_with_status(current_decision, status="rejected", resolved=True)
        updated, latest = await hub.transition_session_decision(
            session_id,
            expected_id=decision_id,
            expected_status="pending",
            next_decision=rejected,
        )
        normalized_latest = _normalize_ui_decision(latest, session_id=session_id)
        if not updated:
            return _decision_mismatch_response(
                expected_id=decision_id,
                actual_decision=normalized_latest,
            )
        normalized = (
            normalized_latest
            if normalized_latest is not None
            else _normalize_ui_decision(rejected, session_id=session_id)
        )
        resolved_status_raw = normalized.get("status") if isinstance(normalized, dict) else None
        resolved_status = (
            resolved_status_raw if isinstance(resolved_status_raw, str) else "rejected"
        )
        return json_response(
            {
                "ok": True,
                "decision": normalized,
                "status": resolved_status,
                "resume_started": False,
                "already_resolved": resolved_status in {"resolved", "rejected"},
                "resume": None,
            }
        )

    executing = _decision_with_status(current_decision, status="executing")
    updated, latest = await hub.transition_session_decision(
        session_id,
        expected_id=decision_id,
        expected_status="pending",
        next_decision=executing,
    )
    normalized_latest = _normalize_ui_decision(latest, session_id=session_id)
    if not updated:
        return _decision_mismatch_response(
            expected_id=decision_id,
            actual_decision=normalized_latest,
        )

    if decision_type == "plan_execute":
        resume = await _resolve_plan_execute_decision()
    else:
        resume = await _resolve_tool_decision()

    if decision_type == "tool_approval":
        await _set_current_plan_step_status(
            hub=hub,
            session_id=session_id,
            status="done" if resume.get("ok") is True else "blocked",
        )

    resolved = _decision_with_status(executing, status="resolved", resolved=True)
    updated_resolved, latest_resolved = await hub.transition_session_decision(
        session_id,
        expected_id=decision_id,
        expected_status="executing",
        next_decision=resolved,
    )
    normalized_resolved = _normalize_ui_decision(latest_resolved, session_id=session_id)
    if not updated_resolved:
        return _decision_mismatch_response(
            expected_id=decision_id,
            actual_decision=normalized_resolved,
        )
    workflow = await hub.get_session_workflow(session_id)
    return json_response(
        {
            "ok": True,
            "decision": normalized_resolved,
            "status": "resolved",
            "resume_started": True,
            "already_resolved": True,
            "resume": resume,
            "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
            "active_task": _normalize_task_payload(workflow.get("active_task")),
        }
    )
