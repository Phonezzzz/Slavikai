from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from core.approval_policy import ApprovalRequired
from server.http.common.responses import error_response, json_response
from server.http.handlers.ui_chat import handle_ui_chat_send
from server.http_api import (
    UI_DECISION_RESPONSES,
    _apply_agent_runtime_state,
    _compile_plan_to_task_packet,
    _decision_categories,
    _decision_mismatch_response,
    _decision_type_value,
    _decision_with_status,
    _ensure_session_owned,
    _load_effective_session_security,
    _normalize_auto_state,
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


def _normalize_message_lane(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "workspace":
            return "workspace"
    return "chat"


def _decision_is_expired(decision: dict[str, JSONValue]) -> bool:
    expires_at_raw = decision.get("expires_at")
    if not isinstance(expires_at_raw, str) or not expires_at_raw.strip():
        return False
    try:
        normalized = expires_at_raw.replace("Z", "+00:00")
        expires_at = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    else:
        expires_at = expires_at.astimezone(UTC)
    return datetime.now(UTC) >= expires_at


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
    if not isinstance(choice_raw, str) or choice_raw not in UI_DECISION_RESPONSES:
        return error_response(
            status=400,
            message=f"choice должен быть одним из: {', '.join(sorted(UI_DECISION_RESPONSES))}.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    session_id = session_id_raw.strip()
    decision_id = decision_id_raw.strip()
    choice = choice_raw
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
    if current_status == "pending" and _decision_is_expired(current_decision):
        expired = _decision_with_status(current_decision, status="expired", resolved=True)
        _, latest_expired = await hub.transition_session_decision(
            session_id,
            expected_id=decision_id,
            expected_status="pending",
            next_decision=expired,
        )
        normalized_expired = _normalize_ui_decision(latest_expired, session_id=session_id)
        return error_response(
            status=410,
            message="Decision packet протух и больше не может быть применён.",
            error_type="invalid_request_error",
            code="decision_expired",
            details={"decision": normalized_expired},
        )
    if current_status != "pending":
        status_value = current_status if isinstance(current_status, str) else "unknown"
        if status_value in {"resolved", "rejected", "expired"}:
            return error_response(
                status=409,
                message="Decision уже завершён и не может быть выполнен повторно.",
                error_type="invalid_request_error",
                code="decision_already_resolved",
                details={"decision_status": status_value},
            )
        return error_response(
            status=409,
            message="Decision не находится в pending состоянии.",
            error_type="invalid_request_error",
            code="decision_not_pending",
            details={"decision_status": status_value},
        )
    decision_type = _decision_type_value(current_decision)
    generic_decision_choices = {
        "ask_user",
        "proceed_safe",
        "retry",
        "abort",
        "select_skill",
        "adjust_threshold",
        "create_candidate",
    }
    effective_choice = choice
    tool_source_endpoint = ""
    if decision_type == "tool_approval":
        context_raw = current_decision.get("context")
        context = context_raw if isinstance(context_raw, dict) else {}
        source_endpoint_raw = context.get("source_endpoint")
        tool_source_endpoint = source_endpoint_raw if isinstance(source_endpoint_raw, str) else ""
        if tool_source_endpoint == "workspace.tool" and not _decision_categories(current_decision):
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
    elif decision_type == "tool_approval":
        if choice in {"edit_plan"}:
            return error_response(
                status=400,
                message="edit_plan доступен только для plan_execute decision.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
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
    else:
        if choice == "reject":
            effective_choice = "abort"
        if effective_choice not in generic_decision_choices:
            return error_response(
                status=400,
                message=(
                    "Для agent_decision доступны "
                    "ask_user|proceed_safe|retry|abort|select_skill|adjust_threshold|create_candidate."
                ),
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
    if tool_source_endpoint == "chat.run_root":
        if choice == "approve_session":
            return error_response(
                status=400,
                message="approve_session не поддерживается для chat.run_root.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        if choice not in {"approve_once", "reject"}:
            return error_response(
                status=400,
                message="Для chat.run_root доступны approve_once|reject.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
    if tool_source_endpoint == "chat.run_missing_file" and choice not in {"approve_once", "reject"}:
        return error_response(
            status=400,
            message="Для chat.run_missing_file доступны approve_once|reject.",
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

        if source_endpoint == "auto.run":
            run_id_raw = resume_payload.get("run_id")
            run_id = (
                run_id_raw.strip() if isinstance(run_id_raw, str) and run_id_raw.strip() else ""
            )
            if not run_id:
                return {
                    "ok": False,
                    "error": "resume_payload.run_id is missing.",
                    "source_endpoint": source_endpoint,
                }
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
            response_text: str | None = None
            async with agent_lock:
                await _apply_agent_runtime_state(agent=agent, hub=hub, session_id=session_id)
                agent.set_session_context(session_id, one_call_categories)
                resume_method = getattr(agent, "resume_auto_run", None)
                if not callable(resume_method):
                    return {
                        "ok": False,
                        "error": "auto_resume_not_supported",
                        "source_endpoint": source_endpoint,
                    }
                try:
                    response_text = resume_method(run_id)
                except ApprovalRequired as exc:
                    approval_payload = _serialize_approval_request(exc.request)
                    return {
                        "ok": False,
                        "error": "approval_required",
                        "approval_request": approval_payload,
                        "source_endpoint": source_endpoint,
                    }
            if not isinstance(response_text, str) or not response_text.strip():
                return {
                    "ok": False,
                    "error": "auto_run_not_found",
                    "source_endpoint": source_endpoint,
                }
            auto_state = _normalize_auto_state(getattr(agent, "last_auto_state", None))
            if auto_state is not None:
                await hub.set_session_workflow(session_id, auto_state=auto_state)
            drain_auto_progress = getattr(agent, "drain_auto_progress_events", None)
            if callable(drain_auto_progress):
                drained = drain_auto_progress()
                if isinstance(drained, list):
                    for item in drained:
                        normalized_progress = _normalize_auto_state(item)
                        if normalized_progress is None:
                            continue
                        await hub.publish(
                            session_id,
                            {
                                "type": "auto.progress",
                                "payload": {
                                    "session_id": session_id,
                                    "auto_state": normalized_progress,
                                },
                            },
                        )
            await hub.set_session_output(session_id, response_text)
            trace_id_raw = getattr(agent, "last_chat_interaction_id", None)
            trace_id = (
                trace_id_raw if isinstance(trace_id_raw, str) and trace_id_raw.strip() else None
            )
            parent_message_id_raw = resume_payload.get("user_message_id")
            parent_message_id = (
                parent_message_id_raw.strip()
                if isinstance(parent_message_id_raw, str) and parent_message_id_raw.strip()
                else None
            )
            assistant_message = hub.create_message(
                role="assistant",
                content=response_text,
                trace_id=trace_id,
                parent_user_message_id=parent_message_id,
            )
            await hub.append_message(session_id, assistant_message)
            auto_status_raw = auto_state.get("status") if isinstance(auto_state, dict) else None
            auto_status = auto_status_raw if isinstance(auto_status_raw, str) else "unknown"
            return {
                "ok": True,
                "source_endpoint": source_endpoint,
                "data": {
                    "run_id": run_id,
                    "status": auto_status,
                    "output": response_text,
                    "auto_state": auto_state,
                },
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

    async def _resolve_agent_decision() -> dict[str, JSONValue]:
        context_raw = current_decision.get("context")
        context = context_raw if isinstance(context_raw, dict) else {}
        source_endpoint_raw = context.get("source_endpoint")
        source_endpoint = (
            source_endpoint_raw.strip()
            if isinstance(source_endpoint_raw, str) and source_endpoint_raw.strip()
            else "chat.agent_decision"
        )
        resume_payload_raw = context.get("resume_payload")
        resume_payload = resume_payload_raw if isinstance(resume_payload_raw, dict) else {}

        if effective_choice in {
            "ask_user",
            "abort",
            "select_skill",
            "adjust_threshold",
            "create_candidate",
        }:
            return {
                "ok": True,
                "source_endpoint": source_endpoint,
                "data": {"action": effective_choice, "acknowledged": True},
                "resume_started": False,
            }

        source_request_raw = resume_payload.get("source_request")
        source_request = source_request_raw if isinstance(source_request_raw, dict) else None
        if source_request is None:
            return {
                "ok": False,
                "source_endpoint": source_endpoint,
                "error": "resume_payload_missing",
                "resume_started": False,
            }
        content_raw = source_request.get("content")
        if not isinstance(content_raw, str) or not content_raw.strip():
            return {
                "ok": False,
                "source_endpoint": source_endpoint,
                "error": "resume_payload_missing",
                "resume_started": False,
            }
        force_canvas_raw = source_request.get("force_canvas")
        force_canvas = force_canvas_raw is True
        lane = _normalize_message_lane(source_request.get("lane"))
        attachments_raw = source_request.get("attachments")
        attachments = attachments_raw if isinstance(attachments_raw, list) else []
        content = content_raw
        if effective_choice == "proceed_safe":
            safe_hint = (
                "SAFE MODE: only read-only/safe guidance. "
                "Do not run risky actions, commands, writes, or network steps "
                "unless explicitly approved."
            )
            content = f"{safe_hint}\n\n{content_raw}"
        payload_override: dict[str, JSONValue] = {
            "content": content,
            "force_canvas": force_canvas,
            "lane": lane,
            "attachments": attachments,
        }
        resumed_response = await handle_ui_chat_send(
            request,
            payload_override=payload_override,
            bypass_root_gate=False,
        )
        parsed_resume_payload: dict[str, JSONValue] = {}
        body_raw = resumed_response.text
        if isinstance(body_raw, str) and body_raw.strip():
            try:
                parsed = json.loads(body_raw)
                if isinstance(parsed, dict):
                    parsed_resume_payload = {
                        str(key): _normalize_json_value(value) for key, value in parsed.items()
                    }
            except json.JSONDecodeError:
                parsed_resume_payload = {}
        resume_ok = resumed_response.status < 400
        if resume_ok:
            return {
                "ok": True,
                "source_endpoint": source_endpoint,
                "data": {
                    "status_code": resumed_response.status,
                    "trace_id": parsed_resume_payload.get("trace_id"),
                    "output": parsed_resume_payload.get("output"),
                    "auto_state": parsed_resume_payload.get("auto_state"),
                    "decision": parsed_resume_payload.get("decision"),
                },
                "resume_started": True,
            }
        error_raw = parsed_resume_payload.get("error")
        resume_error: str | None = None
        if isinstance(error_raw, dict):
            message_raw = error_raw.get("message")
            if isinstance(message_raw, str) and message_raw.strip():
                resume_error = message_raw.strip()
        if resume_error is None:
            resume_error = f"{source_endpoint} failed: {resumed_response.status}"
        return {
            "ok": False,
            "source_endpoint": source_endpoint,
            "error": resume_error,
            "resume_started": True,
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
        workspace_root = await _workspace_root_for_session(hub, session_id)
        try:
            compiled_packet = _compile_plan_to_task_packet(
                plan=running_plan,
                session_id=session_id,
                trace_id=str(uuid.uuid4()),
                workspace_root=str(workspace_root),
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        task_payload["task_packet"] = {
            "task_id": compiled_packet.task_id,
            "packet_hash": compiled_packet.packet_hash,
            "packet_revision": compiled_packet.packet_revision,
            "scope": dict(compiled_packet.scope),
            "budgets": dict(compiled_packet.budgets),
            "approvals": dict(compiled_packet.approvals),
            "verifier": dict(compiled_packet.verifier),
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
                for step in compiled_packet.steps
            ],
        }
        started, _, start_error = await hub.start_plan_task_if_possible(
            session_id,
            expected_mode="plan",
            expected_plan_id=str(plan.get("plan_id")),
            expected_plan_hash=str(plan.get("plan_hash")),
            expected_plan_revision=_plan_revision_value(plan),
            running_plan=running_plan,
            task_payload=task_payload,
            next_mode="act",
        )
        if not started:
            if start_error == "mode_mismatch":
                return {"ok": False, "error": "mode_not_plan"}
            if start_error == "plan_mismatch":
                return {"ok": False, "error": "plan_state_conflict"}
            return {"ok": False, "error": start_error or "plan_state_conflict"}
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

    if (
        decision_type == "tool_approval"
        and tool_source_endpoint == "chat.run_missing_file"
        and choice == "approve_once"
    ):
        resolved = _decision_with_status(current_decision, status="resolved", resolved=True)
        updated, latest = await hub.transition_session_decision(
            session_id,
            expected_id=decision_id,
            expected_status="pending",
            next_decision=resolved,
        )
        normalized_latest = _normalize_ui_decision(latest, session_id=session_id)
        if not updated:
            return _decision_mismatch_response(
                expected_id=decision_id,
                actual_decision=normalized_latest,
            )
        workflow = await hub.get_session_workflow(session_id)
        return json_response(
            {
                "ok": True,
                "decision": normalized_latest,
                "status": "resolved",
                "resume_started": True,
                "already_resolved": True,
                "resume": {
                    "ok": True,
                    "source_endpoint": "chat.run_missing_file",
                    "data": {"acknowledged": True},
                },
                "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
                "active_task": _normalize_task_payload(workflow.get("active_task")),
                "auto_state": _normalize_auto_state(workflow.get("auto_state")),
            }
        )

    if (
        decision_type == "tool_approval"
        and tool_source_endpoint == "chat.run_root"
        and choice == "approve_once"
    ):
        context_raw = current_decision.get("context")
        context = context_raw if isinstance(context_raw, dict) else {}
        resume_payload_raw = context.get("resume_payload")
        resume_payload = resume_payload_raw if isinstance(resume_payload_raw, dict) else {}
        root_raw = resume_payload.get("root_path")
        if not isinstance(root_raw, str) or not root_raw.strip():
            return error_response(
                status=400,
                message="resume_payload.root_path is missing.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        source_request_raw = resume_payload.get("source_request")
        if not isinstance(source_request_raw, dict):
            return error_response(
                status=400,
                message="resume_payload.source_request is missing.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        try:
            root_candidate = _resolve_workspace_root_candidate(
                root_raw.strip(),
                policy_profile="yolo",
            )
        except ValueError as exc:
            return error_response(
                status=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        await hub.set_workspace_root(session_id, str(root_candidate))

        resolved = _decision_with_status(current_decision, status="resolved", resolved=True)
        updated, latest = await hub.transition_session_decision(
            session_id,
            expected_id=decision_id,
            expected_status="pending",
            next_decision=resolved,
        )
        normalized_latest = _normalize_ui_decision(latest, session_id=session_id)
        if not updated:
            return _decision_mismatch_response(
                expected_id=decision_id,
                actual_decision=normalized_latest,
            )

        payload_override: dict[str, JSONValue] = {
            "content": source_request_raw.get("content"),
            "force_canvas": source_request_raw.get("force_canvas"),
            "lane": _normalize_message_lane(source_request_raw.get("lane")),
            "attachments": source_request_raw.get("attachments"),
        }
        resumed_response = await handle_ui_chat_send(
            request,
            payload_override=payload_override,
            bypass_root_gate=True,
        )
        resume_data: dict[str, JSONValue] | None = None
        resume_ok = resumed_response.status < 400
        resume_error: str | None = None
        parsed_resume_payload: dict[str, JSONValue] = {}
        body_raw = resumed_response.text
        if isinstance(body_raw, str) and body_raw.strip():
            try:
                parsed = json.loads(body_raw)
                if isinstance(parsed, dict):
                    parsed_resume_payload = {
                        str(key): _normalize_json_value(value) for key, value in parsed.items()
                    }
            except json.JSONDecodeError:
                parsed_resume_payload = {}
        if resume_ok:
            resume_data = {
                "status_code": resumed_response.status,
                "trace_id": parsed_resume_payload.get("trace_id"),
                "output": parsed_resume_payload.get("output"),
                "auto_state": parsed_resume_payload.get("auto_state"),
                "decision": parsed_resume_payload.get("decision"),
            }
        else:
            error_raw = parsed_resume_payload.get("error")
            if isinstance(error_raw, dict):
                message_raw = error_raw.get("message")
                if isinstance(message_raw, str) and message_raw.strip():
                    resume_error = message_raw.strip()
            if resume_error is None:
                resume_error = f"chat.run_root failed: {resumed_response.status}"

        workflow = await hub.get_session_workflow(session_id)
        latest_decision = _normalize_ui_decision(
            await hub.get_session_decision(session_id),
            session_id=session_id,
        )
        return json_response(
            {
                "ok": True,
                "decision": latest_decision,
                "status": "resolved",
                "resume_started": True,
                "already_resolved": True,
                "resume": {
                    "ok": resume_ok,
                    "source_endpoint": "chat.run_root",
                    "data": resume_data,
                    "error": resume_error,
                },
                "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
                "active_task": _normalize_task_payload(workflow.get("active_task")),
                "auto_state": _normalize_auto_state(workflow.get("auto_state")),
            }
        )

    if choice == "reject" and decision_type in {"tool_approval", "plan_execute"}:
        context_raw = current_decision.get("context")
        context = context_raw if isinstance(context_raw, dict) else {}
        source_endpoint_raw = context.get("source_endpoint")
        source_endpoint = source_endpoint_raw if isinstance(source_endpoint_raw, str) else ""
        resume_payload_raw = context.get("resume_payload")
        resume_payload = resume_payload_raw if isinstance(resume_payload_raw, dict) else {}
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
        resume_payload_response: dict[str, JSONValue] | None = None
        if source_endpoint == "auto.run":
            run_id_raw = resume_payload.get("run_id")
            run_id = (
                run_id_raw.strip() if isinstance(run_id_raw, str) and run_id_raw.strip() else ""
            )
            if run_id:
                try:
                    agent = await _resolve_agent(request)
                except ModelNotAllowedError:
                    agent = None
                if agent is not None:
                    async with agent_lock:
                        await _apply_agent_runtime_state(
                            agent=agent,
                            hub=hub,
                            session_id=session_id,
                        )
                        cancel_method = getattr(agent, "cancel_auto_run", None)
                        if callable(cancel_method):
                            cancelled = cancel_method(run_id, reason="rejected_by_user")
                            cancelled_state = _normalize_auto_state(cancelled)
                            if cancelled_state is not None:
                                await hub.set_session_workflow(
                                    session_id,
                                    auto_state=cancelled_state,
                                )
                                await hub.publish(
                                    session_id,
                                    {
                                        "type": "auto.progress",
                                        "payload": {
                                            "session_id": session_id,
                                            "auto_state": cancelled_state,
                                        },
                                    },
                                )
                                resume_payload_response = {
                                    "ok": True,
                                    "source_endpoint": source_endpoint,
                                    "data": {
                                        "run_id": run_id,
                                        "status": "cancelled",
                                    },
                                }
        workflow = await hub.get_session_workflow(session_id)
        return json_response(
            {
                "ok": True,
                "decision": normalized,
                "status": resolved_status,
                "resume_started": False,
                "already_resolved": resolved_status in {"resolved", "rejected"},
                "resume": resume_payload_response,
                "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
                "active_task": _normalize_task_payload(workflow.get("active_task")),
                "auto_state": _normalize_auto_state(workflow.get("auto_state")),
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
    elif decision_type == "tool_approval":
        resume = await _resolve_tool_decision()
    else:
        resume = await _resolve_agent_decision()

    if decision_type == "tool_approval":
        await _set_current_plan_step_status(
            hub=hub,
            session_id=session_id,
            status="done" if resume.get("ok") is True else "blocked",
        )
    validation_error_codes = {
        "resume_payload_missing",
        "plan_revision_mismatch",
        "plan_not_approved",
        "switch_to_act_required",
        "task_already_running",
        "mode_not_plan",
        "mode_not_act",
        "plan_state_conflict",
    }
    resume_error_code: str | None = None
    if isinstance(resume, dict):
        error_raw = resume.get("error")
        if isinstance(error_raw, str) and error_raw.strip():
            resume_error_code = error_raw.strip()
    if (
        isinstance(resume, dict)
        and resume.get("ok") is False
        and resume_error_code in validation_error_codes
    ):
        pending = _decision_with_status(executing, status="pending")
        await hub.transition_session_decision(
            session_id,
            expected_id=decision_id,
            expected_status="executing",
            next_decision=pending,
        )
        message_by_code = {
            "resume_payload_missing": "Невозможно продолжить: отсутствует resume_payload.",
            "plan_revision_mismatch": "План изменился, требуется обновление состояния.",
            "plan_not_approved": "Сначала approve план.",
            "switch_to_act_required": "Для выполнения нужно перейти в act.",
            "task_already_running": "План уже выполняется.",
            "mode_not_plan": "Операция доступна только в plan-режиме.",
            "mode_not_act": "Операция доступна только в act-режиме.",
            "plan_state_conflict": "Состояние плана изменилось, повторите действие.",
        }
        details_raw = resume.get("details")
        details = details_raw if isinstance(details_raw, dict) else None
        return error_response(
            status=409,
            message=message_by_code.get(
                resume_error_code,
                "Невозможно продолжить: конфликт состояния.",
            ),
            error_type="invalid_request_error",
            code=resume_error_code,
            details=details,
        )
    resume_started = True
    if isinstance(resume, dict) and "resume_started" in resume:
        resume_started = resume.get("resume_started") is True
        resume = {key: value for key, value in resume.items() if key != "resume_started"}

    resolved = _decision_with_status(executing, status="resolved", resolved=True)
    updated_resolved, latest_resolved = await hub.transition_session_decision(
        session_id,
        expected_id=decision_id,
        expected_status="executing",
        next_decision=resolved,
    )
    normalized_resolved = _normalize_ui_decision(latest_resolved, session_id=session_id)
    if not updated_resolved:
        if decision_type == "agent_decision":
            workflow = await hub.get_session_workflow(session_id)
            latest_decision = _normalize_ui_decision(
                await hub.get_session_decision(session_id),
                session_id=session_id,
            )
            return json_response(
                {
                    "ok": True,
                    "decision": latest_decision,
                    "status": "resolved",
                    "resume_started": resume_started,
                    "already_resolved": True,
                    "resume": resume,
                    "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
                    "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
                    "active_task": _normalize_task_payload(workflow.get("active_task")),
                    "auto_state": _normalize_auto_state(workflow.get("auto_state")),
                }
            )
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
            "resume_started": resume_started,
            "already_resolved": True,
            "resume": resume,
            "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
            "active_task": _normalize_task_payload(workflow.get("active_task")),
            "auto_state": _normalize_auto_state(workflow.get("auto_state")),
        }
    )
