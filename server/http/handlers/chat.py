from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import cast

from aiohttp import web

from core.response_preferences import ResponsePreferences
from core.task_run_storage import RunAlreadyAdmittedError, RunConflictError, TaskRunStore
from server.agent_provider import ScopedAgentProvider
from server.http.common.auth import _request_principal_id
from server.http.common.chat_payload import (
    _project_agent_response,
)
from server.http.common.idempotency import fingerprint_json_payload, normalize_idempotency_key
from server.http.common.proxy_model import PUBLIC_PROXY_MODEL_ID
from server.http.common.responses import error_response, json_response
from server.http.common.runtime_contract import (
    AgentProtocol,
    RuntimeModelResolverProtocol,
    RuntimeModelStateProtocol,
    _require_custom_provider_owner,
)
from server.http_api import (
    TOOL_PIPELINE_ENABLED,
    WORKSPACE_ROOT,
    _agent_lock_for_request,
    _agent_scope,
    _build_effective_tools_state,
    _load_effective_session_security,
    _model_not_selected_response,
    _parse_chat_request,
    _resolve_provider_api_key,
    _workspace_root_for_session,
)
from shared.models import JSONValue
from shared.task_run import RunScope, RunState, TaskRun, TransitionAuthority
from shared.tool_capture import CaptureBinding
from tools.workspace_tools import workspace_root_context


async def handle_chat_completions(request: web.Request) -> web.Response:
    session_store = request.app["session_store"]
    hub = request.app["ui_hub"]

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

    parsed, error_text = _parse_chat_request(payload)
    if parsed is None:
        return error_response(
            status=400,
            message=error_text,
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if parsed.model != PUBLIC_PROXY_MODEL_ID:
        return error_response(
            status=404,
            message=(
                f"Модель '{parsed.model}' не найдена. "
                f"Публичный proxy model id: '{PUBLIC_PROXY_MODEL_ID}'."
            ),
            error_type="invalid_request_error",
            code="model_not_found",
            details={
                "requested_model": parsed.model,
                "available_models": [PUBLIC_PROXY_MODEL_ID],
            },
        )

    header_session_id = request.headers.get("X-Slavik-Session", "").strip() or None
    declared_session_ids = {
        value
        for value in (header_session_id, parsed.session_id, parsed.runtime_session_id)
        if value is not None
    }
    if len(declared_session_ids) > 1:
        return error_response(
            status=400,
            message="Все переданные session_id должны совпадать.",
            error_type="invalid_request_error",
            code="session_scope_mismatch",
        )
    explicit_session_id = next(iter(declared_session_ids), None)
    scope = _agent_scope(request, explicit_session_id)
    session_id = scope.session_id
    runtime_session_id = session_id
    runtime_mode = parsed.runtime_mode

    if runtime_mode in {"plan", "act"}:
        return error_response(
            status=400,
            message="runtime_mode=plan|act для /v1 пока не поддерживается.",
            error_type="invalid_request_error",
            code="invalid_request_error",
            details={
                "next_steps": [
                    "Используйте UI workflow endpoints: /ui/api/plan/* и /ui/api/mode.",
                    "Для /v1 доступен Ask по умолчанию или явный runtime_mode=auto.",
                ],
            },
        )

    if parsed.stream:
        return error_response(
            status=400,
            message="Streaming is not supported.",
            error_type="not_supported",
            code="streaming_not_supported",
        )

    if parsed.tool_calling_present and not TOOL_PIPELINE_ENABLED:
        return error_response(
            status=400,
            message="Tool calling is not supported.",
            error_type="not_supported",
            code="tool_calling_not_supported",
        )

    if explicit_session_id is not None:
        principal_id = _request_principal_id(request)
        if principal_id is None:
            return error_response(
                status=401,
                message="Unauthorized.",
                error_type="invalid_request_error",
                code="unauthorized",
            )
        try:
            await hub.get_or_create_session(explicit_session_id, principal_id=principal_id)
        except PermissionError:
            return error_response(
                status=403,
                message="Session access forbidden.",
                error_type="invalid_request_error",
                code="session_forbidden",
            )

    agent_lock = _agent_lock_for_request(request, explicit_session_id)

    client_key = normalize_idempotency_key(request.headers.get("Idempotency-Key"))
    durable_key = hashlib.sha256(
        json.dumps(["/v1/chat/completions", session_id, client_key or uuid.uuid4().hex]).encode()
    ).hexdigest()
    run_store: TaskRunStore = request.app["task_run_store"]
    run_scope = RunScope(scope.principal_id, session_id)
    command_lane = parsed.messages[-1].content.strip().startswith("/")
    admitted_run: TaskRun | None = None
    trace_id: str | None = None
    decision_type: str | None = None
    mwv_report: dict[str, JSONValue] | None = None
    try:
        async with agent_lock:
            if command_lane and client_key is not None:
                if run_store.find_by_request(run_scope, durable_key) is not None:
                    return error_response(
                        status=409,
                        message="Ключ уже связан с принятым run.",
                        error_type="invalid_request_error",
                        code="idempotency_key_reused",
                    )
            approved_categories = await session_store.get_categories(scope)
            resolver = cast(RuntimeModelResolverProtocol, request.app["runtime_model_resolver"])
            execution_config = await resolver.resolve_main(explicit_session_id)
            if execution_config is None:
                return _model_not_selected_response()
            try:
                _require_custom_provider_owner(request, execution_config.provider, execution_config)
            except PermissionError:
                return error_response(
                    status=403,
                    message="Owner access required for custom provider.",
                    error_type="invalid_request_error",
                    code="owner_required",
                )
            if explicit_session_id is not None and await hub.get_session_model(session_id) is None:
                await hub.set_session_model(
                    session_id, execution_config.provider, execution_config.model
                )
                model_state = cast(RuntimeModelStateProtocol, request.app["runtime_model_state"])
                await model_state.set_session_override(session_id, execution_config)
            execution_key = _resolve_provider_api_key(execution_config.provider)

            async def request_context() -> tuple[
                tuple[dict[str, bool], dict[str, JSONValue]], Path
            ]:
                if explicit_session_id is None:
                    tools = _build_effective_tools_state(session_override=None)
                    tools["safe_mode"] = True
                    return (tools, {"profile": "sandbox"}), WORKSPACE_ROOT
                return (
                    await _load_effective_session_security(hub=hub, session_id=session_id),
                    await _workspace_root_for_session(hub, session_id),
                )

            security_snapshot, accepted_root = await request_context()

            async def context_current() -> bool:
                return (
                    await resolver.resolve_main(explicit_session_id) == execution_config
                    and await session_store.get_categories(scope) == approved_categories
                    and await request_context() == (security_snapshot, accepted_root)
                    and _resolve_provider_api_key(execution_config.provider) == execution_key
                )

            if not command_lane:
                if not await context_current():
                    return error_response(
                        status=409,
                        message="Контекст запроса изменился до dispatch.",
                        error_type="invalid_request_error",
                        code="task_run_context_changed",
                    )
                binding: dict[str, JSONValue] = {
                    "request": payload,
                    "runtime_mode": runtime_mode,
                    "execution_config": asdict(execution_config),
                    "credential_hash": hashlib.sha256((execution_key or "").encode()).hexdigest(),
                    "approved_categories": sorted(approved_categories),
                    "security": [security_snapshot[0], security_snapshot[1]],
                    "workspace_root": str(accepted_root),
                }
                try:
                    admitted_run = run_store.admit(
                        run_scope,
                        request_key=durable_key,
                        goal=parsed.messages[-1].content
                        if parsed.messages[-1].content.strip()
                        else json.dumps(
                            payload["messages"][-1], ensure_ascii=False, separators=(",", ":")
                        ),
                        mode=runtime_mode,
                        request_fingerprint=fingerprint_json_payload(binding),
                        require_new=True,
                    )
                except RunAlreadyAdmittedError:
                    return error_response(
                        status=409,
                        message="Принятый run нельзя повторно выполнить без continuation.",
                        error_type="invalid_request_error",
                        code="task_run_continuation_unavailable",
                    )
                except RunConflictError:
                    return error_response(
                        status=409,
                        message="Ключ связан с другим принятым запросом.",
                        error_type="invalid_request_error",
                        code="idempotency_key_reused",
                    )
                admitted_run = run_store.transition(
                    run_scope,
                    admitted_run.task_run_id,
                    expected_version=admitted_run.version,
                    state=RunState.RUNNING,
                    authority=TransitionAuthority.RUNTIME,
                    reason_code="api_runtime_started",
                )
            provider = cast(ScopedAgentProvider[AgentProtocol], request.app["agent_provider"])
            agent = await provider.get_for_current_task(scope, execution_config)
            previous_trace_id = agent.last_chat_interaction_id
            previous_decision_packet = getattr(agent, "last_decision_packet", None)
            agent.response_preferences = ResponsePreferences()
            agent.set_session_context(session_id, approved_categories)
            agent.reconfigure_models(execution_config, main_api_key=execution_key, persist=False)
            agent.apply_runtime_tools_enabled(security_snapshot[0])
            agent.apply_runtime_workspace_root(str(accepted_root))
            agent.set_runtime_state(
                mode=runtime_mode,
                active_plan=None,
                active_task=None,
                auto_state=None,
                enforce_plan_guard=False,
            )
            if not await context_current():
                return error_response(
                    status=409,
                    message="Контекст запроса изменился до dispatch.",
                    error_type="invalid_request_error",
                    code="task_run_context_changed",
                )
            with workspace_root_context(accepted_root):
                if runtime_mode == "auto" and admitted_run is not None:
                    agent_response = agent.respond(
                        parsed.messages, auto_capture=CaptureBinding(run_store, admitted_run)
                    )
                else:
                    agent_response = agent.respond(parsed.messages)
            if admitted_run is not None:
                if getattr(agent, "last_approval_source_endpoint", None) == "chat.tool_continue":
                    pending = agent.last_approval_resume_payload
                    if isinstance(pending, dict) and isinstance(
                        pending.get("continuation_id"), str
                    ):
                        agent.bind_chat_approval_run(
                            str(pending["continuation_id"]),
                            admitted_run,
                            approved_categories=approved_categories,
                            security_fingerprint=fingerprint_json_payload(
                                {"tools": security_snapshot[0], "policy": security_snapshot[1]}
                            ),
                        )
                if agent_response.runtime_result is not None:
                    run_store.submit_result(
                        run_scope,
                        admitted_run.task_run_id,
                        expected_version=admitted_run.version,
                        report=agent_response.producer_report(),
                        reason_code="api_result_submitted",
                    )
            response_text, mwv_report = _project_agent_response(agent_response)
            trace_id = agent.last_chat_interaction_id
            if trace_id == previous_trace_id:
                trace_id = None
            decision_packet = getattr(agent, "last_decision_packet", None)
            decision_type_raw = (
                getattr(decision_packet, "decision_type", None)
                if decision_packet is not previous_decision_packet
                else None
            )
            decision_type = (
                decision_type_raw.strip()
                if isinstance(decision_type_raw, str) and decision_type_raw.strip()
                else None
            )
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=500,
            message=f"Agent error: {exc}",
            error_type="internal_error",
            code="agent_error",
            trace_id=trace_id,
        )

    if parsed.sampling_warnings:
        agent.tracer.log(
            "request_warning",
            "Ignored unknown sampling params",
            {"fields": parsed.sampling_warnings},
        )

    approval_request = getattr(agent, "last_approval_request", None)
    if approval_request is not None:
        if trace_id is None:
            trace_id = agent.last_chat_interaction_id
        return error_response(
            status=400,
            message="Approval required.",
            error_type="tool_error",
            code="approval_required",
            trace_id=trace_id,
            details={
                "category": approval_request.category,
                "required_categories": approval_request.required_categories,
                "session_id": approval_request.session_id,
                "prompt": {
                    "what": approval_request.prompt.what,
                    "why": approval_request.prompt.why,
                    "risk": approval_request.prompt.risk,
                    "changes": approval_request.prompt.changes,
                },
                "blocked_reason": "approval_required",
            },
        )

    if trace_id is None and decision_type != "memory_save":
        return error_response(
            status=500,
            message="Trace ID was not generated.",
            error_type="internal_error",
            code="trace_id_missing",
        )

    session_approved = bool(approved_categories)
    safe_mode = bool(agent.tools_enabled.get("safe_mode", False))

    slavik_meta: dict[str, JSONValue] = {
        "trace_id": trace_id,
        "session_id": session_id,
        "session_approved": session_approved,
        "safe_mode": safe_mode,
    }
    if mwv_report is not None:
        slavik_meta["mwv_report"] = mwv_report
    slavik_meta["runtime_mode"] = runtime_mode
    slavik_meta["runtime_session_id"] = runtime_session_id

    response_payload: dict[str, JSONValue] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": PUBLIC_PROXY_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": response_text},
            }
        ],
        "slavik_meta": slavik_meta,
    }
    return json_response(response_payload)
