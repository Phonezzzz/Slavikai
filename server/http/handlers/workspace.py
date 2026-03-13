from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from config.tools_config import DEFAULT_TOOLS_STATE
from core.approval_policy import ApprovalCategory, ApprovalRequired
from llm.brain_factory import create_brain
from llm.types import ModelConfig
from server import http_api as api
from server.http.common.responses import error_response, json_response
from server.http.common.runtime_contract import AgentProtocol
from server.http_api import (
    POLICY_PROFILES,
    UI_SESSION_HEADER,
    _build_ui_approval_decision,
    _decision_workflow_context,
    _index_workspace_root,
    _load_effective_session_security,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_policy_profile,
    _normalize_task_payload,
    _normalize_tools_state_payload,
    _normalize_ui_decision,
    _resolve_ui_session_id_for_principal,
    _resolve_workspace_root_candidate,
    _session_forbidden_response,
    _set_current_plan_step_status,
    _utc_now_iso,
    _workspace_git_diff,
    _workspace_root_for_session,
)
from server.ui_hub import UIHub
from shared.models import JSONValue, LLMMessage, ToolRequest, ToolResult
from tools.workspace_tools import (
    ApplyPatchTool,
    ListFilesTool,
    ReadFileTool,
)
from tools.workspace_tools import (
    set_workspace_root as set_runtime_workspace_root,
)

EDITOR_ACTIONS = frozenset({"fix", "improve", "review", "explain"})
EDITOR_PATCH_ACTIONS = frozenset({"fix", "improve"})
EDITOR_PROVIDER = "inception"
EDITOR_MODEL = "mercury-edit"


async def handle_ui_redirect(request: web.Request) -> web.StreamResponse:
    raise web.HTTPFound("/ui/")


async def handle_ui_index(request: web.Request) -> web.FileResponse:
    dist_path: Path = request.app["ui_dist_path"]
    index_path = dist_path / "index.html"
    return web.FileResponse(path=index_path)


async def handle_workspace_index(request: web.Request) -> web.FileResponse:
    return await handle_ui_index(request)


async def handle_ui_workspace_root_select(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_store = request.app["session_store"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    if _normalize_mode_value(workflow.get("mode"), default="ask") == "plan":
        return error_response(
            status=409,
            message="PLAN_READ_ONLY_BLOCK: plan-режим допускает только read-only действия.",
            error_type="invalid_request_error",
            code="PLAN_READ_ONLY_BLOCK",
        )
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
    path_raw = payload.get("root_path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        return error_response(
            status=400,
            message="root_path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    _, effective_policy = await _load_effective_session_security(hub=hub, session_id=session_id)
    profile_raw = effective_policy.get("profile")
    profile = _normalize_policy_profile(profile_raw)
    try:
        target_root = _resolve_workspace_root_candidate(path_raw.strip(), policy_profile=profile)
    except ValueError as exc:
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    current_root = await _workspace_root_for_session(hub, session_id)
    if current_root == target_root:
        response = json_response(
            {
                "session_id": session_id,
                "root_path": str(current_root),
                "applied": True,
            }
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response

    require_approval = profile == "yolo"
    required_category = "FS_OUTSIDE_WORKSPACE"
    approved = await session_store.get_categories(session_id)
    if not require_approval or required_category in approved:
        await hub.set_workspace_root(session_id, str(target_root))
        response = json_response(
            {
                "session_id": session_id,
                "root_path": str(target_root),
                "applied": True,
            }
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response

    approval_request: dict[str, JSONValue] = {
        "category": required_category,
        "required_categories": [required_category],
        "tool": "workspace_root_select",
        "prompt": {
            "what": "Сменить Workspace Root",
            "why": "Требуется подтверждение смены рабочей директории.",
            "risk": "Доступ к другой директории проекта.",
            "changes": [str(target_root)],
        },
        "details": {
            "root_path": str(target_root),
            "policy_profile": profile,
        },
    }
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    active_plan = _normalize_plan_payload(workflow.get("active_plan"))
    active_task = _normalize_task_payload(workflow.get("active_task"))
    decision = _build_ui_approval_decision(
        approval_request=approval_request,
        session_id=session_id,
        source_endpoint="workspace.root_select",
        resume_payload={
            "root_path": str(target_root),
            "session_id": session_id,
        },
        workflow_context=_decision_workflow_context(
            mode=mode,
            active_plan=active_plan,
            active_task=active_task,
        ),
    )
    await _set_current_plan_step_status(
        hub=hub,
        session_id=session_id,
        status="waiting_approval",
    )
    await hub.set_session_decision(session_id, decision)
    response = json_response(
        {
            "session_id": session_id,
            "decision": _normalize_ui_decision(decision, session_id=session_id),
        },
        status=202,
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_index_run(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    if _normalize_mode_value(workflow.get("mode"), default="ask") == "plan":
        return error_response(
            status=409,
            message="PLAN_READ_ONLY_BLOCK: plan-режим допускает только read-only действия.",
            error_type="invalid_request_error",
            code="PLAN_READ_ONLY_BLOCK",
        )
    root_path = await _workspace_root_for_session(hub, session_id)
    try:
        stats = _index_workspace_root(root_path)
    except ValueError as exc:
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="workspace_index_failed",
        )
    except RuntimeError as exc:
        return error_response(
            status=500,
            message=str(exc),
            error_type="internal_error",
            code="workspace_index_failed",
        )
    response = json_response(
        {
            "session_id": session_id,
            **stats,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_git_diff(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    root_path = await _workspace_root_for_session(hub, session_id)
    diff, error = _workspace_git_diff(root_path)
    response = json_response(
        {
            "session_id": session_id,
            "root_path": str(root_path),
            "diff": diff,
            "error": error,
            "ok": error is None,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def _resolve_workspace_session(
    request: web.Request,
) -> tuple[AgentProtocol | None, str | None, set[ApprovalCategory], web.Response | None]:
    hub: UIHub = request.app["ui_hub"]
    session_store = request.app["session_store"]
    agent = await api._resolve_agent(request)
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return agent, None, set(), session_error
    if session_id is None:
        return agent, None, set(), _session_forbidden_response()
    approved_categories = await session_store.get_categories(session_id)
    return agent, session_id, approved_categories, None


async def _resolve_workspace_session_id(
    request: web.Request,
) -> tuple[str | None, web.Response | None]:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return None, session_error
    if session_id is None:
        return None, _session_forbidden_response()
    return session_id, None


def _normalize_bool_query_param(raw: str | None) -> bool:
    if raw is None:
        return False
    normalized = raw.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _workspace_content_version(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _editor_model_config() -> ModelConfig:
    return ModelConfig(
        provider="inception",
        model=EDITOR_MODEL,
        temperature=0.1,
        reasoning_effort="instant",
        reasoning_summary=True,
        reasoning_summary_wait=False,
        diffusing=True,
        max_tokens=3_000,
    )


def _strip_optional_json_fence(raw_text: str) -> str:
    stripped = raw_text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if not lines:
        return stripped
    first_line = lines[0].strip().lower()
    if first_line not in {"```", "```json"}:
        return stripped
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_editor_model_payload(raw_text: str) -> dict[str, object]:
    try:
        parsed = json.loads(_strip_optional_json_fence(raw_text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Editor model вернула не-JSON payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Editor model должна вернуть JSON object.")
    return parsed


async def _read_workspace_file_content(
    *,
    request: web.Request,
    session_id: str,
    path_value: str,
) -> tuple[str, str] | web.Response:
    current_read = await _call_workspace_read_tool(
        request=request,
        session_id=session_id,
        tool_name="workspace_read",
        args={"path": path_value},
    )
    if not current_read.ok:
        return error_response(
            status=400,
            message=current_read.error or "Не удалось прочитать файл.",
            error_type="invalid_request_error",
            code="workspace_read_failed",
        )
    current_content_raw = current_read.data.get("output")
    current_content = current_content_raw if isinstance(current_content_raw, str) else ""
    return current_content, _workspace_content_version(current_content)


def _workspace_version_conflict_response(
    *,
    path_value: str,
    expected_version: str,
    current_version: str,
) -> web.Response:
    return error_response(
        status=409,
        message="Конфликт версии: файл был изменён в другой вкладке или процессе.",
        error_type="conflict_error",
        code="workspace_version_conflict",
        details={
            "path": path_value,
            "expected_version": expected_version,
            "current_version": current_version,
        },
    )


def _normalize_editor_tabs(raw_tabs: object) -> list[str]:
    if not isinstance(raw_tabs, list):
        return []
    normalized: list[str] = []
    for item in raw_tabs:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())
    return normalized[:20]


def _editor_context_from_attachments(
    raw_attachments: object,
) -> tuple[list[str], str, str]:
    if not isinstance(raw_attachments, list):
        return [], "", ""
    open_tabs: list[str] = []
    git_diff = ""
    terminal_output = ""
    for item in raw_attachments:
        if not isinstance(item, dict):
            continue
        name_raw = item.get("name")
        content_raw = item.get("content")
        if not isinstance(name_raw, str) or not isinstance(content_raw, str):
            continue
        name = name_raw.strip().lower()
        if name == "open-tabs.json":
            try:
                parsed = json.loads(content_raw)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(parsed, dict):
                continue
            tabs_raw = parsed.get("open_files")
            if isinstance(tabs_raw, list):
                for entry in tabs_raw:
                    if not isinstance(entry, dict):
                        continue
                    path_raw = entry.get("path")
                    if isinstance(path_raw, str) and path_raw.strip() and path_raw.strip() not in open_tabs:
                        open_tabs.append(path_raw.strip())
            active_file_raw = parsed.get("active_file")
            if (
                isinstance(active_file_raw, str)
                and active_file_raw.strip()
                and active_file_raw.strip() not in open_tabs
            ):
                open_tabs.insert(0, active_file_raw.strip())
        elif name == "git-diff.patch" and not git_diff:
            git_diff = content_raw
        elif name == "terminal-last.txt" and not terminal_output:
            terminal_output = content_raw
    return open_tabs[:20], git_diff, terminal_output


def _editor_action_messages(
    *,
    action: str,
    path_value: str,
    selection_text: str,
    editor_content: str,
    saved_content: str,
    git_diff: str,
    terminal_output: str,
    open_tabs: list[str],
) -> list[LLMMessage]:
    if action in EDITOR_PATCH_ACTIONS:
        contract = (
            'Верни только JSON-объект вида {"summary":"...","patch":"@@ ... @@"}. '
            "patch должен быть single-file unified diff без заголовков diff --git/---/+++."
        )
    else:
        contract = 'Верни только JSON-объект вида {"message":"..."} без markdown fences.'
    scope = "selection" if selection_text else "full_file"
    tabs_block = ", ".join(open_tabs) if open_tabs else "(none)"
    selection_block = selection_text if selection_text else "(no explicit selection)"
    user_prompt = (
        f"action={action}\n"
        f"path={path_value}\n"
        f"scope={scope}\n"
        f"open_tabs={tabs_block}\n\n"
        f"[selection]\n{selection_block}\n\n"
        f"[saved_content]\n{saved_content}\n\n"
        f"[editor_content]\n{editor_content}\n\n"
        f"[git_diff]\n{git_diff or '(empty)'}\n\n"
        f"[terminal]\n{terminal_output or '(empty)'}\n"
    )
    return [
        LLMMessage(
            role="system",
            content=(
                "Ты backend editor assistant для code editor. "
                "Никакого prose вне JSON. "
                f"{contract}"
            ),
        ),
        LLMMessage(role="user", content=user_prompt),
    ]


def _validate_editor_patch_preview(
    *,
    workspace_root: Path,
    path_value: str,
    patch_text: str,
) -> ToolResult:
    set_runtime_workspace_root(workspace_root)
    try:
        return ApplyPatchTool().handle(
            ToolRequest(
                name="workspace_patch",
                args={"path": path_value, "patch": patch_text, "dry_run": True},
            )
        )
    finally:
        set_runtime_workspace_root(None)


def _workspace_read_tool(tool_name: str) -> ListFilesTool | ReadFileTool:
    if tool_name == "workspace_list":
        return ListFilesTool()
    if tool_name == "workspace_read":
        return ReadFileTool()
    raise ValueError(f"unsupported workspace read tool: {tool_name}")


async def _call_workspace_read_tool(
    *,
    request: web.Request,
    session_id: str,
    tool_name: str,
    args: dict[str, JSONValue] | None = None,
) -> ToolResult:
    hub: UIHub = request.app["ui_hub"]
    session_root = await _workspace_root_for_session(hub, session_id)
    set_runtime_workspace_root(session_root)
    try:
        tool = _workspace_read_tool(tool_name)
        return tool.handle(ToolRequest(name=tool_name, args=args or {}))
    finally:
        set_runtime_workspace_root(None)


async def _call_workspace_tool(
    *,
    request: web.Request,
    agent: AgentProtocol,
    session_id: str,
    approved_categories: set[ApprovalCategory],
    tool_name: str,
    args: dict[str, JSONValue] | None = None,
    raw_input: str,
) -> ToolResult | web.Response:
    hub: UIHub = request.app["ui_hub"]
    agent_lock = request.app["agent_lock"]
    session_root = await _workspace_root_for_session(hub, session_id)
    async with agent_lock:
        set_runtime_workspace_root(session_root)
        try:
            try:
                await api._apply_agent_runtime_state(agent=agent, hub=hub, session_id=session_id)
                agent.set_session_context(session_id, approved_categories)
            except Exception:  # noqa: BLE001
                api.logger.debug(
                    "Failed to set session context for workspace tool call",
                    exc_info=True,
                )
            try:
                return agent.call_tool(
                    tool_name,
                    args=args or {},
                    raw_input=raw_input,
                )
            except ApprovalRequired as exc:
                approval_payload = api._serialize_approval_request(exc.request)
                workflow = await hub.get_session_workflow(session_id)
                mode = _normalize_mode_value(workflow.get("mode"), default="ask")
                active_plan = _normalize_plan_payload(workflow.get("active_plan"))
                active_task = _normalize_task_payload(workflow.get("active_task"))
                approval_request_payload: dict[str, JSONValue] = approval_payload or {}
                decision = _build_ui_approval_decision(
                    approval_request=approval_request_payload,
                    session_id=session_id,
                    source_endpoint="workspace.tool",
                    resume_payload={
                        "tool_name": tool_name,
                        "args": dict(args or {}),
                        "raw_input": raw_input,
                        "session_id": session_id,
                    },
                    workflow_context=_decision_workflow_context(
                        mode=mode,
                        active_plan=active_plan,
                        active_task=active_task,
                    ),
                )
                await _set_current_plan_step_status(
                    hub=hub,
                    session_id=session_id,
                    status="waiting_approval",
                )
                await hub.set_session_decision(session_id, decision)
                normalized_decision = _normalize_ui_decision(decision, session_id=session_id)
                response = json_response(
                    {
                        "session_id": session_id,
                        "decision": normalized_decision,
                        "approval_request": approval_payload,
                    },
                    status=202,
                )
                response.headers[UI_SESSION_HEADER] = session_id
                return response
        finally:
            set_runtime_workspace_root(None)


async def handle_ui_workspace_tree(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_workspace_session_id(request)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    path_raw = request.query.get("path")
    recursive_raw = request.query.get("recursive")
    max_depth_raw = request.query.get("max_depth")
    args: dict[str, JSONValue] = {"recursive": _normalize_bool_query_param(recursive_raw)}
    max_depth = 12
    if isinstance(path_raw, str) and path_raw.strip():
        args["path"] = path_raw.strip()
    if isinstance(max_depth_raw, str) and max_depth_raw.strip():
        try:
            max_depth = int(max_depth_raw.strip())
        except ValueError:
            return error_response(
                status=400,
                message="max_depth должен быть int.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
    if max_depth < 0:
        max_depth = 0
    if max_depth > 12:
        max_depth = 12
    args["max_depth"] = max_depth
    tool_result = await _call_workspace_read_tool(
        request=request,
        session_id=session_id,
        tool_name="workspace_list",
        args=args,
    )
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось получить структуру workspace.",
            error_type="invalid_request_error",
            code="workspace_list_failed",
        )
    tree_raw = tool_result.data.get("tree")
    tree: list[JSONValue] = tree_raw if isinstance(tree_raw, list) else []
    listed_path_raw = tool_result.data.get("path")
    listed_path = listed_path_raw if isinstance(listed_path_raw, str) else ""
    tree_meta_raw = tool_result.data.get("tree_meta")
    tree_meta: dict[str, JSONValue]
    if isinstance(tree_meta_raw, dict):
        tree_meta = {
            "returned_entries": int(tree_meta_raw.get("returned_entries", 0)),
            "returned_dirs": int(tree_meta_raw.get("returned_dirs", 0)),
            "returned_files": int(tree_meta_raw.get("returned_files", 0)),
            "truncated": bool(tree_meta_raw.get("truncated", False)),
            "truncated_reasons": tree_meta_raw.get("truncated_reasons")
            if isinstance(tree_meta_raw.get("truncated_reasons"), list)
            else [],
            "max_depth_applied": int(tree_meta_raw.get("max_depth_applied", 0)),
            "max_entries": int(tree_meta_raw.get("max_entries", 0)),
            "max_dirs": int(tree_meta_raw.get("max_dirs", 0)),
            "max_files": int(tree_meta_raw.get("max_files", 0)),
            "max_children_per_dir": int(tree_meta_raw.get("max_children_per_dir", 0)),
        }
    else:
        tree_meta = {
            "returned_entries": 0,
            "returned_dirs": 0,
            "returned_files": 0,
            "truncated": False,
            "truncated_reasons": [],
            "max_depth_applied": max_depth if bool(args.get("recursive")) else 0,
            "max_entries": 0,
            "max_dirs": 0,
            "max_files": 0,
            "max_children_per_dir": 0,
        }
    if tree_meta.get("truncated") is True:
        api.logger.info(
            "Workspace tree truncated",
            extra={
                "path": "/ui/api/workspace/tree",
                "session_id": session_id,
                "tree_path": listed_path,
                "tree_meta": tree_meta,
            },
        )
    root_path = await _workspace_root_for_session(hub, session_id)
    response = json_response(
        {
            "session_id": session_id,
            "tree": tree,
            "root_path": str(root_path),
            "path": listed_path,
            "tree_meta": tree_meta,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_file_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_workspace_session_id(request)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    path_raw = request.query.get("path", "")
    path_value = path_raw.strip()
    if not path_value:
        return error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    tool_result = await _call_workspace_read_tool(
        request=request,
        session_id=session_id,
        tool_name="workspace_read",
        args={"path": path_value},
    )
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось прочитать файл.",
            error_type="invalid_request_error",
            code="workspace_read_failed",
        )
    content_raw = tool_result.data.get("output")
    if not isinstance(content_raw, str):
        content_raw = ""
    version = _workspace_content_version(content_raw)
    response = json_response(
        {
            "session_id": session_id,
            "path": path_value,
            "content": content_raw,
            "version": version,
            "root_path": str(await _workspace_root_for_session(hub, session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_file_put(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    try:
        agent, session_id, approved_categories, session_error = await _resolve_workspace_session(
            request
        )
    except ModelNotAllowedError as exc:
        return api._model_not_allowed_response(exc.model_id)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    if agent is None:
        return api._model_not_selected_response()
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
    path_raw = payload.get("path")
    content_raw = payload.get("content")
    expected_version_raw = payload.get("version")
    if not isinstance(path_raw, str) or not path_raw.strip():
        return error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(content_raw, str):
        return error_response(
            status=400,
            message="content должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    expected_version: str | None = None
    if expected_version_raw is not None:
        if not isinstance(expected_version_raw, str) or not expected_version_raw.strip():
            return error_response(
                status=400,
                message="version должен быть непустой строкой.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        expected_version = expected_version_raw.strip()
    path_value = path_raw.strip()
    if expected_version is not None:
        current_read = await _call_workspace_read_tool(
            request=request,
            session_id=session_id,
            tool_name="workspace_read",
            args={"path": path_value},
        )
        if not current_read.ok:
            return error_response(
                status=409,
                message="Конфликт версии: файл был изменён или удалён перед сохранением.",
                error_type="conflict_error",
                code="workspace_version_conflict",
                details={"path": path_value, "expected_version": expected_version},
            )
        current_content_raw = current_read.data.get("output")
        current_content = current_content_raw if isinstance(current_content_raw, str) else ""
        current_version = _workspace_content_version(current_content)
        if current_version != expected_version:
            return error_response(
                status=409,
                message="Конфликт версии: файл был изменён в другой вкладке или процессе.",
                error_type="conflict_error",
                code="workspace_version_conflict",
                details={
                    "path": path_value,
                    "expected_version": expected_version,
                    "current_version": current_version,
                },
            )
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_write",
        args={"path": path_value, "content": content_raw},
        raw_input=f"ui:workspace_write {path_value}",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось сохранить файл.",
            error_type="invalid_request_error",
            code="workspace_write_failed",
        )
    response = json_response(
        {
            "session_id": session_id,
            "path": path_value,
            "saved": True,
            "version": _workspace_content_version(content_raw),
            "root_path": str(await _workspace_root_for_session(hub, session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_file_create(request: web.Request) -> web.Response:
    try:
        agent, session_id, approved_categories, session_error = await _resolve_workspace_session(
            request
        )
    except ModelNotAllowedError as exc:
        return api._model_not_allowed_response(exc.model_id)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    if agent is None:
        return api._model_not_selected_response()
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
    path_raw = payload.get("path")
    content_raw = payload.get("content", "")
    overwrite_raw = payload.get("overwrite", False)
    if not isinstance(path_raw, str) or not path_raw.strip():
        return error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(content_raw, str):
        return error_response(
            status=400,
            message="content должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(overwrite_raw, bool):
        return error_response(
            status=400,
            message="overwrite должен быть bool.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    path_value = path_raw.strip()
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_create",
        args={"path": path_value, "content": content_raw, "overwrite": overwrite_raw},
        raw_input=f"ui:workspace_create {path_value}",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось создать файл.",
            error_type="invalid_request_error",
            code="workspace_create_failed",
        )
    response = json_response(
        {
            "session_id": session_id,
            "path": path_value,
            "created": True,
            "version": _workspace_content_version(content_raw),
            "root_path": str(
                await _workspace_root_for_session(hub=request.app["ui_hub"], session_id=session_id)
            ),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_file_rename(request: web.Request) -> web.Response:
    try:
        agent, session_id, approved_categories, session_error = await _resolve_workspace_session(
            request
        )
    except ModelNotAllowedError as exc:
        return api._model_not_allowed_response(exc.model_id)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    if agent is None:
        return api._model_not_selected_response()
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
    old_path_raw = payload.get("old_path")
    new_path_raw = payload.get("new_path")
    if not isinstance(old_path_raw, str) or not old_path_raw.strip():
        return error_response(
            status=400,
            message="old_path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(new_path_raw, str) or not new_path_raw.strip():
        return error_response(
            status=400,
            message="new_path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    old_path = old_path_raw.strip()
    new_path = new_path_raw.strip()
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_rename",
        args={"old_path": old_path, "new_path": new_path},
        raw_input=f"ui:workspace_rename {old_path} -> {new_path}",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось переименовать файл.",
            error_type="invalid_request_error",
            code="workspace_rename_failed",
        )
    response = json_response(
        {
            "session_id": session_id,
            "old_path": old_path,
            "new_path": new_path,
            "renamed": True,
            "root_path": str(await _workspace_root_for_session(request.app["ui_hub"], session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_file_move(request: web.Request) -> web.Response:
    try:
        agent, session_id, approved_categories, session_error = await _resolve_workspace_session(
            request
        )
    except ModelNotAllowedError as exc:
        return api._model_not_allowed_response(exc.model_id)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    if agent is None:
        return api._model_not_selected_response()
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
    from_path_raw = payload.get("from_path")
    to_path_raw = payload.get("to_path")
    if not isinstance(from_path_raw, str) or not from_path_raw.strip():
        return error_response(
            status=400,
            message="from_path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(to_path_raw, str) or not to_path_raw.strip():
        return error_response(
            status=400,
            message="to_path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    from_path = from_path_raw.strip()
    to_path = to_path_raw.strip()
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_move",
        args={"from_path": from_path, "to_path": to_path},
        raw_input=f"ui:workspace_move {from_path} -> {to_path}",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось переместить файл.",
            error_type="invalid_request_error",
            code="workspace_move_failed",
        )
    response = json_response(
        {
            "session_id": session_id,
            "from_path": from_path,
            "to_path": to_path,
            "moved": True,
            "root_path": str(await _workspace_root_for_session(request.app["ui_hub"], session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_file_delete(request: web.Request) -> web.Response:
    try:
        agent, session_id, approved_categories, session_error = await _resolve_workspace_session(
            request
        )
    except ModelNotAllowedError as exc:
        return api._model_not_allowed_response(exc.model_id)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    if agent is None:
        return api._model_not_selected_response()
    payload: dict[str, object] = {}
    if request.can_read_body:
        try:
            payload_raw = await request.json()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                status=400,
                message=f"Некорректный JSON: {exc}",
                error_type="invalid_request_error",
                code="invalid_json",
            )
        if payload_raw is not None and not isinstance(payload_raw, dict):
            return error_response(
                status=400,
                message="JSON должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_json",
            )
        if isinstance(payload_raw, dict):
            payload = payload_raw

    path_candidate = request.query.get("path")
    if not isinstance(path_candidate, str) or not path_candidate.strip():
        payload_path = payload.get("path")
        path_candidate = payload_path if isinstance(payload_path, str) else ""
    if not path_candidate.strip():
        return error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    recursive_raw = payload.get("recursive", False)
    if not isinstance(recursive_raw, bool):
        return error_response(
            status=400,
            message="recursive должен быть bool.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    path_value = path_candidate.strip()
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_delete",
        args={"path": path_value, "recursive": recursive_raw},
        raw_input=f"ui:workspace_delete {path_value}",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось удалить путь.",
            error_type="invalid_request_error",
            code="workspace_delete_failed",
        )
    response = json_response(
        {
            "session_id": session_id,
            "path": path_value,
            "deleted": True,
            "root_path": str(await _workspace_root_for_session(request.app["ui_hub"], session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_editor_action(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_workspace_session_id(request)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
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

    action_raw = payload.get("action")
    path_raw = payload.get("path")
    file_content_raw = payload.get("file_content")
    saved_content_raw = payload.get("saved_content")
    selection_text_raw = payload.get("selection_text")
    version_raw = payload.get("version")
    git_diff_raw = payload.get("git_diff")
    terminal_output_raw = payload.get("terminal_output")

    if not isinstance(action_raw, str) or action_raw.strip().lower() not in EDITOR_ACTIONS:
        return error_response(
            status=400,
            message="action должен быть одним из: fix, improve, review, explain.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(path_raw, str) or not path_raw.strip():
        return error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(file_content_raw, str):
        return error_response(
            status=400,
            message="file_content должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if saved_content_raw is not None and not isinstance(saved_content_raw, str):
        return error_response(
            status=400,
            message="saved_content должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if selection_text_raw is not None and not isinstance(selection_text_raw, str):
        return error_response(
            status=400,
            message="selection_text должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if version_raw is not None and (not isinstance(version_raw, str) or not version_raw.strip()):
        return error_response(
            status=400,
            message="version должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    action = action_raw.strip().lower()
    path_value = path_raw.strip()
    editor_content = file_content_raw
    selection_text = selection_text_raw or ""
    attachment_tabs, attachment_git_diff, attachment_terminal = _editor_context_from_attachments(
        payload.get("attachments")
    )
    git_diff = git_diff_raw if isinstance(git_diff_raw, str) else attachment_git_diff
    terminal_output = (
        terminal_output_raw if isinstance(terminal_output_raw, str) else attachment_terminal
    )
    open_tabs = _normalize_editor_tabs(payload.get("open_tabs"))
    for tab in attachment_tabs:
        if tab not in open_tabs:
            open_tabs.append(tab)

    current_file_or_error = await _read_workspace_file_content(
        request=request,
        session_id=session_id,
        path_value=path_value,
    )
    if isinstance(current_file_or_error, web.Response):
        return current_file_or_error
    current_saved_content, current_version = current_file_or_error
    expected_version = version_raw.strip() if isinstance(version_raw, str) else current_version
    if expected_version != current_version:
        return _workspace_version_conflict_response(
            path_value=path_value,
            expected_version=expected_version,
            current_version=current_version,
        )
    if isinstance(saved_content_raw, str) and saved_content_raw != current_saved_content:
        return _workspace_version_conflict_response(
            path_value=path_value,
            expected_version=_workspace_content_version(saved_content_raw),
            current_version=current_version,
        )

    api_key = api._resolve_provider_api_key(EDITOR_PROVIDER)
    if not api_key:
        return error_response(
            status=409,
            message="Editor actions недоступны: не задан INCEPTION_API_KEY для inception/mercury-edit.",
            error_type="configuration_error",
            code="editor_model_unavailable",
            details={"provider": EDITOR_PROVIDER, "model": EDITOR_MODEL},
        )
    available_models, fetch_error = api._fetch_provider_models(EDITOR_PROVIDER)
    if fetch_error is not None or EDITOR_MODEL not in available_models:
        return error_response(
            status=409,
            message="Editor actions недоступны: mercury-edit не найден среди доступных моделей Inception.",
            error_type="configuration_error",
            code="editor_model_unavailable",
            details={
                "provider": EDITOR_PROVIDER,
                "model": EDITOR_MODEL,
                "fetch_error": fetch_error,
            },
        )

    model_config = _editor_model_config()
    brain = create_brain(model_config, api_key=api_key)
    messages = _editor_action_messages(
        action=action,
        path_value=path_value,
        selection_text=selection_text,
        editor_content=editor_content,
        saved_content=current_saved_content,
        git_diff=git_diff,
        terminal_output=terminal_output,
        open_tabs=open_tabs,
    )
    try:
        result = await asyncio.to_thread(brain.generate, messages, model_config)
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=502,
            message=f"Не удалось выполнить editor action: {exc}",
            error_type="provider_error",
            code="editor_action_failed",
        )
    try:
        model_payload = _parse_editor_model_payload(result.text)
    except ValueError as exc:
        return error_response(
            status=502,
            message=str(exc),
            error_type="provider_error",
            code="editor_action_invalid_response",
        )

    root_path = await _workspace_root_for_session(hub, session_id)
    if action in EDITOR_PATCH_ACTIONS:
        summary_raw = model_payload.get("summary")
        patch_raw = model_payload.get("patch")
        if not isinstance(summary_raw, str) or not summary_raw.strip():
            return error_response(
                status=502,
                message="Editor model не вернула summary для patch preview.",
                error_type="provider_error",
                code="editor_action_invalid_response",
            )
        if not isinstance(patch_raw, str) or not patch_raw.strip():
            return error_response(
                status=502,
                message="Editor model не вернула patch для patch preview.",
                error_type="provider_error",
                code="editor_action_invalid_response",
            )
        dry_run_result = _validate_editor_patch_preview(
            workspace_root=root_path,
            path_value=path_value,
            patch_text=patch_raw,
        )
        if not dry_run_result.ok:
            return error_response(
                status=502,
                message=dry_run_result.error or "Patch preview не прошёл dry-run validation.",
                error_type="provider_error",
                code="editor_patch_invalid",
            )
        patched_content_raw = dry_run_result.data.get("content")
        patched_content = patched_content_raw if isinstance(patched_content_raw, str) else None
        response = json_response(
            {
                "session_id": session_id,
                "action": action,
                "mode": "patch_preview",
                "summary": summary_raw.strip(),
                "patch": patch_raw,
                "base_version": current_version,
                "target_path": path_value,
                "apply_available": True,
                "patched_content": patched_content,
                "root_path": str(root_path),
            }
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response

    message_raw = model_payload.get("message")
    if not isinstance(message_raw, str) or not message_raw.strip():
        return error_response(
            status=502,
            message="Editor model не вернула message для read-only action.",
            error_type="provider_error",
            code="editor_action_invalid_response",
        )
    response = json_response(
        {
            "session_id": session_id,
            "action": action,
            "mode": "read_only",
            "message": message_raw.strip(),
            "base_version": current_version,
            "target_path": path_value,
            "root_path": str(root_path),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_patch(request: web.Request) -> web.Response:
    try:
        agent, session_id, approved_categories, session_error = await _resolve_workspace_session(
            request
        )
    except ModelNotAllowedError as exc:
        return api._model_not_allowed_response(exc.model_id)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    if agent is None:
        return api._model_not_selected_response()
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
    path_raw = payload.get("path")
    patch_raw = payload.get("patch")
    dry_run_raw = payload.get("dry_run", False)
    version_raw = payload.get("version")
    if not isinstance(path_raw, str) or not path_raw.strip():
        return error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(patch_raw, str) or not patch_raw.strip():
        return error_response(
            status=400,
            message="patch обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(dry_run_raw, bool):
        return error_response(
            status=400,
            message="dry_run должен быть bool.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if version_raw is not None and (not isinstance(version_raw, str) or not version_raw.strip()):
        return error_response(
            status=400,
            message="version должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    path_value = path_raw.strip()
    if isinstance(version_raw, str):
        current_file_or_error = await _read_workspace_file_content(
            request=request,
            session_id=session_id,
            path_value=path_value,
        )
        if isinstance(current_file_or_error, web.Response):
            return error_response(
                status=409,
                message="Конфликт версии: файл был изменён или удалён перед применением patch.",
                error_type="conflict_error",
                code="workspace_version_conflict",
                details={"path": path_value, "expected_version": version_raw.strip()},
            )
        _, current_version = current_file_or_error
        expected_version = version_raw.strip()
        if current_version != expected_version:
            return _workspace_version_conflict_response(
                path_value=path_value,
                expected_version=expected_version,
                current_version=current_version,
            )
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_patch",
        args={"path": path_value, "patch": patch_raw, "dry_run": dry_run_raw},
        raw_input=f"ui:workspace_patch {path_value}",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось применить patch.",
            error_type="invalid_request_error",
            code="workspace_patch_failed",
        )
    output_raw = tool_result.data.get("output")
    output = output_raw if isinstance(output_raw, str) else ""
    response = json_response(
        {
            "session_id": session_id,
            "path": path_value,
            "dry_run": dry_run_raw,
            "output": output,
            "root_path": str(await _workspace_root_for_session(request.app["ui_hub"], session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_run(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    try:
        agent, session_id, approved_categories, session_error = await _resolve_workspace_session(
            request
        )
    except ModelNotAllowedError as exc:
        return api._model_not_allowed_response(exc.model_id)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    if agent is None:
        return api._model_not_selected_response()
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
    path_raw = payload.get("path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        return error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    path_value = path_raw.strip()
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_run",
        args={"path": path_value},
        raw_input=f"ui:workspace_run {path_value}",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось выполнить файл.",
            error_type="invalid_request_error",
            code="workspace_run_failed",
        )
    stdout_raw = tool_result.data.get("output")
    stderr_raw = tool_result.data.get("stderr")
    exit_code_raw = tool_result.data.get("exit_code")
    stdout = stdout_raw if isinstance(stdout_raw, str) else ""
    stderr = stderr_raw if isinstance(stderr_raw, str) else ""
    exit_code = exit_code_raw if isinstance(exit_code_raw, int) else 0
    response = json_response(
        {
            "session_id": session_id,
            "path": path_value,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "root_path": str(await _workspace_root_for_session(hub, session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_terminal_run(request: web.Request) -> web.Response:
    try:
        agent, session_id, approved_categories, session_error = await _resolve_workspace_session(
            request
        )
    except ModelNotAllowedError as exc:
        return api._model_not_allowed_response(exc.model_id)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    if agent is None:
        return api._model_not_selected_response()
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
    command_raw = payload.get("command")
    cwd_mode_raw = payload.get("cwd_mode", "session_root")
    if not isinstance(command_raw, str) or not command_raw.strip():
        return error_response(
            status=400,
            message="command обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(cwd_mode_raw, str) or cwd_mode_raw.strip() not in {"session_root", "sandbox"}:
        return error_response(
            status=400,
            message="cwd_mode должен быть session_root|sandbox.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    command = command_raw.strip()
    cwd_mode = cwd_mode_raw.strip()
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_terminal_run",
        args={"command": command, "cwd_mode": cwd_mode},
        raw_input=f"ui:workspace_terminal_run {command}",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return error_response(
            status=400,
            message=tool_result.error or "Не удалось выполнить команду терминала.",
            error_type="invalid_request_error",
            code="workspace_terminal_run_failed",
        )
    stdout_raw = tool_result.data.get("output")
    stderr_raw = tool_result.data.get("stderr")
    exit_code_raw = tool_result.data.get("exit_code")
    cwd_raw = tool_result.data.get("cwd")
    mode_raw = tool_result.data.get("cwd_mode")
    response = json_response(
        {
            "session_id": session_id,
            "stdout": stdout_raw if isinstance(stdout_raw, str) else "",
            "stderr": stderr_raw if isinstance(stderr_raw, str) else "",
            "exit_code": exit_code_raw if isinstance(exit_code_raw, int) else 0,
            "cwd": cwd_raw if isinstance(cwd_raw, str) else "",
            "cwd_mode": mode_raw if isinstance(mode_raw, str) else cwd_mode,
            "root_path": str(await _workspace_root_for_session(request.app["ui_hub"], session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_root_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    root_path = await _workspace_root_for_session(hub, session_id)
    _, policy = await _load_effective_session_security(hub=hub, session_id=session_id)
    response = json_response(
        {
            "session_id": session_id,
            "root_path": str(root_path),
            "policy": policy,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_security_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workspace_root = await _workspace_root_for_session(hub, session_id)
    effective_tools, effective_policy = await _load_effective_session_security(
        hub=hub,
        session_id=session_id,
    )
    response = json_response(
        {
            "session_id": session_id,
            "tools_state": effective_tools,
            "policy": effective_policy,
            "workspace_root": str(workspace_root),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_security_post(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
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
    invalid_top_keys = sorted(
        {str(key).strip() for key in payload if str(key).strip() not in {"tools", "policy"}}
    )
    if invalid_top_keys:
        return error_response(
            status=400,
            message=f"Неподдерживаемые поля: {', '.join(invalid_top_keys)}.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    if "policy" in payload:
        policy_raw = payload.get("policy")
        if not isinstance(policy_raw, dict):
            return error_response(
                status=400,
                message="policy должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        profile: str | None = None
        yolo_armed: bool | None = None
        yolo_armed_at: str | None = None

        if "profile" in policy_raw:
            profile_raw = policy_raw.get("profile")
            if (
                not isinstance(profile_raw, str)
                or profile_raw.strip().lower() not in POLICY_PROFILES
            ):
                return error_response(
                    status=400,
                    message="policy.profile должен быть sandbox|index|yolo.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            profile = profile_raw.strip().lower()

        if "yolo_armed" in policy_raw:
            yolo_armed_raw = policy_raw.get("yolo_armed")
            if not isinstance(yolo_armed_raw, bool):
                return error_response(
                    status=400,
                    message="policy.yolo_armed должен быть bool.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            yolo_armed = yolo_armed_raw

        if profile == "yolo" or yolo_armed is True:
            confirm_raw = policy_raw.get("yolo_confirm")
            confirm_text_raw = policy_raw.get("yolo_confirm_text")
            confirm_ok = (
                confirm_raw is True
                and isinstance(confirm_text_raw, str)
                and confirm_text_raw.strip().upper() == "YOLO"
            )
            if not confirm_ok:
                return error_response(
                    status=400,
                    message=(
                        "Для включения YOLO требуется подтверждение "
                        "(yolo_confirm=true, yolo_confirm_text='YOLO')."
                    ),
                    error_type="invalid_request_error",
                    code="yolo_confirmation_required",
                )
            if yolo_armed is None:
                yolo_armed = True

        workspace_root = await _workspace_root_for_session(hub, session_id)
        current_policy = await hub.get_session_policy(session_id)
        current_profile = _normalize_policy_profile(current_policy.get("profile"))
        target_profile = profile or current_profile
        try:
            _resolve_workspace_root_candidate(
                str(workspace_root),
                policy_profile=target_profile,
            )
        except ValueError as exc:
            return error_response(
                status=409,
                message=str(exc),
                error_type="invalid_request_error",
                code="policy_root_incompatible",
                details={
                    "session_id": session_id,
                    "workspace_root": str(workspace_root),
                    "policy_profile": target_profile,
                },
            )

        if yolo_armed is True:
            yolo_armed_at = _utc_now_iso()
        if yolo_armed is False:
            yolo_armed_at = None
        await hub.set_session_policy(
            session_id,
            profile=profile,
            yolo_armed=yolo_armed,
            yolo_armed_at=yolo_armed_at,
        )

    if "tools" in payload:
        tools_raw = payload.get("tools")
        if not isinstance(tools_raw, dict):
            return error_response(
                status=400,
                message="tools должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        state_raw = tools_raw.get("state")
        if not isinstance(state_raw, dict):
            return error_response(
                status=400,
                message="tools.state должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        unknown_tools = sorted(
            {str(key).strip() for key in state_raw if str(key).strip() not in DEFAULT_TOOLS_STATE}
        )
        if unknown_tools:
            return error_response(
                status=400,
                message=f"Неизвестные tools ключи: {', '.join(unknown_tools)}.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        normalized_tools = _normalize_tools_state_payload(state_raw)
        if len(normalized_tools) != len(state_raw):
            return error_response(
                status=400,
                message="tools.state должен быть bool map.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        await hub.set_session_tools_state(
            session_id,
            tools_state=normalized_tools,
            merge=True,
        )

    effective_tools, effective_policy = await _load_effective_session_security(
        hub=hub,
        session_id=session_id,
    )
    workspace_root = await _workspace_root_for_session(hub, session_id)
    response = json_response(
        {
            "session_id": session_id,
            "tools_state": effective_tools,
            "policy": effective_policy,
            "workspace_root": str(workspace_root),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response
