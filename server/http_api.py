from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

import requests
from aiohttp import web

from config.http_server_config import (
    HttpAuthConfig,
    HttpServerConfig,
)
from config.memory_config import MemoryConfig, load_memory_config, save_memory_config
from config.tools_config import (
    DEFAULT_TOOLS_STATE,
    ToolsConfig,
    load_tools_config,
    save_tools_config,
)
from config.ui_embeddings_settings import UIEmbeddingsSettings
from core.approval_policy import (
    ALL_CATEGORIES,
    ApprovalCategory,
    ApprovalRequest,
)
from server.http.common import (
    chat_payload as _chat_payload,
)
from server.http.common import (
    chat_request as _chat_request,
)
from server.http.common import (
    decision_flow,
    github_import,
    plan_edit,
    runtime_contract,
    session_transfer,
    workflow_state,
    workspace_index,
    workspace_paths,
    workspace_runtime,
)
from server.http.common import (
    streaming as _streaming,
)
from server.http.common import (
    trace_views as _trace_views,
)
from server.http.common import (
    ui_artifacts as _ui_artifacts,
)
from server.http.common import (
    ui_settings as _ui_settings,
)
from server.http.common import (
    workflow_runtime as _workflow_runtime,
)
from server.http.common.auth import (
    AUTH_PROTECTED_PREFIXES as AUTH_PROTECTED_PREFIXES,
)
from server.http.common.auth import (
    _ensure_session_owned as _ensure_session_owned,
)
from server.http.common.auth import (
    _extract_bearer_token as _extract_bearer_token,
)
from server.http.common.auth import (
    _extract_ui_session_id as _extract_ui_session_id,
)
from server.http.common.auth import (
    _is_auth_protected_path as _is_auth_protected_path,
)
from server.http.common.auth import (
    _principal_id_from_token as _principal_id_from_token,
)
from server.http.common.auth import (
    _request_principal_id as _request_principal_id,
)
from server.http.common.auth import (
    _require_admin_bearer as _require_admin_bearer,
)
from server.http.common.auth import (
    _resolve_request_principal_id as _resolve_request_principal_id,
)
from server.http.common.auth import (
    _resolve_ui_session_id_for_principal as _resolve_ui_session_id_for_principal,
)
from server.http.common.auth import (
    _session_forbidden_response as _session_forbidden_response,
)
from server.http.common.auth import (
    auth_gate_middleware as auth_gate_middleware,
)
from server.http.common.responses import (
    error_response as _error_response,
)
from server.ui_hub import UIHub
from server.ui_session_storage import PersistedSession, UISessionStorage
from shared.models import JSONValue
from tools.workspace_tools import (
    WORKSPACE_ROOT as DEFAULT_WORKSPACE_ROOT,
)

ChatRequest = _chat_request.ChatRequest
_is_sampling_key = _chat_request._is_sampling_key
_validate_messages = _chat_request._validate_messages
_parse_chat_request = _chat_request._parse_chat_request
_extract_session_id = _chat_payload._extract_session_id
_ui_messages_to_llm = _chat_payload._ui_messages_to_llm
_parse_ui_chat_attachments = _chat_payload._parse_ui_chat_attachments
_extract_decision_payload = _chat_payload._extract_decision_payload
_split_response_and_report = _chat_payload._split_response_and_report
_normalize_trace_id = _chat_payload._normalize_trace_id
_request_likely_web_intent = _chat_payload._request_likely_web_intent
CHAT_STREAM_CHUNK_SIZE = _streaming.CHAT_STREAM_CHUNK_SIZE
CHAT_STREAM_WARMUP_CHARS = _streaming.CHAT_STREAM_WARMUP_CHARS
_split_chat_stream_chunks = _streaming._split_chat_stream_chunks
_publish_chat_stream_start = _streaming._publish_chat_stream_start
_publish_chat_stream_delta = _streaming._publish_chat_stream_delta
_publish_chat_stream_done = _streaming._publish_chat_stream_done
_publish_chat_stream_from_text = _streaming._publish_chat_stream_from_text
_split_canvas_stream_chunks = _streaming._split_canvas_stream_chunks
_publish_canvas_stream_base = _streaming._publish_canvas_stream
TraceGroup = _trace_views.TraceGroup
TRACE_LOG = _trace_views.TRACE_LOG
TOOL_CALLS_LOG = _trace_views.TOOL_CALLS_LOG
_parse_trace_log = _trace_views._parse_trace_log
_build_trace_groups = _trace_views._build_trace_groups
_extract_interaction_id = _trace_views._extract_interaction_id
_last_event_timestamp = _trace_views._last_event_timestamp
_parse_timestamp = _trace_views._parse_timestamp
_filter_tool_calls = _trace_views._filter_tool_calls
_normalize_logged_path = _trace_views._normalize_logged_path
_extract_paths_from_tool_call = _trace_views._extract_paths_from_tool_call
_extract_files_from_tool_calls = _trace_views._extract_files_from_tool_calls
_serialize_trace_events = session_transfer._serialize_trace_events
_parse_imported_message = session_transfer._parse_imported_message
_utc_iso = session_transfer._utc_iso
SessionApprovalStore = runtime_contract.SessionApprovalStore
TracerProtocol = runtime_contract.TracerProtocol
AgentProtocol = runtime_contract.AgentProtocol
RequestsModuleProtocol = runtime_contract.RequestsModuleProtocol
_model_not_selected_response = runtime_contract._model_not_selected_response
_model_not_allowed_response = runtime_contract._model_not_allowed_response
_resolve_agent = runtime_contract._resolve_agent

TOOL_PIPELINE_ENABLED: Final[bool] = False
_CATEGORY_MAP: Final[dict[str, ApprovalCategory]] = {item: item for item in ALL_CATEGORIES}

logger = logging.getLogger("SlavikAI.HttpAPI")

_parse_github_import_args = github_import.parse_github_import_args
_build_github_import_approval_request = github_import.build_github_import_approval_request
_resolve_github_target = github_import.resolve_github_target
_clone_github_repository = github_import.clone_github_repository
_index_imported_project = github_import.index_imported_project

UI_SESSION_HEADER: Final[str] = "X-Slavik-Session"
SUPPORTED_MODEL_PROVIDERS: Final[set[str]] = _ui_settings.SUPPORTED_MODEL_PROVIDERS
API_KEY_SETTINGS_PROVIDERS: Final[set[str]] = _ui_settings.API_KEY_SETTINGS_PROVIDERS
PROVIDER_API_KEY_ENV: Final[dict[str, str]] = _ui_settings.PROVIDER_API_KEY_ENV
XAI_MODELS_ENDPOINT: Final[str] = _ui_settings.XAI_MODELS_ENDPOINT
OPENROUTER_MODELS_ENDPOINT: Final[str] = _ui_settings.OPENROUTER_MODELS_ENDPOINT
OPENAI_STT_ENDPOINT: Final[str] = _ui_settings.OPENAI_STT_ENDPOINT
MODEL_FETCH_TIMEOUT: Final[int] = _ui_settings.MODEL_FETCH_TIMEOUT
UI_PROJECT_COMMANDS: Final[set[str]] = {"find", "index", "github_import"}
UI_SETTINGS_PATH: Final[Path] = _ui_settings.UI_SETTINGS_PATH
DEFAULT_UI_TONE: Final[str] = _ui_settings.DEFAULT_UI_TONE
INDEX_ENABLED_ENV: Final[str] = _ui_settings.INDEX_ENABLED_ENV
DEFAULT_LONG_PASTE_TO_FILE_ENABLED: Final[bool] = _ui_settings.DEFAULT_LONG_PASTE_TO_FILE_ENABLED
DEFAULT_LONG_PASTE_THRESHOLD_CHARS: Final[int] = _ui_settings.DEFAULT_LONG_PASTE_THRESHOLD_CHARS
MIN_LONG_PASTE_THRESHOLD_CHARS: Final[int] = _ui_settings.MIN_LONG_PASTE_THRESHOLD_CHARS
MAX_LONG_PASTE_THRESHOLD_CHARS: Final[int] = _ui_settings.MAX_LONG_PASTE_THRESHOLD_CHARS
UI_GITHUB_REQUIRED_CATEGORIES: Final[list[ApprovalCategory]] = ["NETWORK_RISK", "EXEC_ARBITRARY"]
CANVAS_LINE_THRESHOLD: Final[int] = _ui_artifacts.CANVAS_LINE_THRESHOLD
CANVAS_CHAR_THRESHOLD: Final[int] = _ui_artifacts.CANVAS_CHAR_THRESHOLD
CANVAS_CODE_LINE_THRESHOLD: Final[int] = _ui_artifacts.CANVAS_CODE_LINE_THRESHOLD
CANVAS_DOCUMENT_LINE_THRESHOLD: Final[int] = _ui_artifacts.CANVAS_DOCUMENT_LINE_THRESHOLD
CANVAS_STATUS_CHARS_STEP: Final[int] = _ui_artifacts.CANVAS_STATUS_CHARS_STEP
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT: Final[Path] = DEFAULT_WORKSPACE_ROOT
MAX_DOWNLOAD_BYTES: Final[int] = 5_000_000
MAX_CONTENT_CHARS: Final[int] = 120_000
MAX_TOTAL_PAYLOAD_CHARS: Final[int] = 220_000
MAX_STT_AUDIO_BYTES: Final[int] = 10_000_000
WORKSPACE_INDEX_IGNORED_DIRS: Final[set[str]] = {
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".cache",
}
WORKSPACE_INDEX_ALLOWED_EXTENSIONS: Final[set[str]] = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".sql",
    ".sh",
}
WORKSPACE_INDEX_MAX_FILE_BYTES: Final[int] = 1_000_000
POLICY_PROFILES: Final[set[str]] = _ui_settings.POLICY_PROFILES
DEFAULT_POLICY_PROFILE: Final[str] = _ui_settings.DEFAULT_POLICY_PROFILE
UI_DECISION_KINDS: Final[set[str]] = {"approval", "decision"}
UI_DECISION_STATUSES: Final[set[str]] = {
    "pending",
    "approved",
    "rejected",
    "executing",
    "resolved",
}
UI_DECISION_RESPONSES: Final[set[str]] = {
    "approve_once",
    "approve_session",
    "edit_and_approve",
    "edit_plan",
    "reject",
}
UI_DECISION_EDITABLE_FIELDS: Final[set[str]] = {"details", "args", "query", "branch", "repo_url"}
UI_SETTINGS_USER_ALLOWED_TOP_LEVEL_KEYS: Final[set[str]] = (
    _ui_settings.UI_SETTINGS_USER_ALLOWED_TOP_LEVEL_KEYS
)
UI_SETTINGS_CONTROL_TOP_LEVEL_KEYS: Final[set[str]] = (
    _ui_settings.UI_SETTINGS_CONTROL_TOP_LEVEL_KEYS
)
UI_SETTINGS_ADMIN_ALLOWED_TOP_LEVEL_KEYS: Final[set[str]] = {"tools", "policy"}
SESSION_MODES: Final[set[str]] = {"ask", "plan", "act"}
PLAN_STATUSES: Final[set[str]] = {
    "draft",
    "approved",
    "running",
    "completed",
    "failed",
    "cancelled",
}
PLAN_STEP_STATUSES: Final[set[str]] = {
    "todo",
    "doing",
    "waiting_approval",
    "blocked",
    "done",
    "failed",
}
TASK_STATUSES: Final[set[str]] = {"running", "completed", "failed", "cancelled"}
PLAN_AUDIT_MAX_READ_FILES: Final[int] = 15
PLAN_AUDIT_MAX_TOTAL_BYTES: Final[int] = 300_000
PLAN_AUDIT_MAX_SEARCH_CALLS: Final[int] = 10
PLAN_AUDIT_TIMEOUT_SECONDS: Final[int] = 20
PLAN_MAX_STEPS: Final[int] = 50
PLAN_MAX_PAYLOAD_BYTES: Final[int] = 64 * 1024
PLAN_MAX_TEXT_FIELD_CHARS: Final[int] = 4000
_ASK_ACTION_INTENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(^/|\\b(сделай|измени|запусти|выполни|удали|установи|run|execute|modify|edit|delete|install)\\b)",
    re.IGNORECASE,
)


def _trace_log_path() -> Path:
    return TRACE_LOG


def _tool_calls_for_trace_id(trace_id: str) -> list[dict[str, JSONValue]] | None:
    return _trace_views._tool_calls_for_trace_id(
        trace_id,
        trace_log=TRACE_LOG,
        tool_calls_log=TOOL_CALLS_LOG,
    )


_normalize_provider = _ui_settings._normalize_provider
_build_model_config = _ui_settings._build_model_config
_closest_model_suggestion = _ui_settings._closest_model_suggestion
_local_models_endpoint = _ui_settings._local_models_endpoint
_provider_models_endpoint = _ui_settings._provider_models_endpoint
_parse_models_payload = _ui_settings._parse_models_payload
_load_provider_env_api_key = _ui_settings._load_provider_env_api_key


def _provider_auth_headers(provider: str) -> tuple[dict[str, str], str | None]:
    return _ui_settings._provider_auth_headers(provider, ui_settings_path=UI_SETTINGS_PATH)


def _fetch_provider_models(provider: str) -> tuple[list[str], str | None]:
    return _ui_settings._fetch_provider_models(provider, ui_settings_path=UI_SETTINGS_PATH)


def _requests_module() -> RequestsModuleProtocol:
    return requests


def _load_memory_config_runtime() -> MemoryConfig:
    return load_memory_config()


def _save_memory_config_runtime(config: MemoryConfig) -> None:
    save_memory_config(config)


def _resolve_workspace_file(path_raw: str) -> Path:
    return workspace_runtime._resolve_workspace_file(
        path_raw,
        workspace_root=WORKSPACE_ROOT,
        max_download_bytes=MAX_DOWNLOAD_BYTES,
    )


def _artifact_file_payload(
    artifact: dict[str, JSONValue],
) -> tuple[str, str, str]:
    return workspace_runtime._artifact_file_payload(
        artifact,
        sanitize_download_filename_fn=_sanitize_download_filename,
        artifact_mime_from_ext_fn=_artifact_mime_from_ext,
    )


def _serialize_approval_request(
    approval_request: ApprovalRequest | None,
) -> dict[str, JSONValue] | None:
    if approval_request is None:
        return None
    return {
        "category": approval_request.category,
        "required_categories": list(approval_request.required_categories),
        "tool": approval_request.tool,
        "details": dict(approval_request.details),
        "session_id": approval_request.session_id,
        "prompt": {
            "what": approval_request.prompt.what,
            "why": approval_request.prompt.why,
            "risk": approval_request.prompt.risk,
            "changes": list(approval_request.prompt.changes),
        },
    }


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_mode_value(value: object, *, default: str = "ask") -> str:
    return workflow_state.normalize_mode_value(
        value,
        default=default,
        session_modes=SESSION_MODES,
    )


def _normalize_string_list(value: object) -> list[str]:
    return workflow_state.normalize_string_list(value)


def _normalize_plan_step(step: object) -> dict[str, JSONValue] | None:
    return workflow_state.normalize_plan_step(
        step,
        plan_step_statuses=PLAN_STEP_STATUSES,
    )


def _plan_hash_payload(plan: dict[str, JSONValue]) -> str:
    return workflow_state.plan_hash_payload(plan)


def _normalize_plan_payload(raw: object) -> dict[str, JSONValue] | None:
    return workflow_state.normalize_plan_payload(
        raw,
        plan_statuses=PLAN_STATUSES,
        plan_step_statuses=PLAN_STEP_STATUSES,
        utc_now_iso=_utc_now_iso,
        normalize_plan_step_fn=_normalize_plan_step,
        normalize_string_list_fn=_normalize_string_list,
        normalize_json_value_fn=_normalize_json_value,
        plan_hash_payload_fn=_plan_hash_payload,
    )


def _normalize_task_payload(raw: object) -> dict[str, JSONValue] | None:
    return workflow_state.normalize_task_payload(
        raw,
        task_statuses=TASK_STATUSES,
        utc_now_iso=_utc_now_iso,
    )


def _normalize_tools_state_payload(raw: object) -> dict[str, bool]:
    return _workflow_runtime.normalize_tools_state_payload(
        raw,
        default_tools_state_keys=set(DEFAULT_TOOLS_STATE.keys()),
    )


def _build_effective_tools_state(
    *,
    session_override: dict[str, bool] | None,
) -> dict[str, bool]:
    return _workflow_runtime.build_effective_tools_state(
        session_override=session_override,
        load_tools_state_fn=_load_tools_state,
        default_tools_state_keys=set(DEFAULT_TOOLS_STATE.keys()),
    )


async def _load_effective_session_security(
    *,
    hub: UIHub,
    session_id: str,
) -> tuple[dict[str, bool], dict[str, JSONValue]]:
    return await _workflow_runtime.load_effective_session_security(
        hub=hub,
        session_id=session_id,
        normalize_policy_profile_fn=_normalize_policy_profile,
        build_effective_tools_state_fn=lambda override: _build_effective_tools_state(
            session_override=override
        ),
    )


def _plan_revision_value(plan: dict[str, JSONValue]) -> int:
    return workflow_state.plan_revision_value(plan)


def _increment_plan_revision(plan: dict[str, JSONValue]) -> int:
    return workflow_state.increment_plan_revision(plan)


async def _apply_agent_runtime_state(
    *,
    agent: AgentProtocol,
    hub: UIHub,
    session_id: str,
) -> tuple[str, dict[str, JSONValue] | None, dict[str, JSONValue] | None]:
    async def _security_loader(
        loader_hub: _workflow_runtime.WorkflowHubProtocol,
        loader_session_id: str,
    ) -> tuple[dict[str, bool], dict[str, JSONValue]]:
        return await _load_effective_session_security(
            hub=cast(UIHub, loader_hub),
            session_id=loader_session_id,
        )

    return await _workflow_runtime.apply_agent_runtime_state(
        agent=agent,
        hub=hub,
        session_id=session_id,
        load_effective_session_security_fn=_security_loader,
        normalize_mode_value_fn=lambda value: _normalize_mode_value(value, default="ask"),
        normalize_plan_payload_fn=_normalize_plan_payload,
        normalize_task_payload_fn=_normalize_task_payload,
    )


def _decision_workflow_context(
    *,
    mode: str,
    active_plan: dict[str, JSONValue] | None,
    active_task: dict[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    return workflow_state.decision_workflow_context(
        mode=mode,
        active_plan=active_plan,
        active_task=active_task,
    )


async def _set_current_plan_step_status(
    *,
    hub: UIHub,
    session_id: str,
    status: str,
) -> None:
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    if mode != "act":
        return
    active_plan = _normalize_plan_payload(workflow.get("active_plan"))
    active_task = _normalize_task_payload(workflow.get("active_task"))
    if active_plan is None or active_task is None:
        return
    step_id_raw = active_task.get("current_step_id")
    if not isinstance(step_id_raw, str) or not step_id_raw.strip():
        return
    updated_plan = _plan_mark_step(
        active_plan,
        step_id=step_id_raw.strip(),
        status=status,
    )
    await hub.set_session_workflow(
        session_id,
        mode="act",
        active_plan=updated_plan,
        active_task=active_task,
    )


def _normalize_ui_decision_options(raw: object) -> list[dict[str, JSONValue]]:
    return decision_flow.normalize_ui_decision_options(
        raw,
        normalize_json_value_fn=_normalize_json_value,
    )


def _normalize_ui_decision(
    raw: object,
    *,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> dict[str, JSONValue] | None:
    return decision_flow.normalize_ui_decision(
        raw,
        session_id=session_id,
        trace_id=trace_id,
        utc_now_iso=_utc_now_iso,
        normalize_json_value_fn=_normalize_json_value,
        normalize_ui_decision_options_fn=_normalize_ui_decision_options,
        ui_decision_kinds=UI_DECISION_KINDS,
        ui_decision_statuses=UI_DECISION_STATUSES,
    )


def _build_ui_approval_decision(
    *,
    approval_request: dict[str, JSONValue],
    session_id: str,
    source_endpoint: str,
    resume_payload: dict[str, JSONValue],
    trace_id: str | None = None,
    workflow_context: dict[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    return decision_flow.build_ui_approval_decision(
        approval_request=approval_request,
        session_id=session_id,
        source_endpoint=source_endpoint,
        resume_payload=resume_payload,
        trace_id=trace_id,
        workflow_context=workflow_context,
        all_categories=ALL_CATEGORIES,
        normalize_json_value_fn=_normalize_json_value,
        utc_now_iso=_utc_now_iso,
    )


def _build_plan_execute_decision(
    *,
    session_id: str,
    plan: dict[str, JSONValue],
    mode: str,
    active_task: dict[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    return decision_flow.build_plan_execute_decision(
        session_id=session_id,
        plan=plan,
        mode=mode,
        active_task=active_task,
        utc_now_iso=_utc_now_iso,
        plan_revision_value_fn=_plan_revision_value,
        decision_workflow_context_fn=lambda m, p, t: _decision_workflow_context(
            mode=m,
            active_plan=p,
            active_task=t,
        ),
    )


def _decision_is_pending_blocking(decision: dict[str, JSONValue] | None) -> bool:
    return decision_flow.decision_is_pending_blocking(decision)


def _decision_with_status(
    decision: dict[str, JSONValue],
    *,
    status: str,
    resolved: bool = False,
) -> dict[str, JSONValue]:
    return decision_flow.decision_with_status(
        decision,
        status=status,
        resolved=resolved,
        utc_now_iso=_utc_now_iso,
    )


def _decision_type_value(decision: dict[str, JSONValue]) -> str:
    return decision_flow.decision_type_value(decision)


def _decision_categories(decision: dict[str, JSONValue]) -> set[ApprovalCategory]:
    return decision_flow.decision_categories(decision, category_map=_CATEGORY_MAP)


def _decision_mismatch_response(
    *,
    expected_id: str,
    actual_decision: dict[str, JSONValue] | None,
) -> web.Response:
    details = decision_flow.decision_mismatch_details(
        expected_id=expected_id,
        actual_decision=actual_decision,
    )
    return _error_response(
        status=409,
        message="Decision id mismatch.",
        error_type="invalid_request_error",
        code="decision_id_mismatch",
        details=details,
    )


def _normalize_json_value(value: object) -> JSONValue:
    return workflow_state.normalize_json_value(value)


_is_document_like_output = _ui_artifacts._is_document_like_output
_request_likely_canvas = _ui_artifacts._request_likely_canvas

_should_render_result_in_canvas = _ui_artifacts._should_render_result_in_canvas
_sanitize_download_filename = _ui_artifacts._sanitize_download_filename
_safe_zip_entry_name = _ui_artifacts._safe_zip_entry_name
_artifact_mime_from_ext = _ui_artifacts._artifact_mime_from_ext
_normalize_candidate_file_name = _ui_artifacts._normalize_candidate_file_name
_is_probable_named_file = _ui_artifacts._is_probable_named_file
_extract_named_file_markers = _ui_artifacts._extract_named_file_markers
_normalize_code_fence_content = _ui_artifacts._normalize_code_fence_content
_extract_named_files_from_output = _ui_artifacts._extract_named_files_from_output
_build_output_artifacts = _ui_artifacts._build_output_artifacts
_build_canvas_chat_summary = _ui_artifacts._build_canvas_chat_summary
_canvas_summary_title_from_artifact = _ui_artifacts._canvas_summary_title_from_artifact
_stream_preview_indicates_canvas = _ui_artifacts._stream_preview_indicates_canvas


def _stream_preview_ready_for_chat(preview_text: str) -> bool:
    return _streaming._stream_preview_ready_for_chat(
        preview_text,
        chat_stream_warmup_chars=CHAT_STREAM_WARMUP_CHARS,
    )


async def _publish_canvas_stream(
    hub: UIHub,
    *,
    session_id: str,
    artifact_id: str,
    content: str,
) -> None:
    await _publish_canvas_stream_base(
        hub,
        session_id=session_id,
        artifact_id=artifact_id,
        content=content,
    )


_user_plane_forbidden_settings_key = _ui_settings._user_plane_forbidden_settings_key
_normalize_policy_profile = _ui_settings._normalize_policy_profile
_tools_state_for_profile = _ui_settings._tools_state_for_profile


def _load_tools_state() -> dict[str, bool]:
    try:
        return load_tools_config().to_dict()
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_TOOLS_STATE)


def _save_tools_state(state: dict[str, bool]) -> None:
    payload: dict[str, object] = {key: value for key, value in state.items()}
    save_tools_config(ToolsConfig.from_dict(payload))


def _load_ui_settings_blob() -> dict[str, object]:
    return _ui_settings._load_ui_settings_blob(ui_settings_path=UI_SETTINGS_PATH)


def _save_ui_settings_blob(payload: dict[str, object]) -> None:
    _ui_settings._save_ui_settings_blob(payload, ui_settings_path=UI_SETTINGS_PATH)


def _load_personalization_settings() -> tuple[str, str]:
    return _ui_settings._load_personalization_settings(ui_settings_path=UI_SETTINGS_PATH)


def _save_personalization_settings(*, tone: str, system_prompt: str) -> None:
    _ui_settings._save_personalization_settings(
        tone=tone,
        system_prompt=system_prompt,
        ui_settings_path=UI_SETTINGS_PATH,
    )


def _load_embeddings_settings() -> UIEmbeddingsSettings:
    return _ui_settings._load_embeddings_settings(ui_settings_path=UI_SETTINGS_PATH)


def _save_embeddings_settings(settings: UIEmbeddingsSettings) -> None:
    _ui_settings._save_embeddings_settings(settings, ui_settings_path=UI_SETTINGS_PATH)


def _load_composer_settings() -> tuple[bool, int]:
    return _ui_settings._load_composer_settings(ui_settings_path=UI_SETTINGS_PATH)


def _save_composer_settings(
    *,
    long_paste_to_file_enabled: bool,
    long_paste_threshold_chars: int,
) -> None:
    _ui_settings._save_composer_settings(
        long_paste_to_file_enabled=long_paste_to_file_enabled,
        long_paste_threshold_chars=long_paste_threshold_chars,
        ui_settings_path=UI_SETTINGS_PATH,
    )


def _load_policy_settings() -> tuple[str, bool, str | None]:
    return _ui_settings._load_policy_settings(ui_settings_path=UI_SETTINGS_PATH)


def _save_policy_settings(
    *,
    profile: str,
    yolo_armed: bool,
    yolo_armed_at: str | None,
) -> None:
    _ui_settings._save_policy_settings(
        profile=profile,
        yolo_armed=yolo_armed,
        yolo_armed_at=yolo_armed_at,
        ui_settings_path=UI_SETTINGS_PATH,
    )


def _load_provider_api_keys() -> dict[str, str]:
    return _ui_settings._load_provider_api_keys(ui_settings_path=UI_SETTINGS_PATH)


def _save_provider_api_keys(api_keys: dict[str, str]) -> None:
    _ui_settings._save_provider_api_keys(api_keys, ui_settings_path=UI_SETTINGS_PATH)


def _load_provider_runtime_checks() -> dict[str, dict[str, JSONValue]]:
    return _ui_settings._load_provider_runtime_checks(ui_settings_path=UI_SETTINGS_PATH)


def _save_provider_runtime_checks(
    checks: dict[str, dict[str, JSONValue]],
) -> None:
    _ui_settings._save_provider_runtime_checks(checks, ui_settings_path=UI_SETTINGS_PATH)


def _resolve_provider_api_key(
    provider: str,
    *,
    settings_api_keys: dict[str, str] | None = None,
) -> str | None:
    return _ui_settings._resolve_provider_api_key(
        provider,
        settings_api_keys=settings_api_keys,
        ui_settings_path=UI_SETTINGS_PATH,
    )


def _provider_api_key_source(
    provider: str,
    *,
    settings_api_keys: dict[str, str] | None = None,
) -> Literal["settings", "env", "missing"]:
    return _ui_settings._provider_api_key_source(
        provider,
        settings_api_keys=settings_api_keys,
        ui_settings_path=UI_SETTINGS_PATH,
    )


def _provider_settings_payload() -> list[dict[str, JSONValue]]:
    return _ui_settings._provider_settings_payload(ui_settings_path=UI_SETTINGS_PATH)


def _build_settings_payload() -> dict[str, JSONValue]:
    tone, system_prompt = _load_personalization_settings()
    long_paste_to_file_enabled, long_paste_threshold_chars = _load_composer_settings()
    policy_profile, yolo_armed, yolo_armed_at = _load_policy_settings()
    memory_config = load_memory_config()
    embeddings_settings = _load_embeddings_settings()
    tools_state = _load_tools_state()
    tools_registry = {key: value for key, value in tools_state.items() if key != "safe_mode"}
    return {
        "settings": {
            "personalization": {"tone": tone, "system_prompt": system_prompt},
            "composer": {
                "long_paste_to_file_enabled": long_paste_to_file_enabled,
                "long_paste_threshold_chars": long_paste_threshold_chars,
            },
            "memory": {
                "auto_save_dialogue": memory_config.auto_save_dialogue,
                "inbox_max_items": memory_config.inbox_max_items,
                "inbox_ttl_days": memory_config.inbox_ttl_days,
                "inbox_writes_per_minute": memory_config.inbox_writes_per_minute,
                "embeddings": {
                    "provider": embeddings_settings.provider,
                    "local_model": embeddings_settings.local_model,
                    "openai_model": embeddings_settings.openai_model,
                },
            },
            "tools": {
                "state": tools_state,
                "registry": tools_registry,
            },
            "policy": {
                "profile": policy_profile,
                "yolo_armed": yolo_armed,
                "yolo_armed_at": yolo_armed_at,
            },
            "providers": _provider_settings_payload(),
        },
    }


def _parse_imported_session(raw: object, *, principal_id: str) -> PersistedSession | None:
    return session_transfer._parse_imported_session(
        raw,
        principal_id=principal_id,
        normalize_policy_profile_fn=_normalize_policy_profile,
        normalize_tools_state_payload_fn=_normalize_tools_state_payload,
        utc_iso_fn=_utc_iso,
    )


def _serialize_persisted_session(session: PersistedSession) -> dict[str, JSONValue]:
    return session_transfer._serialize_persisted_session(
        session,
        normalize_policy_profile_fn=_normalize_policy_profile,
        normalize_mode_value_fn=lambda value: _normalize_mode_value(value, default="ask"),
        normalize_plan_payload_fn=_normalize_plan_payload,
        normalize_task_payload_fn=_normalize_task_payload,
    )


async def _publish_agent_activity(
    hub: UIHub,
    *,
    session_id: str,
    phase: str,
    detail: str | None = None,
) -> None:
    payload: dict[str, JSONValue] = {"session_id": session_id, "phase": phase}
    if detail is not None and detail.strip():
        payload["detail"] = detail.strip()
    event: dict[str, JSONValue] = {
        "type": "agent.activity",
        "payload": payload,
    }
    try:
        await hub.publish(session_id, event)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to publish agent activity event", exc_info=True)


async def _workspace_root_for_session(hub: UIHub, session_id: str) -> Path:
    return await workspace_paths.workspace_root_for_session(
        hub=hub,
        session_id=session_id,
        fallback_root=WORKSPACE_ROOT,
    )


def _resolve_workspace_root_candidate(path_raw: str, *, policy_profile: str) -> Path:
    return workspace_paths.resolve_workspace_root_candidate(
        path_raw,
        policy_profile=policy_profile,
        workspace_root=WORKSPACE_ROOT,
    )


def _index_workspace_root(root: Path) -> dict[str, JSONValue]:
    return workspace_index.index_workspace_root(
        root=root,
        load_embeddings_settings=_load_embeddings_settings,
        resolve_provider_api_key=_resolve_provider_api_key,
        index_enabled_env=INDEX_ENABLED_ENV,
        ignored_dirs=WORKSPACE_INDEX_IGNORED_DIRS,
        allowed_extensions=WORKSPACE_INDEX_ALLOWED_EXTENSIONS,
        max_file_bytes=WORKSPACE_INDEX_MAX_FILE_BYTES,
    )


def _workspace_git_diff(root: Path) -> tuple[str, str | None]:
    return workspace_index.workspace_git_diff(root)


def _run_plan_readonly_audit(
    *,
    root: Path,
) -> tuple[list[dict[str, JSONValue]], dict[str, int]]:
    return workspace_runtime._run_plan_readonly_audit(
        root=root,
        plan_audit_timeout_seconds=PLAN_AUDIT_TIMEOUT_SECONDS,
        workspace_index_ignored_dirs=WORKSPACE_INDEX_IGNORED_DIRS,
        workspace_index_allowed_extensions=WORKSPACE_INDEX_ALLOWED_EXTENSIONS,
        plan_audit_max_total_bytes=PLAN_AUDIT_MAX_TOTAL_BYTES,
        plan_audit_max_read_files=PLAN_AUDIT_MAX_READ_FILES,
    )


_build_default_plan_steps = _workflow_runtime.build_default_plan_steps


def _build_plan_draft(
    *,
    goal: str,
    audit_log: list[dict[str, JSONValue]],
) -> dict[str, JSONValue]:
    return _workflow_runtime.build_plan_draft(
        goal=goal,
        audit_log=audit_log,
        utc_now_iso_fn=_utc_now_iso,
        plan_hash_payload_fn=_plan_hash_payload,
        build_default_plan_steps_fn=_build_default_plan_steps,
    )


def _plan_with_status(
    plan: dict[str, JSONValue],
    *,
    status: str,
) -> dict[str, JSONValue]:
    return _workflow_runtime.plan_with_status(
        plan,
        status=status,
        utc_now_iso_fn=_utc_now_iso,
        increment_plan_revision_fn=_increment_plan_revision,
        plan_hash_payload_fn=_plan_hash_payload,
    )


def _task_with_status(
    task: dict[str, JSONValue],
    *,
    status: str,
    current_step_id: str | None = None,
) -> dict[str, JSONValue]:
    return _workflow_runtime.task_with_status(
        task,
        status=status,
        current_step_id=current_step_id,
        utc_now_iso_fn=_utc_now_iso,
    )


def _plan_mark_step(
    plan: dict[str, JSONValue],
    *,
    step_id: str,
    status: str,
    evidence: dict[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    return plan_edit.plan_mark_step(
        plan,
        step_id=step_id,
        status=status,
        evidence=evidence,
        plan_step_statuses=PLAN_STEP_STATUSES,
        utc_now_iso=_utc_now_iso,
        increment_plan_revision_fn=_increment_plan_revision,
        plan_hash_payload_fn=_plan_hash_payload,
    )


def _validate_text_limit(value: str, *, field: str) -> None:
    plan_edit.validate_text_limit(
        value,
        field=field,
        max_chars=PLAN_MAX_TEXT_FIELD_CHARS,
    )


def _find_forbidden_plan_key(value: JSONValue) -> str | None:
    return plan_edit.find_forbidden_plan_key(value)


def _normalize_plan_step_insert(raw: object) -> dict[str, JSONValue]:
    return plan_edit.normalize_plan_step_insert(
        raw,
        plan_step_statuses=PLAN_STEP_STATUSES,
        normalize_string_list_fn=_normalize_string_list,
        normalize_json_value_fn=_normalize_json_value,
        validate_text_limit_fn=lambda value, field: _validate_text_limit(value, field=field),
    )


def _normalize_plan_step_changes(raw: object) -> dict[str, JSONValue]:
    return plan_edit.normalize_plan_step_changes(
        raw,
        plan_step_statuses=PLAN_STEP_STATUSES,
        normalize_string_list_fn=_normalize_string_list,
        normalize_json_value_fn=_normalize_json_value,
        validate_text_limit_fn=lambda value, field: _validate_text_limit(value, field=field),
    )


def _validate_plan_document(plan: dict[str, JSONValue]) -> None:
    plan_edit.validate_plan_document(
        plan,
        plan_max_payload_bytes=PLAN_MAX_PAYLOAD_BYTES,
        plan_max_steps=PLAN_MAX_STEPS,
        validate_text_limit_fn=lambda value, field: _validate_text_limit(value, field=field),
        find_forbidden_plan_key_fn=_find_forbidden_plan_key,
        normalize_plan_step_insert_fn=_normalize_plan_step_insert,
    )


def _plan_apply_edit_operation(
    *,
    plan: dict[str, JSONValue],
    operation: dict[str, JSONValue],
) -> dict[str, JSONValue]:
    return plan_edit.plan_apply_edit_operation(
        plan=plan,
        operation=operation,
        normalize_plan_step_insert_fn=_normalize_plan_step_insert,
        normalize_plan_step_changes_fn=_normalize_plan_step_changes,
        utc_now_iso=_utc_now_iso,
        increment_plan_revision_fn=_increment_plan_revision,
        validate_plan_document_fn=_validate_plan_document,
        plan_hash_payload_fn=_plan_hash_payload,
    )


def _find_next_todo_step(plan: dict[str, JSONValue]) -> dict[str, JSONValue] | None:
    return plan_edit.find_next_todo_step(plan)


async def _run_plan_runner(
    *,
    app: web.Application,
    session_id: str,
    plan_id: str,
    task_id: str,
) -> None:
    return await _workflow_runtime.run_plan_runner(
        app=app,
        session_id=session_id,
        plan_id=plan_id,
        task_id=task_id,
        normalize_plan_payload_fn=_normalize_plan_payload,
        normalize_task_payload_fn=_normalize_task_payload,
        normalize_mode_value_fn=lambda value: _normalize_mode_value(value, default="ask"),
        find_next_todo_step_fn=_find_next_todo_step,
        plan_with_status_fn=lambda plan, status: _plan_with_status(plan, status=status),
        task_with_status_fn=lambda task, status, current_step_id: _task_with_status(
            task,
            status=status,
            current_step_id=current_step_id,
        ),
        plan_mark_step_fn=lambda plan, step_id, status, evidence: _plan_mark_step(
            plan,
            step_id=step_id,
            status=status,
            evidence=evidence,
        ),
        utc_now_iso_fn=_utc_now_iso,
    )


def create_app(
    *,
    agent: AgentProtocol | None = None,
    max_request_bytes: int | None = None,
    ui_storage: UISessionStorage | None = None,
    auth_config: HttpAuthConfig | None = None,
) -> web.Application:
    from server.http.app import create_app as _create_app

    return _create_app(
        agent=agent,
        max_request_bytes=max_request_bytes,
        ui_storage=ui_storage,
        auth_config=auth_config,
    )


def run_server(config: HttpServerConfig) -> None:
    from server.http.app import run_server as _run_server

    _run_server(config)


def main() -> None:
    from server.http.app import main as _main

    _main()


__all__ = ["create_app", "main", "run_server"]
