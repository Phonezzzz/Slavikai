from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, cast

import requests
from aiohttp import web

from config.http_server_config import (
    HttpAuthConfig,
    HttpServerConfig,
)
from config.memory_config import load_memory_config, save_memory_config
from config.tools_config import (
    DEFAULT_TOOLS_STATE,
    load_tools_config,
    save_tools_config,
)
from core.approval_policy import ApprovalCategory
from server.http.common import (
    chat_payload as _chat_payload,
)
from server.http.common import (
    chat_request as _chat_request,
)
from server.http.common import (
    github_import,
    runtime_contract,
    session_transfer,
    settings_runtime,
    workspace_index,
    workspace_runtime,
    workspace_runtime_bindings,
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
    ui_runtime as _ui_runtime,
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
_CATEGORY_MAP: Final[dict[str, ApprovalCategory]] = _ui_runtime._CATEGORY_MAP

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
WORKSPACE_INDEX_IGNORED_DIRS: Final[set[str]] = workspace_index.DEFAULT_IGNORED_DIRS
WORKSPACE_INDEX_ALLOWED_EXTENSIONS: Final[set[str]] = workspace_index.DEFAULT_ALLOWED_EXTENSIONS
WORKSPACE_INDEX_MAX_FILE_BYTES: Final[int] = workspace_index.DEFAULT_MAX_FILE_BYTES
POLICY_PROFILES: Final[set[str]] = _ui_settings.POLICY_PROFILES
DEFAULT_POLICY_PROFILE: Final[str] = _ui_settings.DEFAULT_POLICY_PROFILE
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
SESSION_MODES: Final[set[str]] = _ui_runtime.SESSION_MODES
PLAN_AUDIT_MAX_READ_FILES: Final[int] = 15
PLAN_AUDIT_MAX_TOTAL_BYTES: Final[int] = 300_000
PLAN_AUDIT_MAX_SEARCH_CALLS: Final[int] = 10
PLAN_AUDIT_TIMEOUT_SECONDS: Final[int] = 20


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


def _requests_module() -> RequestsModuleProtocol:
    return requests


_SETTINGS_RUNTIME = settings_runtime.SettingsRuntimeBindings(
    ui_settings_path_getter=lambda: UI_SETTINGS_PATH,
    load_memory_config_fn=lambda: load_memory_config(),
    save_memory_config_fn=lambda config: save_memory_config(config),
    load_tools_config_fn=lambda: load_tools_config(),
    save_tools_config_fn=lambda config: save_tools_config(config),
)

_provider_auth_headers = _SETTINGS_RUNTIME.provider_auth_headers
_fetch_provider_models = _SETTINGS_RUNTIME.fetch_provider_models
_load_memory_config_runtime = _SETTINGS_RUNTIME.load_memory_config_runtime
_save_memory_config_runtime = _SETTINGS_RUNTIME.save_memory_config_runtime
_load_tools_state = _SETTINGS_RUNTIME.load_tools_state
_save_tools_state = _SETTINGS_RUNTIME.save_tools_state
_load_ui_settings_blob = _SETTINGS_RUNTIME.load_ui_settings_blob
_save_ui_settings_blob = _SETTINGS_RUNTIME.save_ui_settings_blob
_load_personalization_settings = _SETTINGS_RUNTIME.load_personalization_settings
_save_personalization_settings = _SETTINGS_RUNTIME.save_personalization_settings
_load_embeddings_settings = _SETTINGS_RUNTIME.load_embeddings_settings
_save_embeddings_settings = _SETTINGS_RUNTIME.save_embeddings_settings
_load_composer_settings = _SETTINGS_RUNTIME.load_composer_settings
_save_composer_settings = _SETTINGS_RUNTIME.save_composer_settings
_load_policy_settings = _SETTINGS_RUNTIME.load_policy_settings
_save_policy_settings = _SETTINGS_RUNTIME.save_policy_settings
_load_provider_api_keys = _SETTINGS_RUNTIME.load_provider_api_keys
_save_provider_api_keys = _SETTINGS_RUNTIME.save_provider_api_keys
_load_provider_runtime_checks = _SETTINGS_RUNTIME.load_provider_runtime_checks
_save_provider_runtime_checks = _SETTINGS_RUNTIME.save_provider_runtime_checks
_resolve_provider_api_key = _SETTINGS_RUNTIME.resolve_provider_api_key
_provider_api_key_source = _SETTINGS_RUNTIME.provider_api_key_source
_provider_settings_payload = _SETTINGS_RUNTIME.provider_settings_payload
_build_settings_payload = _SETTINGS_RUNTIME.build_settings_payload


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


_serialize_approval_request = _ui_runtime._serialize_approval_request
_utc_now_iso = _ui_runtime._utc_now_iso
_normalize_mode_value = _ui_runtime._normalize_mode_value
_normalize_string_list = _ui_runtime._normalize_string_list
_normalize_plan_step = _ui_runtime._normalize_plan_step
_plan_hash_payload = _ui_runtime._plan_hash_payload
_normalize_plan_payload = _ui_runtime._normalize_plan_payload
_normalize_task_payload = _ui_runtime._normalize_task_payload
_plan_revision_value = _ui_runtime._plan_revision_value
_increment_plan_revision = _ui_runtime._increment_plan_revision
_decision_workflow_context = _ui_runtime._decision_workflow_context
_set_current_plan_step_status = _ui_runtime._set_current_plan_step_status
_normalize_ui_decision_options = _ui_runtime._normalize_ui_decision_options
_normalize_ui_decision = _ui_runtime._normalize_ui_decision
_build_ui_approval_decision = _ui_runtime._build_ui_approval_decision
_build_plan_execute_decision = _ui_runtime._build_plan_execute_decision
_decision_is_pending_blocking = _ui_runtime._decision_is_pending_blocking
_decision_with_status = _ui_runtime._decision_with_status
_decision_type_value = _ui_runtime._decision_type_value
_decision_categories = _ui_runtime._decision_categories
_normalize_json_value = _ui_runtime._normalize_json_value


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


def _decision_mismatch_response(
    *,
    expected_id: str,
    actual_decision: dict[str, JSONValue] | None,
) -> web.Response:
    details = _ui_runtime._decision_mismatch_details(
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


_stream_preview_ready_for_chat = _ui_runtime._stream_preview_ready_for_chat
_publish_canvas_stream = _ui_runtime._publish_canvas_stream


_user_plane_forbidden_settings_key = _ui_settings._user_plane_forbidden_settings_key
_normalize_policy_profile = _ui_settings._normalize_policy_profile
_tools_state_for_profile = _ui_settings._tools_state_for_profile


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


_publish_agent_activity = _ui_runtime._publish_agent_activity

_WORKSPACE_RUNTIME = workspace_runtime_bindings.WorkspaceRuntimeBindings(
    workspace_root_getter=lambda: WORKSPACE_ROOT,
    load_embeddings_settings_fn=_load_embeddings_settings,
    resolve_provider_api_key_fn=lambda provider: _resolve_provider_api_key(provider),
    index_enabled_env=INDEX_ENABLED_ENV,
    workspace_index_ignored_dirs=WORKSPACE_INDEX_IGNORED_DIRS,
    workspace_index_allowed_extensions=WORKSPACE_INDEX_ALLOWED_EXTENSIONS,
    workspace_index_max_file_bytes=WORKSPACE_INDEX_MAX_FILE_BYTES,
    plan_audit_timeout_seconds=PLAN_AUDIT_TIMEOUT_SECONDS,
    plan_audit_max_total_bytes=PLAN_AUDIT_MAX_TOTAL_BYTES,
    plan_audit_max_read_files=PLAN_AUDIT_MAX_READ_FILES,
)
_workspace_root_for_session = _WORKSPACE_RUNTIME.workspace_root_for_session
_resolve_workspace_root_candidate = _WORKSPACE_RUNTIME.resolve_workspace_root_candidate
_index_workspace_root = _WORKSPACE_RUNTIME.index_workspace_root
_workspace_git_diff = _WORKSPACE_RUNTIME.workspace_git_diff
_run_plan_readonly_audit = _WORKSPACE_RUNTIME.run_plan_readonly_audit


_build_plan_draft = _ui_runtime._build_plan_draft
_plan_with_status = _ui_runtime._plan_with_status
_task_with_status = _ui_runtime._task_with_status
_plan_mark_step = _ui_runtime._plan_mark_step
_validate_text_limit = _ui_runtime._validate_text_limit
_find_forbidden_plan_key = _ui_runtime._find_forbidden_plan_key
_normalize_plan_step_insert = _ui_runtime._normalize_plan_step_insert
_normalize_plan_step_changes = _ui_runtime._normalize_plan_step_changes
_validate_plan_document = _ui_runtime._validate_plan_document
_plan_apply_edit_operation = _ui_runtime._plan_apply_edit_operation
_find_next_todo_step = _ui_runtime._find_next_todo_step
_run_plan_runner = _ui_runtime._run_plan_runner


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
