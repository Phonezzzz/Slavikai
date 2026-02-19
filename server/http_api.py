from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import requests
from aiohttp import web

from config.http_server_config import (
    DEFAULT_MAX_REQUEST_BYTES,
    HttpAuthConfig,
    HttpServerConfig,
    ensure_http_auth_boot_config,
    resolve_http_auth_config,
    resolve_http_server_config,
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
from core.mwv.models import MWV_REPORT_PREFIX
from core.tracer import TraceRecord
from llm.types import ModelConfig
from server.http.common import (
    chat_request as _chat_request,
)
from server.http.common import (
    decision_flow,
    github_import,
    plan_edit,
    workflow_state,
    workspace_index,
    workspace_paths,
)
from server.http.common import (
    streaming as _streaming,
)
from server.http.common import (
    trace_views as _trace_views,
)
from server.http.common import (
    ui_settings as _ui_settings,
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
from server.lazy_agent import LazyAgentProvider
from server.ui_hub import UIHub
from server.ui_session_storage import PersistedSession, SQLiteUISessionStorage, UISessionStorage
from shared.memory_companion_models import FeedbackLabel, FeedbackRating
from shared.models import JSONValue, LLMMessage, ToolResult
from shared.sanitize import safe_json_loads
from tools.workspace_tools import (
    WORKSPACE_ROOT as DEFAULT_WORKSPACE_ROOT,
)

ChatRequest = _chat_request.ChatRequest
_is_sampling_key = _chat_request._is_sampling_key
_validate_messages = _chat_request._validate_messages
_parse_chat_request = _chat_request._parse_chat_request
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
CANVAS_LINE_THRESHOLD: Final[int] = 40
CANVAS_CHAR_THRESHOLD: Final[int] = 1800
CANVAS_CODE_LINE_THRESHOLD: Final[int] = 28
CANVAS_DOCUMENT_LINE_THRESHOLD: Final[int] = 24
CANVAS_STATUS_CHARS_STEP: Final[int] = 640
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT: Final[Path] = DEFAULT_WORKSPACE_ROOT
MAX_DOWNLOAD_BYTES: Final[int] = 5_000_000
MAX_ATTACHMENTS_PER_MESSAGE: Final[int] = 8
MAX_ATTACHMENT_CHARS: Final[int] = 80_000
MAX_TOTAL_ATTACHMENTS_CHARS: Final[int] = 160_000
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
_REQUEST_FILENAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<name>[A-Za-z0-9._/\-]+\.[A-Za-z0-9]{1,12})"
)
_CODE_FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]*)(?P<sep>\n|\\n)(?P<code>.*?)```",
    re.DOTALL,
)
_CHAT_WEB_INTENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(проверь( в интернете)?|поищи|найди в интернете|поиск(ай)? в интернете|"
    r"search( the)? web|check( in| on)?( the)? internet|look up|google)",
    re.IGNORECASE,
)
_ARTIFACT_FILE_EXTENSIONS: Final[set[str]] = {
    "bash",
    "c",
    "cfg",
    "conf",
    "cpp",
    "css",
    "csv",
    "env",
    "go",
    "h",
    "hpp",
    "html",
    "ini",
    "java",
    "js",
    "json",
    "jsx",
    "kt",
    "md",
    "php",
    "py",
    "rb",
    "rs",
    "sh",
    "sql",
    "swift",
    "toml",
    "ts",
    "tsx",
    "txt",
    "xml",
    "yaml",
    "yml",
}
_EXT_TO_MIME: Final[dict[str, str]] = {
    "bash": "text/x-shellscript",
    "c": "text/plain",
    "cpp": "text/plain",
    "py": "text/x-python",
    "js": "text/javascript",
    "jsx": "text/javascript",
    "ts": "text/typescript",
    "tsx": "text/typescript",
    "json": "application/json",
    "html": "text/html",
    "css": "text/css",
    "md": "text/markdown",
    "txt": "text/plain",
    "sh": "text/x-shellscript",
    "yaml": "text/plain",
    "yml": "text/plain",
    "toml": "text/plain",
    "xml": "application/xml",
    "sql": "text/plain",
}
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


class SessionApprovalStore:
    def __init__(self) -> None:
        self._approved: dict[str, set[ApprovalCategory]] = {}
        self._lock = asyncio.Lock()

    async def approve(
        self, session_id: str, categories: set[ApprovalCategory]
    ) -> set[ApprovalCategory]:
        async with self._lock:
            existing = self._approved.get(session_id, set())
            existing.update(categories)
            self._approved[session_id] = existing
            return set(existing)

    async def is_approved(self, session_id: str) -> bool:
        async with self._lock:
            return bool(self._approved.get(session_id))

    async def get_categories(self, session_id: str) -> set[ApprovalCategory]:
        async with self._lock:
            return set(self._approved.get(session_id, set()))


class TracerProtocol(Protocol):
    def log(
        self,
        event_type: str,
        message: str,
        meta: dict[str, JSONValue] | None = None,
    ) -> None: ...


class AgentProtocol(Protocol):
    brain: object
    tools_enabled: dict[str, bool]
    last_approval_request: ApprovalRequest | None
    last_chat_interaction_id: str | None
    tracer: TracerProtocol

    def set_session_context(
        self,
        session_id: str | None,
        approved_categories: set[ApprovalCategory],
    ) -> None: ...

    def reconfigure_models(
        self,
        main_config: ModelConfig,
        main_api_key: str | None = None,
        *,
        persist: bool = True,
    ) -> None: ...

    def respond(self, messages: list[LLMMessage]) -> str: ...

    def update_tools_enabled(self, state: dict[str, bool]) -> None: ...
    def apply_runtime_tools_enabled(self, state: dict[str, bool]) -> None: ...

    def call_tool(
        self,
        name: str,
        args: dict[str, JSONValue] | None = None,
        raw_input: str | None = None,
    ) -> ToolResult: ...

    def record_feedback_event(
        self,
        *,
        interaction_id: str,
        rating: FeedbackRating,
        labels: list[FeedbackLabel],
        free_text: str | None,
    ) -> None: ...


class RequestsModuleProtocol(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = ...,
        data: dict[str, str] | None = ...,
        files: dict[str, tuple[str, bytes, str]] | None = ...,
        timeout: int | float | None = ...,
    ) -> requests.Response: ...


def _model_not_selected_response() -> web.Response:
    return _error_response(
        status=409,
        message="Не выбрана модель. Выберите модель в UI и повторите.",
        error_type="configuration_error",
        code="model_not_selected",
    )


def _model_not_allowed_response(model_id: str) -> web.Response:
    return _error_response(
        status=409,
        message=f"Модель '{model_id}' не входит в whitelist.",
        error_type="configuration_error",
        code="model_not_allowed",
        details={"model": model_id},
    )


async def _resolve_agent(request: web.Request) -> AgentProtocol | None:
    provider: LazyAgentProvider[AgentProtocol] = request.app["agent_provider"]
    try:
        return await provider.get()
    except RuntimeError as exc:
        if "Не выбрана модель" in str(exc):
            return None
        raise


def _extract_session_id(request: web.Request, payload: dict[str, object]) -> str | None:
    header_value = request.headers.get("X-Slavik-Session", "").strip()
    if header_value:
        return header_value
    meta_raw = payload.get("slavik_meta")
    if meta_raw is None:
        return None
    if not isinstance(meta_raw, dict):
        return None
    session_raw = meta_raw.get("session_id")
    if isinstance(session_raw, str) and session_raw.strip():
        return session_raw.strip()
    return None


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
    normalized = path_raw.strip()
    if not normalized:
        raise ValueError("path required")
    candidate = (WORKSPACE_ROOT / normalized).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError("path outside workspace") from exc
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError("file not found")
    if candidate.stat().st_size > MAX_DOWNLOAD_BYTES:
        raise ValueError("file too large")
    return candidate


def _artifact_file_payload(
    artifact: dict[str, JSONValue],
) -> tuple[str, str, str]:
    artifact_kind = artifact.get("artifact_kind")
    if artifact_kind != "file":
        raise ValueError("artifact is not file")
    file_name_raw = artifact.get("file_name")
    file_content_raw = artifact.get("file_content")
    file_ext_raw = artifact.get("file_ext")
    if not isinstance(file_name_raw, str) or not file_name_raw.strip():
        raise ValueError("artifact file_name missing")
    if not isinstance(file_content_raw, str):
        raise ValueError("artifact file_content missing")
    ext = file_ext_raw.strip().lower() if isinstance(file_ext_raw, str) else ""
    file_name = _sanitize_download_filename(file_name_raw)
    inferred_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    mime = _artifact_mime_from_ext(ext or inferred_ext)
    return file_name, file_content_raw, mime


def _serialize_trace_events(
    events: list[TraceRecord],
) -> list[dict[str, JSONValue]]:
    serialized: list[dict[str, JSONValue]] = []
    for event in events:
        meta = event.get("meta")
        serialized.append(
            {
                "timestamp": str(event.get("timestamp") or ""),
                "event": str(event.get("event") or ""),
                "message": str(event.get("message") or ""),
                "meta": meta if isinstance(meta, dict) else {},
            },
        )
    return serialized


def _ui_messages_to_llm(messages: list[dict[str, JSONValue]]) -> list[LLMMessage]:
    def _message_attachments(raw: object) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        normalized: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name_raw = item.get("name")
            mime_raw = item.get("mime")
            content_raw = item.get("content")
            if (
                isinstance(name_raw, str)
                and name_raw.strip()
                and isinstance(mime_raw, str)
                and mime_raw.strip()
                and isinstance(content_raw, str)
            ):
                normalized.append(
                    {
                        "name": name_raw.strip(),
                        "mime": mime_raw.strip(),
                        "content": content_raw,
                    },
                )
        return normalized

    def _serialize_attachments_for_llm(attachments: list[dict[str, str]]) -> str:
        if not attachments:
            return ""
        encoded = json.dumps(attachments, ensure_ascii=False)
        return f"\n\n<slavik-attachments-json>\n{encoded}\n</slavik-attachments-json>"

    parsed: list[LLMMessage] = []
    for item in messages:
        role_raw = item.get("role")
        content_raw = item.get("content")
        attachments_raw = item.get("attachments")
        role = role_raw if isinstance(role_raw, str) else None
        content = content_raw if isinstance(content_raw, str) else ""
        attachments = _message_attachments(attachments_raw)
        llm_content = content + _serialize_attachments_for_llm(attachments)
        if role == "user":
            parsed.append(LLMMessage(role="user", content=llm_content))
        elif role == "assistant":
            parsed.append(LLMMessage(role="assistant", content=llm_content))
        elif role == "system":
            parsed.append(LLMMessage(role="system", content=llm_content))
    return parsed


def _parse_ui_chat_attachments(raw: object) -> tuple[list[dict[str, str]], int]:
    if raw is None:
        return [], 0
    if not isinstance(raw, list):
        raise ValueError("attachments должен быть списком.")
    if len(raw) > MAX_ATTACHMENTS_PER_MESSAGE:
        raise OverflowError(f"attachments превышает лимит: {MAX_ATTACHMENTS_PER_MESSAGE} файлов.")
    attachments: list[dict[str, str]] = []
    total_chars = 0
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("attachments содержит не-объект.")
        name_raw = item.get("name")
        mime_raw = item.get("mime")
        content_raw = item.get("content")
        if not isinstance(name_raw, str) or not name_raw.strip():
            raise ValueError("attachments[].name должен быть непустой строкой.")
        if not isinstance(mime_raw, str) or not mime_raw.strip():
            raise ValueError("attachments[].mime должен быть непустой строкой.")
        if not isinstance(content_raw, str):
            raise ValueError("attachments[].content должен быть строкой.")
        normalized_name = name_raw.strip()
        normalized_mime = mime_raw.strip()
        if len(normalized_name) > 255:
            raise ValueError("attachments[].name слишком длинный (max 255).")
        if len(normalized_mime) > 128:
            raise ValueError("attachments[].mime слишком длинный (max 128).")
        if len(content_raw) > MAX_ATTACHMENT_CHARS:
            raise OverflowError(
                f"attachments[].content превышает лимит {MAX_ATTACHMENT_CHARS} символов."
            )
        total_chars += len(content_raw)
        attachments.append(
            {
                "name": normalized_name,
                "mime": normalized_mime,
                "content": content_raw,
            },
        )
    if total_chars > MAX_TOTAL_ATTACHMENTS_CHARS:
        raise OverflowError("Суммарный размер attachments превышает допустимый лимит.")
    return attachments, total_chars


def _extract_decision_payload(response_text: str) -> dict[str, JSONValue] | None:
    decision = safe_json_loads(response_text)
    if not isinstance(decision, dict):
        return None
    decision_id = decision.get("id")
    reason = decision.get("reason")
    summary = decision.get("summary")
    options = decision.get("options")
    if not isinstance(decision_id, str):
        return None
    if not isinstance(reason, str):
        return None
    if not isinstance(summary, str):
        return None
    if not isinstance(options, list):
        return None
    return decision


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
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, bool] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if key not in DEFAULT_TOOLS_STATE:
            continue
        if isinstance(value, bool):
            normalized[key] = value
    return normalized


def _build_effective_tools_state(
    *,
    session_override: dict[str, bool] | None,
) -> dict[str, bool]:
    defaults = _load_tools_state()
    effective = dict(defaults)
    if isinstance(session_override, dict):
        for key, value in session_override.items():
            if key in DEFAULT_TOOLS_STATE and isinstance(value, bool):
                effective[key] = value
    return effective


async def _load_effective_session_security(
    *,
    hub: UIHub,
    session_id: str,
) -> tuple[dict[str, bool], dict[str, JSONValue]]:
    policy = await hub.get_session_policy(session_id)
    profile_raw = policy.get("profile")
    profile = _normalize_policy_profile(profile_raw)
    session_tools_override = await hub.get_session_tools_state(session_id)
    effective_tools = _build_effective_tools_state(session_override=session_tools_override)
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
    effective_tools, _ = await _load_effective_session_security(
        hub=hub,
        session_id=session_id,
    )
    runtime_tools_setter = getattr(agent, "apply_runtime_tools_enabled", None)
    if callable(runtime_tools_setter):
        runtime_tools_setter(effective_tools)

    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    active_plan = _normalize_plan_payload(workflow.get("active_plan"))
    active_task = _normalize_task_payload(workflow.get("active_task"))
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


def _split_response_and_report(response_text: str) -> tuple[str, dict[str, JSONValue] | None]:
    marker_index = response_text.rfind(MWV_REPORT_PREFIX)
    if marker_index < 0:
        return response_text, None

    report_raw = response_text[marker_index + len(MWV_REPORT_PREFIX) :].strip()
    if not report_raw:
        return response_text, None

    parsed_report = safe_json_loads(report_raw)
    if not isinstance(parsed_report, dict):
        return response_text, None

    clean_text = response_text[:marker_index].rstrip()
    normalized_report: dict[str, JSONValue] = {}
    for key, value in parsed_report.items():
        normalized_report[str(key)] = _normalize_json_value(value)

    if not clean_text:
        return response_text, normalized_report
    return clean_text, normalized_report


def _normalize_trace_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _is_document_like_output(normalized: str) -> bool:
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) < CANVAS_DOCUMENT_LINE_THRESHOLD:
        return False
    heading_like = sum(
        1
        for line in lines[: min(len(lines), 20)]
        if line.startswith("#") or line.lower().startswith(("section ", "chapter ", "## "))
    )
    return heading_like >= 2


def _request_likely_canvas(user_input: str) -> bool:
    normalized = user_input.strip().lower()
    if not normalized:
        return False
    if _extract_named_file_markers(normalized):
        return True
    if re.search(r"\b\d+\s*(files?|файл(а|ов)?)\b", normalized):
        return True
    keywords = (
        "файл",
        "files",
        "project",
        "readme",
        "документ",
        "module",
        "класс",
        "config",
        "конфиг",
        "целиком",
        "полностью",
        "mini app",
        "mini ap",
        "мини приложение",
        "скрипт",
        "script",
        "напиши файл",
        "создай файл",
        "сгенерируй файл",
        "prilozhen",
        "skript",
        "fail",
        "celikom",
        "polnost",
    )
    return any(token in normalized for token in keywords)


def _request_likely_web_intent(user_input: str) -> bool:
    normalized = user_input.strip()
    if not normalized:
        return False
    if normalized.startswith("/"):
        return False
    return bool(_CHAT_WEB_INTENT_PATTERN.search(normalized))


def _should_render_result_in_canvas(
    *,
    response_text: str,
    files_from_tools: list[str],
    named_files_count: int,
    force_canvas: bool,
) -> bool:
    if force_canvas:
        return True
    if named_files_count > 0:
        return True
    if len(files_from_tools) >= 2:
        return True
    normalized = response_text.strip()
    if not normalized:
        return False
    lines = normalized.splitlines()
    line_count = len(lines)
    if line_count >= CANVAS_LINE_THRESHOLD:
        return True
    if len(normalized) >= CANVAS_CHAR_THRESHOLD:
        return True
    has_code_block = "```" in normalized
    if len(files_from_tools) == 1 and has_code_block and line_count >= 12:
        return True
    if has_code_block and line_count >= CANVAS_CODE_LINE_THRESHOLD:
        return True
    if _is_document_like_output(normalized):
        return True
    return False


def _sanitize_download_filename(file_name: str) -> str:
    normalized = file_name.replace("\\", "/").strip().split("/")[-1]
    if not normalized:
        return "artifact.txt"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in normalized)
    if safe.startswith("."):
        safe = f"file{safe}"
    return safe or "artifact.txt"


def _safe_zip_entry_name(file_name: str) -> str:
    normalized = file_name.replace("\\", "/").strip()
    parts = [part for part in normalized.split("/") if part and part not in {".", ".."}]
    if not parts:
        return "artifact.txt"
    safe_parts: list[str] = []
    for part in parts:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in part)
        if safe.startswith("."):
            safe = f"file{safe}"
        safe_parts.append(safe or "file")
    return "/".join(safe_parts)


def _artifact_mime_from_ext(ext: str | None) -> str:
    if not ext:
        return "text/plain"
    return _EXT_TO_MIME.get(ext.lower(), "text/plain")


def _normalize_candidate_file_name(raw_name: str) -> str:
    return raw_name.strip().strip("`\"'()[]{}<>.,;:!?")


def _is_probable_named_file(candidate: str) -> bool:
    normalized = _normalize_candidate_file_name(candidate)
    if not normalized:
        return False
    if normalized.startswith("//") or "://" in normalized:
        return False
    if "/" in normalized:
        head = normalized.split("/", 1)[0]
        if "." in head and not head.startswith("."):
            return False
    if "." not in normalized:
        return False
    ext = normalized.rsplit(".", 1)[-1].lower()
    if ext not in _ARTIFACT_FILE_EXTENSIONS:
        return False
    return True


def _extract_named_file_markers(marker_chunk: str) -> list[str]:
    names: list[str] = []
    for match in _REQUEST_FILENAME_PATTERN.finditer(marker_chunk):
        raw_name = match.group("name")
        if not _is_probable_named_file(raw_name):
            continue
        normalized = _normalize_candidate_file_name(raw_name)
        if normalized:
            names.append(normalized)
    return names


def _normalize_code_fence_content(code_raw: str, sep: str) -> str:
    normalized = code_raw
    if sep == "\\n" and "\n" not in normalized and "\\n" in normalized:
        normalized = normalized.replace("\\n", "\n")
    return normalized.rstrip("\n")


def _extract_named_files_from_output(response_text: str) -> list[dict[str, str]]:
    if not response_text.strip():
        return []
    files: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in _CODE_FENCE_PATTERN.finditer(response_text):
        code = _normalize_code_fence_content(match.group("code"), match.group("sep"))
        if not code.strip():
            continue
        lang = match.group("lang").strip().lower()
        marker_start = max(0, match.start() - 220)
        marker_chunk = response_text[marker_start : match.start()]
        marker_matches = _extract_named_file_markers(marker_chunk)
        if not marker_matches:
            continue
        file_name_raw = marker_matches[-1]
        file_name = _safe_zip_entry_name(file_name_raw)
        if file_name in seen:
            continue
        seen.add(file_name)
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if ext not in _ARTIFACT_FILE_EXTENSIONS:
            continue
        files.append(
            {
                "file_name": file_name,
                "file_ext": ext,
                "language": lang or ext,
                "file_content": code,
            }
        )
    return files


def _build_output_artifacts(
    *,
    response_text: str,
    display_target: Literal["chat", "canvas"],
    files_from_tools: list[str],
) -> list[dict[str, JSONValue]]:
    normalized = response_text.strip()
    if not normalized:
        return []

    now = datetime.now(UTC).isoformat()
    artifacts: list[dict[str, JSONValue]] = []
    named_files = _extract_named_files_from_output(response_text)
    for named_file in named_files:
        file_name = named_file["file_name"]
        file_ext = named_file["file_ext"] or None
        language = named_file["language"] or None
        file_content = named_file["file_content"]
        artifacts.append(
            {
                "id": uuid.uuid4().hex,
                "kind": "output",
                "artifact_kind": "file",
                "title": file_name,
                "content": file_content,
                "file_name": file_name,
                "file_ext": file_ext,
                "language": language,
                "file_content": file_content,
                "created_at": now,
                "display_target": display_target,
            }
        )

    del files_from_tools
    should_include_text = display_target == "canvas"
    if should_include_text and not named_files:
        first_line = next(
            (line.strip() for line in normalized.splitlines() if line.strip()),
            "Result",
        )
        artifacts.append(
            {
                "id": uuid.uuid4().hex,
                "kind": "output",
                "artifact_kind": "text",
                "title": first_line[:80],
                "content": response_text,
                "file_name": None,
                "file_ext": None,
                "language": None,
                "file_content": None,
                "created_at": now,
                "display_target": display_target,
            }
        )

    return artifacts


def _build_canvas_chat_summary(*, artifact_title: str | None) -> str:
    if isinstance(artifact_title, str):
        normalized = artifact_title.strip()
        if normalized:
            return f"Статус: результат сформирован в Canvas ({normalized})"
    return "Статус: результат сформирован в Canvas."


def _canvas_summary_title_from_artifact(
    artifact: dict[str, JSONValue] | None,
) -> str | None:
    if artifact is None:
        return None
    artifact_kind = artifact.get("artifact_kind")
    if artifact_kind != "file":
        return None
    file_name_raw = artifact.get("file_name")
    if isinstance(file_name_raw, str) and file_name_raw.strip():
        return _sanitize_download_filename(file_name_raw)
    title_raw = artifact.get("title")
    if not isinstance(title_raw, str):
        return None
    normalized = " ".join(title_raw.replace("`", " ").split())
    if not normalized:
        return None
    return normalized[:80]


def _stream_preview_indicates_canvas(preview_text: str) -> bool:
    return _streaming._stream_preview_indicates_canvas(
        preview_text,
        canvas_char_threshold=CANVAS_CHAR_THRESHOLD,
        canvas_code_line_threshold=CANVAS_CODE_LINE_THRESHOLD,
        extract_named_files_from_output_fn=_extract_named_files_from_output,
        extract_named_file_markers_fn=_extract_named_file_markers,
    )


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


def _parse_imported_message(raw: object) -> dict[str, JSONValue] | None:
    if not isinstance(raw, dict):
        return None
    message_id_raw = raw.get("message_id")
    role_raw = raw.get("role")
    content_raw = raw.get("content")
    created_at_raw = raw.get("created_at")
    trace_id_raw = raw.get("trace_id")
    parent_raw = raw.get("parent_user_message_id")
    attachments_raw = raw.get("attachments")
    if not isinstance(message_id_raw, str) or not message_id_raw.strip():
        return None
    if not isinstance(role_raw, str) or role_raw not in {"user", "assistant", "system"}:
        return None
    if not isinstance(content_raw, str):
        return None
    if not isinstance(created_at_raw, str) or not created_at_raw.strip():
        return None
    if trace_id_raw is not None and (not isinstance(trace_id_raw, str) or not trace_id_raw.strip()):
        return None
    if parent_raw is not None and (not isinstance(parent_raw, str) or not parent_raw.strip()):
        return None
    trace_id: str | None = trace_id_raw.strip() if isinstance(trace_id_raw, str) else None
    parent_user_message_id: str | None = parent_raw.strip() if isinstance(parent_raw, str) else None
    attachments: list[dict[str, str]] = []
    if attachments_raw is None:
        attachments = []
    elif isinstance(attachments_raw, list):
        for item in attachments_raw:
            if not isinstance(item, dict):
                return None
            name_raw = item.get("name")
            mime_raw = item.get("mime")
            content_item_raw = item.get("content")
            if (
                not isinstance(name_raw, str)
                or not name_raw.strip()
                or not isinstance(mime_raw, str)
                or not mime_raw.strip()
                or not isinstance(content_item_raw, str)
            ):
                return None
            attachments.append(
                {
                    "name": name_raw.strip(),
                    "mime": mime_raw.strip(),
                    "content": content_item_raw,
                },
            )
    else:
        return None
    if role_raw != "assistant":
        trace_id = None
        parent_user_message_id = None
    if role_raw != "user":
        attachments = []
    return {
        "message_id": message_id_raw.strip(),
        "role": role_raw,
        "content": content_raw,
        "created_at": created_at_raw.strip(),
        "trace_id": trace_id,
        "parent_user_message_id": parent_user_message_id,
        "attachments": attachments,
    }


def _parse_imported_session(raw: object, *, principal_id: str) -> PersistedSession | None:
    if not isinstance(raw, dict):
        return None
    session_id_raw = raw.get("session_id")
    if not isinstance(session_id_raw, str) or not session_id_raw.strip():
        return None
    session_id = session_id_raw.strip()
    created_at_raw = raw.get("created_at")
    updated_at_raw = raw.get("updated_at")
    created_at = (
        created_at_raw if isinstance(created_at_raw, str) and created_at_raw.strip() else _utc_iso()
    )
    updated_at = (
        updated_at_raw if isinstance(updated_at_raw, str) and updated_at_raw.strip() else created_at
    )
    messages_raw = raw.get("messages")
    if not isinstance(messages_raw, list):
        return None
    messages: list[dict[str, JSONValue]] = []
    for item in messages_raw:
        parsed_message = _parse_imported_message(item)
        if parsed_message is None:
            return None
        messages.append(parsed_message)
    policy_raw = raw.get("policy")
    policy = policy_raw if isinstance(policy_raw, dict) else {}
    policy_profile = _normalize_policy_profile(policy.get("profile"))
    yolo_armed = policy.get("yolo_armed") is True
    yolo_armed_at_raw = policy.get("yolo_armed_at")
    yolo_armed_at = (
        yolo_armed_at_raw.strip()
        if isinstance(yolo_armed_at_raw, str) and yolo_armed_at_raw.strip()
        else None
    )
    workspace_root_raw = raw.get("workspace_root")
    workspace_root = (
        workspace_root_raw.strip()
        if isinstance(workspace_root_raw, str) and workspace_root_raw.strip()
        else None
    )
    return PersistedSession(
        session_id=session_id,
        principal_id=principal_id,
        created_at=created_at,
        updated_at=updated_at,
        status="ok",
        decision=None,
        messages=messages,
        workspace_root=workspace_root,
        policy_profile=policy_profile,
        yolo_armed=yolo_armed,
        yolo_armed_at=yolo_armed_at,
        tools_state=_normalize_tools_state_payload(raw.get("tools_state")),
    )


def _serialize_persisted_session(session: PersistedSession) -> dict[str, JSONValue]:
    selected_model: dict[str, JSONValue] | None = None
    if session.model_provider and session.model_id:
        selected_model = {
            "provider": session.model_provider,
            "model": session.model_id,
        }
    payload: dict[str, JSONValue] = {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "status": session.status,
        "messages": [dict(item) for item in session.messages],
        "output": {
            "content": session.output_text,
            "updated_at": session.output_updated_at,
        },
        "files": list(session.files),
        "decision": dict(session.decision) if session.decision is not None else None,
        "selected_model": selected_model,
        "title_override": session.title_override,
        "folder_id": session.folder_id,
        "workspace_root": session.workspace_root,
        "policy": {
            "profile": _normalize_policy_profile(session.policy_profile),
            "yolo_armed": bool(session.yolo_armed),
            "yolo_armed_at": session.yolo_armed_at,
        },
        "tools_state": dict(session.tools_state) if isinstance(session.tools_state, dict) else None,
        "mode": _normalize_mode_value(session.mode, default="ask"),
        "active_plan": (
            _normalize_plan_payload(session.active_plan)
            if session.active_plan is not None
            else None
        ),
        "active_task": (
            _normalize_task_payload(session.active_task)
            if session.active_task is not None
            else None
        ),
    }
    return payload


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


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
    started = time.monotonic()
    audit_entries: list[dict[str, JSONValue]] = []
    read_files = 0
    total_bytes = 0
    search_calls = 0
    for current_root, dirs, files in os.walk(root):
        if time.monotonic() - started > PLAN_AUDIT_TIMEOUT_SECONDS:
            break
        dirs[:] = [name for name in dirs if name not in WORKSPACE_INDEX_IGNORED_DIRS]
        current = Path(current_root)
        for filename in files:
            if read_files >= PLAN_AUDIT_MAX_READ_FILES:
                break
            if time.monotonic() - started > PLAN_AUDIT_TIMEOUT_SECONDS:
                break
            if filename.endswith(".sqlite"):
                continue
            full_path = current / filename
            suffix = full_path.suffix.lower()
            if suffix and suffix not in WORKSPACE_INDEX_ALLOWED_EXTENSIONS:
                continue
            try:
                raw = full_path.read_bytes()
            except Exception:  # noqa: BLE001
                continue
            if not raw:
                continue
            next_size = min(len(raw), 4000)
            if total_bytes + next_size > PLAN_AUDIT_MAX_TOTAL_BYTES:
                break
            preview = raw[:next_size].decode("utf-8", errors="ignore")
            rel_path = str(full_path.relative_to(root))
            audit_entries.append(
                {
                    "kind": "read_file",
                    "path": rel_path,
                    "bytes": next_size,
                    "preview": preview[:240],
                }
            )
            total_bytes += next_size
            read_files += 1
        if read_files >= PLAN_AUDIT_MAX_READ_FILES or total_bytes >= PLAN_AUDIT_MAX_TOTAL_BYTES:
            break
    return audit_entries, {
        "read_files": read_files,
        "total_bytes": total_bytes,
        "search_calls": search_calls,
    }


def _build_default_plan_steps() -> list[dict[str, JSONValue]]:
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


def _build_plan_draft(
    *,
    goal: str,
    audit_log: list[dict[str, JSONValue]],
) -> dict[str, JSONValue]:
    now = _utc_now_iso()
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
        "steps": _build_default_plan_steps(),
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
    plan["plan_hash"] = _plan_hash_payload(plan)
    return plan


def _plan_with_status(
    plan: dict[str, JSONValue],
    *,
    status: str,
) -> dict[str, JSONValue]:
    updated = dict(plan)
    updated["status"] = status
    updated["updated_at"] = _utc_now_iso()
    if status == "approved":
        updated["approved_at"] = _utc_now_iso()
        updated["approved_by"] = "user"
    elif status == "draft":
        updated["approved_at"] = None
        updated["approved_by"] = None
    _increment_plan_revision(updated)
    updated["plan_hash"] = _plan_hash_payload(updated)
    return updated


def _task_with_status(
    task: dict[str, JSONValue],
    *,
    status: str,
    current_step_id: str | None = None,
) -> dict[str, JSONValue]:
    updated = dict(task)
    updated["status"] = status
    updated["current_step_id"] = current_step_id
    updated["updated_at"] = _utc_now_iso()
    return updated


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
    hub: UIHub = app["ui_hub"]
    while True:
        workflow = await hub.get_session_workflow(session_id)
        plan = _normalize_plan_payload(workflow.get("active_plan"))
        task = _normalize_task_payload(workflow.get("active_task"))
        mode = _normalize_mode_value(workflow.get("mode"), default="ask")
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
            next_step = _find_next_todo_step(plan)
            if next_step is None:
                completed_plan = _plan_with_status(plan, status="completed")
                completed_task = _task_with_status(task, status="completed", current_step_id=None)
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
                failed_plan = _plan_with_status(plan, status="failed")
                failed_task = _task_with_status(task, status="failed", current_step_id=None)
                await hub.set_session_workflow(
                    session_id,
                    mode="act",
                    active_plan=failed_plan,
                    active_task=failed_task,
                )
                return
            plan = _plan_mark_step(plan, step_id=step_id, status="doing")
            task = _task_with_status(task, status="running", current_step_id=step_id)
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
            "completed_at": _utc_now_iso(),
        }
        plan = _plan_mark_step(plan, step_id=current_step_id, status="done", evidence=evidence)
        task = _task_with_status(task, status="running", current_step_id=None)
        await hub.set_session_workflow(
            session_id,
            mode="act",
            active_plan=plan,
            active_task=task,
        )
        await asyncio.sleep(0)


def create_app(
    *,
    agent: AgentProtocol | None = None,
    max_request_bytes: int | None = None,
    ui_storage: UISessionStorage | None = None,
    auth_config: HttpAuthConfig | None = None,
) -> web.Application:
    config_max_bytes = max_request_bytes or DEFAULT_MAX_REQUEST_BYTES
    resolved_auth_config = auth_config or resolve_http_auth_config()
    app = web.Application(
        client_max_size=config_max_bytes,
        middlewares=[auth_gate_middleware],
    )
    app["auth_config"] = resolved_auth_config
    app["http_api_logger"] = logger
    app["settings_snapshot_builder"] = _build_settings_payload
    if agent is None:

        def _factory() -> AgentProtocol:
            module = importlib.import_module("core.agent")
            agent_factory = getattr(module, "Agent", None)
            if not callable(agent_factory):
                raise RuntimeError("Agent class not found in core.agent")
            return cast(AgentProtocol, agent_factory())

        app["agent"] = None
        app["agent_provider"] = LazyAgentProvider(factory=_factory)
    else:
        app["agent"] = agent
        app["agent_provider"] = LazyAgentProvider.from_instance(agent)
    app["agent_lock"] = asyncio.Lock()
    app["session_store"] = SessionApprovalStore()
    resolved_ui_storage = ui_storage or SQLiteUISessionStorage(
        Path(__file__).resolve().parent.parent / ".run" / "ui_sessions.db",
    )
    app["ui_hub"] = UIHub(storage=resolved_ui_storage)
    dist_path = Path(__file__).resolve().parent.parent / "ui" / "dist"
    app["ui_dist_path"] = dist_path
    from server.http.routes import register_routes

    register_routes(app)
    assets_path = dist_path / "assets"
    if assets_path.exists():
        app.router.add_static("/ui/assets/", assets_path)
    else:
        logger.warning("UI assets directory missing at %s; skipping static assets.", assets_path)
    return app


def run_server(config: HttpServerConfig) -> None:
    ensure_http_auth_boot_config()
    app = create_app(max_request_bytes=config.max_request_bytes)
    web.run_app(app, host=config.host, port=config.port)


def main() -> None:
    config = resolve_http_server_config()
    run_server(config)


__all__ = ["create_app", "main", "run_server"]
