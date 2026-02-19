from __future__ import annotations

import asyncio
import difflib
import importlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
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
from config.ui_embeddings_settings import (
    UIEmbeddingsSettings,
    load_ui_embeddings_settings,
    save_ui_embeddings_settings,
)
from core.approval_policy import (
    ALL_CATEGORIES,
    ApprovalCategory,
    ApprovalRequest,
)
from core.mwv.models import MWV_REPORT_PREFIX
from core.tracer import TRACE_LOG, TraceRecord
from llm.local_http_brain import DEFAULT_LOCAL_ENDPOINT
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
from tools.tool_logger import DEFAULT_LOG_PATH as TOOL_CALLS_LOG
from tools.workspace_tools import (
    WORKSPACE_ROOT as DEFAULT_WORKSPACE_ROOT,
)

ChatRequest = _chat_request.ChatRequest
_is_sampling_key = _chat_request._is_sampling_key
_validate_messages = _chat_request._validate_messages
_parse_chat_request = _chat_request._parse_chat_request

TOOL_PIPELINE_ENABLED: Final[bool] = False
_CATEGORY_MAP: Final[dict[str, ApprovalCategory]] = {item: item for item in ALL_CATEGORIES}

logger = logging.getLogger("SlavikAI.HttpAPI")

_parse_github_import_args = github_import.parse_github_import_args
_build_github_import_approval_request = github_import.build_github_import_approval_request
_resolve_github_target = github_import.resolve_github_target
_clone_github_repository = github_import.clone_github_repository
_index_imported_project = github_import.index_imported_project

UI_SESSION_HEADER: Final[str] = "X-Slavik-Session"
SUPPORTED_MODEL_PROVIDERS: Final[set[str]] = {"xai", "openrouter", "local"}
API_KEY_SETTINGS_PROVIDERS: Final[set[str]] = {"xai", "openrouter", "local", "openai"}
PROVIDER_API_KEY_ENV: Final[dict[str, str]] = {
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "local": "LOCAL_LLM_API_KEY",
    "openai": "OPENAI_API_KEY",
}
XAI_MODELS_ENDPOINT: Final[str] = "https://api.x.ai/v1/models"
OPENROUTER_MODELS_ENDPOINT: Final[str] = "https://openrouter.ai/api/v1/models"
OPENAI_STT_ENDPOINT: Final[str] = "https://api.openai.com/v1/audio/transcriptions"
MODEL_FETCH_TIMEOUT: Final[int] = 20
UI_PROJECT_COMMANDS: Final[set[str]] = {"find", "index", "github_import"}
UI_SETTINGS_PATH: Final[Path] = Path(__file__).resolve().parent.parent / ".run" / "ui_settings.json"
DEFAULT_UI_TONE: Final[str] = "balanced"
INDEX_ENABLED_ENV: Final[str] = "SLAVIK_INDEX_ENABLED"
DEFAULT_LONG_PASTE_TO_FILE_ENABLED: Final[bool] = True
DEFAULT_LONG_PASTE_THRESHOLD_CHARS: Final[int] = 12_000
MIN_LONG_PASTE_THRESHOLD_CHARS: Final[int] = 1_000
MAX_LONG_PASTE_THRESHOLD_CHARS: Final[int] = 80_000
UI_GITHUB_REQUIRED_CATEGORIES: Final[list[ApprovalCategory]] = ["NETWORK_RISK", "EXEC_ARBITRARY"]
CANVAS_LINE_THRESHOLD: Final[int] = 40
CANVAS_CHAR_THRESHOLD: Final[int] = 1800
CANVAS_CODE_LINE_THRESHOLD: Final[int] = 28
CANVAS_DOCUMENT_LINE_THRESHOLD: Final[int] = 24
CHAT_STREAM_CHUNK_SIZE: Final[int] = 80
CHAT_STREAM_WARMUP_CHARS: Final[int] = 220
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
POLICY_PROFILES: Final[set[str]] = {"sandbox", "index", "yolo"}
DEFAULT_POLICY_PROFILE: Final[str] = "sandbox"
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
UI_SETTINGS_USER_ALLOWED_TOP_LEVEL_KEYS: Final[set[str]] = {
    "personalization",
    "composer",
    "memory",
    "providers",
}
UI_SETTINGS_CONTROL_TOP_LEVEL_KEYS: Final[set[str]] = {
    "tools",
    "policy",
    "risk",
    "security",
    "risk_categories",
    "security_categories",
    "approval_categories",
    "approved_categories",
    "safe_mode",
}
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


@dataclass(frozen=True)
class TraceGroup:
    start_ts: str
    end_ts: str | None
    interaction_id: str | None
    events: list[TraceRecord]


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


def _normalize_provider(raw_provider: str) -> str | None:
    normalized = raw_provider.strip().lower()
    if normalized in SUPPORTED_MODEL_PROVIDERS:
        return normalized
    return None


def _build_model_config(provider: str, model_id: str) -> ModelConfig:
    if provider == "xai":
        return ModelConfig(provider="xai", model=model_id)
    if provider == "openrouter":
        return ModelConfig(provider="openrouter", model=model_id)
    if provider == "local":
        return ModelConfig(provider="local", model=model_id)
    raise ValueError(f"Неизвестный провайдер: {provider}")


def _closest_model_suggestion(model_id: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    matches = difflib.get_close_matches(model_id, candidates, n=1, cutoff=0.4)
    if not matches:
        return None
    return matches[0]


def _local_models_endpoint() -> str:
    base_url = os.getenv("LOCAL_LLM_URL", DEFAULT_LOCAL_ENDPOINT).strip()
    if not base_url:
        base_url = DEFAULT_LOCAL_ENDPOINT
    if base_url.endswith("/chat/completions"):
        return f"{base_url.removesuffix('/chat/completions')}/models"
    return f"{base_url.rstrip('/')}/models"


def _provider_models_endpoint(provider: str) -> str:
    if provider == "xai":
        return XAI_MODELS_ENDPOINT
    if provider == "openrouter":
        return OPENROUTER_MODELS_ENDPOINT
    if provider == "local":
        return _local_models_endpoint()
    raise ValueError(f"Неизвестный провайдер: {provider}")


def _provider_auth_headers(provider: str) -> tuple[dict[str, str], str | None]:
    if provider == "xai":
        api_key = _resolve_provider_api_key("xai")
        if not api_key:
            return {}, "Не задан XAI_API_KEY (env или UI settings)."
        return {"Authorization": f"Bearer {api_key}"}, None
    if provider == "openrouter":
        api_key = _resolve_provider_api_key("openrouter")
        if not api_key:
            return {}, "Не задан OPENROUTER_API_KEY (env или UI settings)."
        return {"Authorization": f"Bearer {api_key}"}, None
    if provider == "local":
        api_key = _resolve_provider_api_key("local")
        if not api_key:
            return {}, None
        return {"Authorization": f"Bearer {api_key}"}, None
    return {}, f"Неизвестный провайдер: {provider}"


def _parse_models_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            models.append(model_id.strip())
    # dedupe, preserve order
    unique: list[str] = []
    for model_id in models:
        if model_id not in unique:
            unique.append(model_id)
    return unique


def _fetch_provider_models(provider: str) -> tuple[list[str], str | None]:
    try:
        url = _provider_models_endpoint(provider)
    except ValueError as exc:
        return [], str(exc)
    headers, auth_error = _provider_auth_headers(provider)
    if auth_error:
        return [], auth_error
    try:
        response = requests.get(url, headers=headers, timeout=MODEL_FETCH_TIMEOUT)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        if provider == "local":
            fallback = os.getenv("LOCAL_LLM_DEFAULT_MODEL", "local-default").strip()
            if fallback:
                return [fallback], None
        return [], f"Не удалось получить список моделей провайдера {provider}: {exc}"
    models = _parse_models_payload(response.json())
    if not models and provider == "local":
        fallback = os.getenv("LOCAL_LLM_DEFAULT_MODEL", "local-default").strip()
        if fallback:
            models = [fallback]
    return models, None


def _parse_trace_log(path: Path) -> list[TraceRecord]:
    if not path.exists():
        return []
    records: list[TraceRecord] = []
    required_keys = {"timestamp", "event", "message"}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = safe_json_loads(line)
            if not isinstance(data, dict):
                continue
            if not required_keys.issubset(data):
                continue
            meta = data.get("meta")
            if meta is not None and not isinstance(meta, dict):
                continue
            record: TraceRecord = {
                "timestamp": str(data.get("timestamp")),
                "event": str(data.get("event")),
                "message": str(data.get("message")),
                "meta": meta or {},
            }
            records.append(record)
    return records


def _trace_log_path() -> Path:
    return TRACE_LOG


def _requests_module() -> RequestsModuleProtocol:
    return requests


def _load_memory_config_runtime() -> MemoryConfig:
    return load_memory_config()


def _save_memory_config_runtime(config: MemoryConfig) -> None:
    save_memory_config(config)


def _build_trace_groups(records: list[TraceRecord]) -> list[TraceGroup]:
    groups: list[TraceGroup] = []
    current_events: list[TraceRecord] = []
    current_start: str | None = None
    for record in records:
        if record.get("event") == "user_input":
            if current_start is not None:
                groups.append(
                    TraceGroup(
                        start_ts=current_start,
                        end_ts=None,
                        interaction_id=_extract_interaction_id(current_events),
                        events=current_events,
                    )
                )
            current_start = str(record.get("timestamp", ""))
            current_events = [record]
            continue
        if current_start is not None:
            current_events.append(record)
    if current_start is not None:
        groups.append(
            TraceGroup(
                start_ts=current_start,
                end_ts=None,
                interaction_id=_extract_interaction_id(current_events),
                events=current_events,
            )
        )

    for idx, group in enumerate(groups):
        if idx + 1 < len(groups):
            next_start = groups[idx + 1].start_ts
            groups[idx] = TraceGroup(
                start_ts=group.start_ts,
                end_ts=next_start,
                interaction_id=group.interaction_id,
                events=group.events,
            )
        else:
            groups[idx] = TraceGroup(
                start_ts=group.start_ts,
                end_ts=_last_event_timestamp(group.events),
                interaction_id=group.interaction_id,
                events=group.events,
            )
    return groups


def _extract_interaction_id(events: list[TraceRecord]) -> str | None:
    for record in events:
        if record.get("event") != "interaction_logged":
            continue
        meta = record.get("meta")
        if not isinstance(meta, dict):
            continue
        raw = meta.get("interaction_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _last_event_timestamp(events: list[TraceRecord]) -> str | None:
    for record in reversed(events):
        ts = record.get("timestamp")
        if isinstance(ts, str) and ts.strip():
            return ts
    return None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _filter_tool_calls(
    *,
    path: Path,
    start_ts: str | None,
    end_ts: str | None,
) -> list[dict[str, JSONValue]]:
    if not path.exists():
        return []
    start_dt = _parse_timestamp(start_ts) if start_ts else None
    end_dt = _parse_timestamp(end_ts) if end_ts else None
    results: list[dict[str, JSONValue]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = safe_json_loads(line)
            if not isinstance(data, dict):
                continue
            ts_raw = data.get("timestamp")
            if not isinstance(ts_raw, str):
                continue
            ts_dt = _parse_timestamp(ts_raw)
            if ts_dt is None:
                continue
            if start_dt and ts_dt < start_dt:
                continue
            if end_dt and ts_dt > end_dt:
                continue
            results.append(
                {
                    "timestamp": ts_raw,
                    "tool": str(data.get("tool") or ""),
                    "ok": bool(data.get("ok")),
                    "error": data.get("error"),
                    "args": (data.get("args") if isinstance(data.get("args"), dict) else {}),
                    "meta": (data.get("meta") if isinstance(data.get("meta"), dict) else {}),
                }
            )
    return results


def _tool_calls_for_trace_id(trace_id: str) -> list[dict[str, JSONValue]] | None:
    records = _parse_trace_log(TRACE_LOG)
    groups = _build_trace_groups(records)
    target: TraceGroup | None = None
    for group in groups:
        if group.interaction_id == trace_id:
            target = group
            break
    if target is None:
        return None
    return _filter_tool_calls(
        path=TOOL_CALLS_LOG,
        start_ts=target.start_ts,
        end_ts=target.end_ts,
    )


def _normalize_logged_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _extract_paths_from_tool_call(
    *,
    tool: str,
    args: dict[str, JSONValue],
) -> list[str]:
    if tool in {"workspace_write", "workspace_patch"}:
        path = _normalize_logged_path(args.get("path"))
        return [path] if path is not None else []

    if tool == "fs":
        op_raw = args.get("op")
        op = op_raw.strip().lower() if isinstance(op_raw, str) else ""
        if op != "write":
            return []
        path = _normalize_logged_path(args.get("path"))
        return [path] if path is not None else []

    return []


def _extract_files_from_tool_calls(tool_calls: list[dict[str, JSONValue]]) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for call in tool_calls:
        tool_raw = call.get("tool")
        if not isinstance(tool_raw, str):
            continue
        ok_raw = call.get("ok")
        if not isinstance(ok_raw, bool) or not ok_raw:
            continue
        args_raw = call.get("args")
        args: dict[str, JSONValue] = {}
        if isinstance(args_raw, dict):
            for key, value in args_raw.items():
                args[str(key)] = _normalize_json_value(value)
        for path in _extract_paths_from_tool_call(tool=tool_raw, args=args):
            if path in seen:
                continue
            seen.add(path)
            files.append(path)
    return files


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


def _split_chat_stream_chunks(content: str) -> list[str]:
    if not content:
        return []
    return [
        content[idx : idx + CHAT_STREAM_CHUNK_SIZE]
        for idx in range(0, len(content), CHAT_STREAM_CHUNK_SIZE)
    ]


def _stream_preview_indicates_canvas(preview_text: str) -> bool:
    normalized = preview_text.strip()
    if not normalized:
        return False
    if len(normalized) >= CANVAS_CHAR_THRESHOLD:
        return True
    if len(normalized.splitlines()) >= CANVAS_CODE_LINE_THRESHOLD:
        return True
    if _extract_named_files_from_output(normalized):
        return True
    tail = normalized[-320:]
    if "```" in tail and _extract_named_file_markers(tail):
        return True
    return False


def _stream_preview_ready_for_chat(preview_text: str) -> bool:
    normalized = preview_text.strip()
    if not normalized:
        return False
    if len(normalized) >= CHAT_STREAM_WARMUP_CHARS:
        return True
    if len(normalized) >= 96 and "```" not in normalized and normalized.count("\n") <= 1:
        return True
    return False


async def _publish_chat_stream_start(
    hub: UIHub,
    *,
    session_id: str,
    stream_id: str,
) -> None:
    await hub.publish(
        session_id,
        {
            "type": "chat.stream.start",
            "payload": {
                "session_id": session_id,
                "stream_id": stream_id,
            },
        },
    )


async def _publish_chat_stream_delta(
    hub: UIHub,
    *,
    session_id: str,
    stream_id: str,
    delta: str,
) -> None:
    if not delta:
        return
    await hub.publish(
        session_id,
        {
            "type": "chat.stream.delta",
            "payload": {
                "session_id": session_id,
                "stream_id": stream_id,
                "delta": delta,
            },
        },
    )


async def _publish_chat_stream_done(
    hub: UIHub,
    *,
    session_id: str,
    stream_id: str,
) -> None:
    await hub.publish(
        session_id,
        {
            "type": "chat.stream.done",
            "payload": {
                "session_id": session_id,
                "stream_id": stream_id,
            },
        },
    )


async def _publish_chat_stream_from_text(
    hub: UIHub,
    *,
    session_id: str,
    stream_id: str,
    content: str,
) -> None:
    await _publish_chat_stream_start(hub, session_id=session_id, stream_id=stream_id)
    for chunk in _split_chat_stream_chunks(content):
        await _publish_chat_stream_delta(
            hub,
            session_id=session_id,
            stream_id=stream_id,
            delta=chunk,
        )
        await asyncio.sleep(0.01)
    await _publish_chat_stream_done(hub, session_id=session_id, stream_id=stream_id)


def _split_canvas_stream_chunks(content: str) -> list[str]:
    if not content:
        return []
    lines = content.splitlines(keepends=True)
    if len(lines) <= 2:
        chunk_size = 120
        return [content[idx : idx + chunk_size] for idx in range(0, len(content), chunk_size)]
    chunks: list[str] = []
    lines_per_chunk = 4
    for start in range(0, len(lines), lines_per_chunk):
        chunks.append("".join(lines[start : start + lines_per_chunk]))
    return [chunk for chunk in chunks if chunk]


async def _publish_canvas_stream(
    hub: UIHub,
    *,
    session_id: str,
    artifact_id: str,
    content: str,
) -> None:
    await hub.publish(
        session_id,
        {
            "type": "canvas.stream.start",
            "payload": {
                "session_id": session_id,
                "artifact_id": artifact_id,
            },
        },
    )
    chunks = _split_canvas_stream_chunks(content)
    if not chunks:
        await hub.publish(
            session_id,
            {
                "type": "canvas.stream.done",
                "payload": {
                    "session_id": session_id,
                    "artifact_id": artifact_id,
                },
            },
        )
        return
    for delta in chunks:
        await hub.publish(
            session_id,
            {
                "type": "canvas.stream.delta",
                "payload": {
                    "session_id": session_id,
                    "artifact_id": artifact_id,
                    "delta": delta,
                },
            },
        )
        await asyncio.sleep(0.02)
    await hub.publish(
        session_id,
        {
            "type": "canvas.stream.done",
            "payload": {
                "session_id": session_id,
                "artifact_id": artifact_id,
            },
        },
    )


def _load_ui_settings_blob() -> dict[str, object]:
    if not UI_SETTINGS_PATH.exists():
        return {}
    try:
        raw = json.loads(UI_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_ui_settings_blob(payload: dict[str, object]) -> None:
    UI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_SETTINGS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_personalization_settings() -> tuple[str, str]:
    payload = _load_ui_settings_blob()
    personalization_raw = payload.get("personalization")
    if not isinstance(personalization_raw, dict):
        return DEFAULT_UI_TONE, ""
    tone_raw = personalization_raw.get("tone")
    prompt_raw = personalization_raw.get("system_prompt")
    tone = tone_raw.strip() if isinstance(tone_raw, str) and tone_raw.strip() else DEFAULT_UI_TONE
    system_prompt = prompt_raw if isinstance(prompt_raw, str) else ""
    return tone, system_prompt


def _save_personalization_settings(*, tone: str, system_prompt: str) -> None:
    payload = _load_ui_settings_blob()
    payload["personalization"] = {
        "tone": tone.strip() or DEFAULT_UI_TONE,
        "system_prompt": system_prompt,
    }
    _save_ui_settings_blob(payload)


def _load_embeddings_settings() -> UIEmbeddingsSettings:
    return load_ui_embeddings_settings(path=UI_SETTINGS_PATH)


def _save_embeddings_settings(settings: UIEmbeddingsSettings) -> None:
    save_ui_embeddings_settings(settings, path=UI_SETTINGS_PATH)


def _load_composer_settings() -> tuple[bool, int]:
    payload = _load_ui_settings_blob()
    composer_raw = payload.get("composer")
    if not isinstance(composer_raw, dict):
        return DEFAULT_LONG_PASTE_TO_FILE_ENABLED, DEFAULT_LONG_PASTE_THRESHOLD_CHARS
    enabled_raw = composer_raw.get("long_paste_to_file_enabled")
    threshold_raw = composer_raw.get("long_paste_threshold_chars")
    enabled = enabled_raw if isinstance(enabled_raw, bool) else DEFAULT_LONG_PASTE_TO_FILE_ENABLED
    threshold = (
        threshold_raw
        if isinstance(threshold_raw, int) and not isinstance(threshold_raw, bool)
        else DEFAULT_LONG_PASTE_THRESHOLD_CHARS
    )
    if threshold < MIN_LONG_PASTE_THRESHOLD_CHARS:
        threshold = MIN_LONG_PASTE_THRESHOLD_CHARS
    if threshold > MAX_LONG_PASTE_THRESHOLD_CHARS:
        threshold = MAX_LONG_PASTE_THRESHOLD_CHARS
    return enabled, threshold


def _save_composer_settings(
    *,
    long_paste_to_file_enabled: bool,
    long_paste_threshold_chars: int,
) -> None:
    payload = _load_ui_settings_blob()
    payload["composer"] = {
        "long_paste_to_file_enabled": long_paste_to_file_enabled,
        "long_paste_threshold_chars": long_paste_threshold_chars,
    }
    _save_ui_settings_blob(payload)


def _user_plane_forbidden_settings_key(payload: dict[str, object]) -> str | None:
    invalid = sorted(
        {
            normalized
            for key in payload
            if (
                (normalized := str(key).strip()) in UI_SETTINGS_CONTROL_TOP_LEVEL_KEYS
                or normalized not in UI_SETTINGS_USER_ALLOWED_TOP_LEVEL_KEYS
            )
        }
    )
    if not invalid:
        return None
    return invalid[0]


def _normalize_policy_profile(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in POLICY_PROFILES:
            return normalized
    return DEFAULT_POLICY_PROFILE


def _load_policy_settings() -> tuple[str, bool, str | None]:
    payload = _load_ui_settings_blob()
    policy_raw = payload.get("policy")
    if not isinstance(policy_raw, dict):
        return DEFAULT_POLICY_PROFILE, False, None
    profile = _normalize_policy_profile(policy_raw.get("profile"))
    yolo_armed_raw = policy_raw.get("yolo_armed")
    yolo_armed = yolo_armed_raw if isinstance(yolo_armed_raw, bool) else False
    yolo_armed_at_raw = policy_raw.get("yolo_armed_at")
    yolo_armed_at = (
        yolo_armed_at_raw.strip()
        if isinstance(yolo_armed_at_raw, str) and yolo_armed_at_raw.strip()
        else None
    )
    if not yolo_armed:
        yolo_armed_at = None
    return profile, yolo_armed, yolo_armed_at


def _save_policy_settings(
    *,
    profile: str,
    yolo_armed: bool,
    yolo_armed_at: str | None,
) -> None:
    payload = _load_ui_settings_blob()
    payload["policy"] = {
        "profile": _normalize_policy_profile(profile),
        "yolo_armed": yolo_armed,
        "yolo_armed_at": yolo_armed_at if yolo_armed else None,
    }
    _save_ui_settings_blob(payload)


def _tools_state_for_profile(profile: str, current: dict[str, bool]) -> dict[str, bool]:
    normalized = _normalize_policy_profile(profile)
    if normalized == "sandbox":
        return {
            **current,
            "fs": True,
            "project": False,
            "shell": False,
            "web": False,
            "safe_mode": True,
        }
    if normalized == "index":
        return {
            **current,
            "fs": True,
            "project": True,
            "shell": False,
            "web": False,
            "safe_mode": True,
        }
    # yolo keeps current tool preferences, but still never bypasses safe_mode by default.
    return {
        **current,
        "safe_mode": True,
    }


def _load_tools_state() -> dict[str, bool]:
    try:
        return load_tools_config().to_dict()
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_TOOLS_STATE)


def _save_tools_state(state: dict[str, bool]) -> None:
    payload: dict[str, object] = {key: value for key, value in state.items()}
    save_tools_config(ToolsConfig.from_dict(payload))


def _load_provider_api_keys() -> dict[str, str]:
    payload = _load_ui_settings_blob()
    providers_raw = payload.get("providers")
    if not isinstance(providers_raw, dict):
        return {}
    api_keys: dict[str, str] = {}
    for provider in API_KEY_SETTINGS_PROVIDERS:
        entry = providers_raw.get(provider)
        key_raw: object | None = None
        if isinstance(entry, dict):
            key_raw = entry.get("api_key")
        elif isinstance(entry, str):
            key_raw = entry
        if isinstance(key_raw, str):
            normalized = key_raw.strip()
            if normalized:
                api_keys[provider] = normalized
    return api_keys


def _save_provider_api_keys(api_keys: dict[str, str]) -> None:
    payload = _load_ui_settings_blob()
    providers_payload: dict[str, object] = {}
    for provider in sorted(API_KEY_SETTINGS_PROVIDERS):
        key_raw = api_keys.get(provider)
        if not isinstance(key_raw, str):
            continue
        normalized = key_raw.strip()
        if normalized:
            providers_payload[provider] = {"api_key": normalized}
    if providers_payload:
        payload["providers"] = providers_payload
    else:
        payload.pop("providers", None)
    _save_ui_settings_blob(payload)


def _load_provider_runtime_checks() -> dict[str, dict[str, JSONValue]]:
    payload = _load_ui_settings_blob()
    checks_raw = payload.get("provider_runtime_checks")
    if not isinstance(checks_raw, dict):
        return {}
    checks: dict[str, dict[str, JSONValue]] = {}
    for provider in API_KEY_SETTINGS_PROVIDERS:
        item_raw = checks_raw.get(provider)
        if not isinstance(item_raw, dict):
            continue
        valid_raw = item_raw.get("api_key_valid")
        error_raw = item_raw.get("last_check_error")
        checked_at_raw = item_raw.get("last_checked_at")
        checks[provider] = {
            "api_key_valid": valid_raw if isinstance(valid_raw, bool) else None,
            "last_check_error": error_raw if isinstance(error_raw, str) else None,
            "last_checked_at": checked_at_raw if isinstance(checked_at_raw, str) else None,
        }
    return checks


def _save_provider_runtime_checks(
    checks: dict[str, dict[str, JSONValue]],
) -> None:
    payload = _load_ui_settings_blob()
    serialized: dict[str, object] = {}
    for provider in sorted(API_KEY_SETTINGS_PROVIDERS):
        item = checks.get(provider)
        if not isinstance(item, dict):
            continue
        valid_raw = item.get("api_key_valid")
        error_raw = item.get("last_check_error")
        checked_at_raw = item.get("last_checked_at")
        serialized[provider] = {
            "api_key_valid": valid_raw if isinstance(valid_raw, bool) else None,
            "last_check_error": error_raw if isinstance(error_raw, str) else None,
            "last_checked_at": checked_at_raw if isinstance(checked_at_raw, str) else None,
        }
    if serialized:
        payload["provider_runtime_checks"] = serialized
    else:
        payload.pop("provider_runtime_checks", None)
    _save_ui_settings_blob(payload)


def _load_provider_env_api_key(provider: str) -> str | None:
    env_name = PROVIDER_API_KEY_ENV.get(provider)
    if env_name is None:
        return None
    key_raw = os.getenv(env_name, "")
    normalized = key_raw.strip()
    return normalized or None


def _resolve_provider_api_key(
    provider: str,
    *,
    settings_api_keys: dict[str, str] | None = None,
) -> str | None:
    saved = settings_api_keys if settings_api_keys is not None else _load_provider_api_keys()
    from_settings = saved.get(provider, "").strip()
    if from_settings:
        return from_settings
    return _load_provider_env_api_key(provider)


def _provider_api_key_source(
    provider: str,
    *,
    settings_api_keys: dict[str, str] | None = None,
) -> Literal["settings", "env", "missing"]:
    saved = settings_api_keys if settings_api_keys is not None else _load_provider_api_keys()
    from_settings = saved.get(provider, "").strip()
    if from_settings:
        return "settings"
    if _load_provider_env_api_key(provider) is not None:
        return "env"
    return "missing"


def _provider_settings_payload() -> list[dict[str, JSONValue]]:
    local_endpoint = (
        os.getenv("LOCAL_LLM_URL", DEFAULT_LOCAL_ENDPOINT).strip() or DEFAULT_LOCAL_ENDPOINT
    )
    saved_api_keys = _load_provider_api_keys()
    runtime_checks = _load_provider_runtime_checks()

    def _runtime_status(provider: str) -> dict[str, JSONValue]:
        item = runtime_checks.get(provider, {})
        valid_raw = item.get("api_key_valid")
        error_raw = item.get("last_check_error")
        checked_raw = item.get("last_checked_at")
        return {
            "api_key_valid": valid_raw if isinstance(valid_raw, bool) else None,
            "last_check_error": error_raw if isinstance(error_raw, str) else None,
            "last_checked_at": checked_raw if isinstance(checked_raw, str) else None,
        }

    return [
        {
            "provider": "xai",
            "api_key_env": "XAI_API_KEY",
            "api_key_set": _resolve_provider_api_key("xai", settings_api_keys=saved_api_keys)
            is not None,
            "api_key_source": _provider_api_key_source("xai", settings_api_keys=saved_api_keys),
            "endpoint": XAI_MODELS_ENDPOINT,
            **_runtime_status("xai"),
        },
        {
            "provider": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY",
            "api_key_set": _resolve_provider_api_key(
                "openrouter",
                settings_api_keys=saved_api_keys,
            )
            is not None,
            "api_key_source": _provider_api_key_source(
                "openrouter",
                settings_api_keys=saved_api_keys,
            ),
            "endpoint": OPENROUTER_MODELS_ENDPOINT,
            **_runtime_status("openrouter"),
        },
        {
            "provider": "local",
            "api_key_env": "LOCAL_LLM_API_KEY",
            "api_key_set": _resolve_provider_api_key("local", settings_api_keys=saved_api_keys)
            is not None,
            "api_key_source": _provider_api_key_source(
                "local",
                settings_api_keys=saved_api_keys,
            ),
            "endpoint": local_endpoint,
            **_runtime_status("local"),
        },
        {
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "api_key_set": _resolve_provider_api_key("openai", settings_api_keys=saved_api_keys)
            is not None,
            "api_key_source": _provider_api_key_source(
                "openai",
                settings_api_keys=saved_api_keys,
            ),
            "endpoint": OPENAI_STT_ENDPOINT,
            **_runtime_status("openai"),
        },
    ]


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
