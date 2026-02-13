from __future__ import annotations

import asyncio
import difflib
import hashlib
import hmac
import importlib
import io
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Protocol, cast
from urllib.parse import urlparse

import requests
from aiohttp import web
from aiohttp.multipart import BodyPartReader

from config.http_server_config import (
    DEFAULT_MAX_REQUEST_BYTES,
    HttpAuthConfig,
    HttpServerConfig,
    ensure_http_auth_boot_config,
    resolve_http_auth_config,
    resolve_http_server_config,
)
from config.memory_config import MemoryConfig, load_memory_config, save_memory_config
from config.model_whitelist import ModelNotAllowedError
from config.tools_config import (
    DEFAULT_TOOLS_STATE,
    ToolsConfig,
    load_tools_config,
    save_tools_config,
)
from core.approval_policy import (
    ALL_CATEGORIES,
    ApprovalCategory,
    ApprovalPrompt,
    ApprovalRequest,
    ApprovalRequired,
)
from core.mwv.models import MWV_REPORT_PREFIX
from core.tracer import TRACE_LOG, TraceRecord
from llm.local_http_brain import DEFAULT_LOCAL_ENDPOINT
from llm.types import ModelConfig
from memory.vector_index import VectorIndex
from server.lazy_agent import LazyAgentProvider
from server.ui_hub import UIHub
from server.ui_session_storage import PersistedSession, SQLiteUISessionStorage, UISessionStorage
from shared.memory_companion_models import FeedbackLabel, FeedbackRating
from shared.models import JSONValue, LLMMessage, ToolRequest, ToolResult
from shared.sanitize import safe_json_loads
from tools.project_tool import handle_project_request
from tools.tool_logger import DEFAULT_LOG_PATH as TOOL_CALLS_LOG
from tools.workspace_tools import (
    WORKSPACE_ROOT as DEFAULT_WORKSPACE_ROOT,
)
from tools.workspace_tools import (
    set_workspace_root as set_runtime_workspace_root,
)

ALLOWED_ROLES: Final[set[str]] = {"system", "user", "assistant", "tool"}
ALLOWED_MESSAGE_KEYS: Final[set[str]] = {"role", "content", "tool_calls"}
ALLOWED_TOP_LEVEL_KEYS: Final[set[str]] = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "slavik_meta",
}
KNOWN_SAMPLING_KEYS: Final[set[str]] = {
    "temperature",
    "top_p",
    "max_tokens",
}
EXTRA_SAMPLING_KEYS: Final[set[str]] = {
    "top_k",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "min_p",
    "num_ctx",
}
SAMPLING_PREFIXES: Final[tuple[str, ...]] = ("ollama_", "mirostat")
TOOL_PIPELINE_ENABLED: Final[bool] = False
_CATEGORY_MAP: Final[dict[str, ApprovalCategory]] = {item: item for item in ALL_CATEGORIES}

logger = logging.getLogger("SlavikAI.HttpAPI")

UI_SESSION_HEADER: Final[str] = "X-Slavik-Session"
AUTH_PROTECTED_PREFIXES: Final[tuple[str, ...]] = ("/ui/api/", "/v1/", "/slavik/")
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
DEFAULT_EMBEDDINGS_MODEL: Final[str] = "all-MiniLM-L6-v2"
DEFAULT_LONG_PASTE_TO_FILE_ENABLED: Final[bool] = True
DEFAULT_LONG_PASTE_THRESHOLD_CHARS: Final[int] = 12_000
MIN_LONG_PASTE_THRESHOLD_CHARS: Final[int] = 1_000
MAX_LONG_PASTE_THRESHOLD_CHARS: Final[int] = 80_000
UI_GITHUB_REQUIRED_CATEGORIES: Final[list[ApprovalCategory]] = ["NETWORK_RISK", "EXEC_ARBITRARY"]
UI_GITHUB_ROOT: Final[Path] = Path("sandbox/project/github").resolve()
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
_EXT_TO_TYPE: Final[dict[str, str]] = {
    "bash": "SH",
    "c": "TXT",
    "cpp": "TXT",
    "py": "PY",
    "js": "JS",
    "jsx": "JS",
    "ts": "TS",
    "tsx": "TS",
    "json": "JSON",
    "html": "HTML",
    "css": "CSS",
    "md": "MD",
    "txt": "TXT",
    "sh": "SH",
    "yaml": "TXT",
    "yml": "TXT",
    "toml": "TXT",
    "xml": "TXT",
    "sql": "TXT",
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
UI_DECISION_RESPONSES: Final[set[str]] = {"approve", "reject"}
UI_DECISION_EDITABLE_FIELDS: Final[set[str]] = {"details", "args", "query", "branch", "repo_url"}
UI_SETTINGS_FORBIDDEN_TOP_LEVEL_KEYS: Final[set[str]] = {
    "policy",
    "risk",
    "security",
    "risk_categories",
    "security_categories",
    "approval_categories",
    "approved_categories",
    "safe_mode",
}
UI_SETTINGS_FORBIDDEN_TOOL_KEYS: Final[set[str]] = {"shell", "safe_mode"}
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
_ASK_ACTION_INTENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(^/|\\b(сделай|измени|запусти|выполни|удали|установи|run|execute|modify|edit|delete|install)\\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[LLMMessage]
    stream: bool
    session_id: str | None
    sampling_warnings: list[str]
    tool_calling_present: bool


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


def _json_response(payload: dict[str, JSONValue], *, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status)


def _error_response(
    *,
    status: int,
    message: str,
    error_type: str,
    code: str,
    trace_id: str | None = None,
    details: dict[str, JSONValue] | None = None,
) -> web.Response:
    error_payload: dict[str, JSONValue] = {
        "message": message,
        "type": error_type,
        "code": code,
        "trace_id": trace_id,
        "details": details or {},
    }
    return _json_response({"error": error_payload}, status=status)


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


def _is_sampling_key(key: str) -> bool:
    if key in EXTRA_SAMPLING_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in SAMPLING_PREFIXES)


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


def _extract_ui_session_id(request: web.Request) -> str | None:
    header_value = request.headers.get(UI_SESSION_HEADER, "").strip()
    if header_value:
        return header_value
    query_value = request.query.get("session_id", "").strip()
    if query_value:
        return query_value
    return None


def _extract_bearer_token(request: web.Request) -> str | None:
    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header:
        return None
    parts = auth_header.split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    normalized = token.strip()
    return normalized or None


def _is_auth_protected_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in AUTH_PROTECTED_PREFIXES)


def _is_request_authorized(request: web.Request, auth_config: HttpAuthConfig) -> bool:
    if auth_config.allow_unauth_local:
        return True
    presented_token = _extract_bearer_token(request)
    if presented_token is None:
        return False
    candidate_tokens: list[str] = []
    if auth_config.api_token:
        candidate_tokens.append(auth_config.api_token)
    admin_token = os.environ.get("SLAVIK_ADMIN_TOKEN", "").strip()
    if admin_token:
        candidate_tokens.append(admin_token)
    return any(hmac.compare_digest(presented_token, token) for token in candidate_tokens)


@web.middleware
async def auth_gate_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    if not _is_auth_protected_path(request.path):
        return await handler(request)
    auth_config: HttpAuthConfig = request.app["auth_config"]
    if _is_request_authorized(request, auth_config):
        return await handler(request)
    return _error_response(
        status=401,
        message="Unauthorized.",
        error_type="invalid_request_error",
        code="unauthorized",
    )


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


def _validate_messages(
    raw_messages: object,
) -> tuple[list[LLMMessage] | None, str | None, bool]:
    if not isinstance(raw_messages, list):
        return None, "messages должен быть списком.", False
    if not raw_messages:
        return None, "messages не должен быть пустым.", False
    parsed: list[LLMMessage] = []
    tool_calling_present = False
    for item in raw_messages:
        if not isinstance(item, dict):
            return None, "messages[*] должен быть объектом.", False
        extra_keys = set(item.keys()) - ALLOWED_MESSAGE_KEYS
        if extra_keys:
            return (
                None,
                f"messages[*] содержит неизвестные поля: {sorted(extra_keys)}",
                False,
            )
        role_raw = item.get("role")
        content_raw = item.get("content")
        if not isinstance(role_raw, str):
            return None, "messages[*].role должен быть строкой.", False
        role = role_raw.strip()
        if role not in ALLOWED_ROLES:
            return None, f"Недопустимая роль: {role}", False
        if role == "tool":
            tool_calling_present = True
        if "tool_calls" in item:
            tool_calling_present = True
        if not isinstance(content_raw, str):
            return None, "messages[*].content должен быть строкой.", False
        if role == "system":
            parsed.append(LLMMessage(role="system", content=content_raw))
        elif role == "user":
            parsed.append(LLMMessage(role="user", content=content_raw))
        elif role == "assistant":
            parsed.append(LLMMessage(role="assistant", content=content_raw))
    return parsed, None, tool_calling_present


def _parse_chat_request(payload: dict[str, object]) -> tuple[ChatRequest | None, str]:
    structural_unknown: list[str] = []
    sampling_unknown: list[str] = []

    for key in payload.keys():
        if key in ALLOWED_TOP_LEVEL_KEYS:
            continue
        if _is_sampling_key(key):
            sampling_unknown.append(key)
        else:
            structural_unknown.append(key)

    if structural_unknown:
        return None, f"Неизвестные поля запроса: {sorted(structural_unknown)}"

    model_raw = payload.get("model")
    if not isinstance(model_raw, str) or not model_raw.strip():
        return None, "model должен быть непустой строкой."
    model = model_raw.strip()
    messages, msg_error, tool_calling_present = _validate_messages(payload.get("messages"))
    if msg_error:
        return None, msg_error
    if messages is None:
        return None, "messages невалидны."

    stream_raw = payload.get("stream", False)
    if not isinstance(stream_raw, bool):
        return None, "stream должен быть bool."
    stream = stream_raw

    for key in KNOWN_SAMPLING_KEYS:
        if key not in payload:
            continue
        value = payload.get(key)
        if not isinstance(value, (int, float)):
            return None, f"{key} должен быть числом."

    meta_raw = payload.get("slavik_meta")
    session_id: str | None = None
    if meta_raw is not None:
        if not isinstance(meta_raw, dict):
            return None, "slavik_meta должен быть объектом."
        session_raw = meta_raw.get("session_id")
        if session_raw is not None and not isinstance(session_raw, str):
            return None, "slavik_meta.session_id должен быть строкой."
        if isinstance(session_raw, str) and session_raw.strip():
            session_id = session_raw.strip()

    return (
        ChatRequest(
            model=model,
            messages=messages,
            stream=stream,
            session_id=session_id,
            sampling_warnings=sampling_unknown,
            tool_calling_present=tool_calling_present,
        ),
        "",
    )


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


async def handle_models(request: web.Request) -> web.Response:
    models = [
        {"id": "slavik", "object": "model", "owned_by": "slavik"},
    ]
    return _json_response({"object": "list", "data": models})


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
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in SESSION_MODES:
            return normalized
    return default


def _normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized


def _normalize_plan_step(step: object) -> dict[str, JSONValue] | None:
    if not isinstance(step, dict):
        return None
    step_id_raw = step.get("step_id")
    title_raw = step.get("title")
    description_raw = step.get("description")
    if not isinstance(step_id_raw, str) or not step_id_raw.strip():
        return None
    if not isinstance(title_raw, str) or not title_raw.strip():
        return None
    if not isinstance(description_raw, str):
        return None
    status_raw = step.get("status")
    status = (
        status_raw if isinstance(status_raw, str) and status_raw in PLAN_STEP_STATUSES else "todo"
    )
    evidence_raw = step.get("evidence")
    evidence = _normalize_json_value(evidence_raw) if isinstance(evidence_raw, dict) else None
    return {
        "step_id": step_id_raw.strip(),
        "title": title_raw.strip(),
        "description": description_raw,
        "allowed_tool_kinds": _normalize_string_list(step.get("allowed_tool_kinds")),
        "acceptance_checks": _normalize_string_list(step.get("acceptance_checks")),
        "status": status,
        "evidence": evidence,
    }


def _plan_hash_payload(plan: dict[str, JSONValue]) -> str:
    steps_raw = plan.get("steps")
    step_items = steps_raw if isinstance(steps_raw, list) else []
    payload = {
        "goal": plan.get("goal"),
        "scope_in": plan.get("scope_in"),
        "scope_out": plan.get("scope_out"),
        "assumptions": plan.get("assumptions"),
        "inputs_needed": plan.get("inputs_needed"),
        "steps": [
            {
                "step_id": item.get("step_id"),
                "title": item.get("title"),
                "description": item.get("description"),
                "allowed_tool_kinds": item.get("allowed_tool_kinds"),
                "acceptance_checks": item.get("acceptance_checks"),
            }
            for item in step_items
            if isinstance(item, dict)
        ],
        "exit_criteria": plan.get("exit_criteria"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_plan_payload(raw: object) -> dict[str, JSONValue] | None:
    if not isinstance(raw, dict):
        return None
    plan_id_raw = raw.get("plan_id")
    if not isinstance(plan_id_raw, str) or not plan_id_raw.strip():
        return None
    status_raw = raw.get("status")
    status = status_raw if isinstance(status_raw, str) and status_raw in PLAN_STATUSES else "draft"
    goal_raw = raw.get("goal")
    goal = goal_raw if isinstance(goal_raw, str) else ""
    created_at_raw = raw.get("created_at")
    updated_at_raw = raw.get("updated_at")
    now = _utc_now_iso()
    steps_raw = raw.get("steps")
    steps: list[dict[str, JSONValue]] = []
    if isinstance(steps_raw, list):
        for item in steps_raw:
            normalized = _normalize_plan_step(item)
            if normalized is not None:
                steps.append(normalized)
    normalized_plan: dict[str, JSONValue] = {
        "plan_id": plan_id_raw.strip(),
        "plan_hash": "",
        "status": status,
        "goal": goal,
        "scope_in": _normalize_string_list(raw.get("scope_in")),
        "scope_out": _normalize_string_list(raw.get("scope_out")),
        "assumptions": _normalize_string_list(raw.get("assumptions")),
        "inputs_needed": _normalize_string_list(raw.get("inputs_needed")),
        "audit_log": (
            [_normalize_json_value(item) for item in raw.get("audit_log", [])]
            if isinstance(raw.get("audit_log"), list)
            else []
        ),
        "steps": steps,
        "exit_criteria": _normalize_string_list(raw.get("exit_criteria")),
        "created_at": (
            created_at_raw if isinstance(created_at_raw, str) and created_at_raw.strip() else now
        ),
        "updated_at": (
            updated_at_raw if isinstance(updated_at_raw, str) and updated_at_raw.strip() else now
        ),
        "approved_at": raw.get("approved_at") if isinstance(raw.get("approved_at"), str) else None,
        "approved_by": raw.get("approved_by") if isinstance(raw.get("approved_by"), str) else None,
    }
    normalized_plan["plan_hash"] = _plan_hash_payload(normalized_plan)
    return normalized_plan


def _normalize_task_payload(raw: object) -> dict[str, JSONValue] | None:
    if not isinstance(raw, dict):
        return None
    task_id_raw = raw.get("task_id")
    plan_id_raw = raw.get("plan_id")
    plan_hash_raw = raw.get("plan_hash")
    if not isinstance(task_id_raw, str) or not task_id_raw.strip():
        return None
    if not isinstance(plan_id_raw, str) or not plan_id_raw.strip():
        return None
    if not isinstance(plan_hash_raw, str) or not plan_hash_raw.strip():
        return None
    status_raw = raw.get("status")
    status = (
        status_raw if isinstance(status_raw, str) and status_raw in TASK_STATUSES else "running"
    )
    current_step_raw = raw.get("current_step_id")
    current_step = (
        current_step_raw.strip()
        if isinstance(current_step_raw, str) and current_step_raw.strip()
        else None
    )
    started_at_raw = raw.get("started_at")
    updated_at_raw = raw.get("updated_at")
    now = _utc_now_iso()
    return {
        "task_id": task_id_raw.strip(),
        "plan_id": plan_id_raw.strip(),
        "plan_hash": plan_hash_raw.strip(),
        "current_step_id": current_step,
        "status": status,
        "started_at": (
            started_at_raw if isinstance(started_at_raw, str) and started_at_raw.strip() else now
        ),
        "updated_at": (
            updated_at_raw if isinstance(updated_at_raw, str) and updated_at_raw.strip() else now
        ),
    }


def _build_stop_text(*, happened: str, why: str, steps: list[str], code: str) -> str:
    next_steps = steps[:3] if steps else ["Повторите запрос после корректировки режима."]
    lines = [
        f"Что случилось: {happened}",
        f"Почему: {why}",
        "Что делать дальше:",
    ]
    for item in next_steps:
        lines.append(f"- {item}")
    report = {
        "route": "chat",
        "trace_id": None,
        "stop_reason_code": code,
        "next_steps": next_steps,
    }
    lines.append(f"MWV_REPORT_JSON={json.dumps(report, ensure_ascii=False)}")
    return "\n".join(lines)


async def _apply_agent_runtime_state(
    *,
    agent: AgentProtocol,
    hub: UIHub,
    session_id: str,
) -> tuple[str, dict[str, JSONValue] | None, dict[str, JSONValue] | None]:
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


def _request_likely_action_intent(user_input: str) -> bool:
    normalized = user_input.strip()
    if not normalized:
        return False
    return bool(_ASK_ACTION_INTENT_PATTERN.search(normalized))


def _decision_workflow_context(
    *,
    mode: str,
    active_plan: dict[str, JSONValue] | None,
    active_task: dict[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    step_id: str | None = None
    task_id: str | None = None
    if isinstance(active_task, dict):
        step_raw = active_task.get("current_step_id")
        task_raw = active_task.get("task_id")
        if isinstance(step_raw, str) and step_raw.strip():
            step_id = step_raw.strip()
        if isinstance(task_raw, str) and task_raw.strip():
            task_id = task_raw.strip()
    plan_id: str | None = None
    plan_hash: str | None = None
    if isinstance(active_plan, dict):
        plan_id_raw = active_plan.get("plan_id")
        plan_hash_raw = active_plan.get("plan_hash")
        if isinstance(plan_id_raw, str) and plan_id_raw.strip():
            plan_id = plan_id_raw.strip()
        if isinstance(plan_hash_raw, str) and plan_hash_raw.strip():
            plan_hash = plan_hash_raw.strip()
    return {
        "mode": mode,
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "task_id": task_id,
        "step_id": step_id,
    }


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
    if not isinstance(raw, list):
        return []
    options: list[dict[str, JSONValue]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        option_id = item.get("id")
        title = item.get("title")
        action = item.get("action")
        if not isinstance(option_id, str) or not option_id.strip():
            continue
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(action, str) or not action.strip():
            continue
        payload_raw = item.get("payload")
        risk_raw = item.get("risk")
        payload_normalized = (
            _normalize_json_value(payload_raw) if isinstance(payload_raw, dict) else {}
        )
        risk = risk_raw if isinstance(risk_raw, str) and risk_raw.strip() else "low"
        options.append(
            {
                "id": option_id.strip(),
                "title": title.strip(),
                "action": action.strip(),
                "payload": payload_normalized,
                "risk": risk,
            }
        )
    return options


def _normalize_ui_decision(
    raw: object,
    *,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> dict[str, JSONValue] | None:
    if not isinstance(raw, dict):
        return None
    if not raw:
        return None

    decision_id_raw = raw.get("id")
    if not isinstance(decision_id_raw, str) or not decision_id_raw.strip():
        return None
    decision_id = decision_id_raw.strip()
    now = _utc_now_iso()

    kind_raw = raw.get("kind")
    status_raw = raw.get("status")
    reason_raw = raw.get("reason")
    summary_raw = raw.get("summary")
    context_raw = raw.get("context")
    options_raw = raw.get("options")
    default_option_id_raw = raw.get("default_option_id")
    created_at_raw = raw.get("created_at")
    updated_at_raw = raw.get("updated_at")
    resolved_at_raw = raw.get("resolved_at")

    context: dict[str, JSONValue] = {}
    if isinstance(context_raw, dict):
        for key, value in context_raw.items():
            context[str(key)] = _normalize_json_value(value)
    if session_id and "session_id" not in context:
        context["session_id"] = session_id
    if trace_id and "trace_id" not in context:
        context["trace_id"] = trace_id

    options = _normalize_ui_decision_options(options_raw)
    default_option_id: str | None = None
    if isinstance(default_option_id_raw, str) and default_option_id_raw.strip():
        default_option_id = default_option_id_raw.strip()
    elif options:
        first_id = options[0].get("id")
        default_option_id = first_id if isinstance(first_id, str) else None

    if (
        isinstance(kind_raw, str)
        and kind_raw in UI_DECISION_KINDS
        and isinstance(status_raw, str)
        and status_raw in UI_DECISION_STATUSES
    ):
        kind = kind_raw
        status = status_raw
        blocking = raw.get("blocking") is True
        reason = reason_raw if isinstance(reason_raw, str) and reason_raw.strip() else "decision"
        summary = summary_raw if isinstance(summary_raw, str) and summary_raw.strip() else reason
        proposed_action_raw = raw.get("proposed_action")
        proposed_action = (
            _normalize_json_value(proposed_action_raw)
            if isinstance(proposed_action_raw, dict)
            else {}
        )
        created_at = (
            created_at_raw if isinstance(created_at_raw, str) and created_at_raw.strip() else now
        )
        updated_at = (
            updated_at_raw if isinstance(updated_at_raw, str) and updated_at_raw.strip() else now
        )
        resolved_at = (
            resolved_at_raw
            if isinstance(resolved_at_raw, str) and resolved_at_raw.strip()
            else None
        )
        return {
            "id": decision_id,
            "kind": kind,
            "status": status,
            "blocking": blocking,
            "reason": reason,
            "summary": summary,
            "proposed_action": proposed_action,
            "options": options,
            "default_option_id": default_option_id,
            "context": context,
            "created_at": created_at,
            "updated_at": updated_at,
            "resolved_at": resolved_at,
        }

    # Legacy DecisionPacket -> UiDecision
    reason = reason_raw if isinstance(reason_raw, str) and reason_raw.strip() else "decision"
    summary = summary_raw if isinstance(summary_raw, str) and summary_raw.strip() else reason
    created_at = (
        created_at_raw if isinstance(created_at_raw, str) and created_at_raw.strip() else now
    )
    return {
        "id": decision_id,
        "kind": "decision",
        "status": "pending",
        "blocking": True,
        "reason": reason,
        "summary": summary,
        "proposed_action": {},
        "options": options,
        "default_option_id": default_option_id,
        "context": context,
        "created_at": created_at,
        "updated_at": now,
        "resolved_at": None,
    }


def _build_ui_approval_decision(
    *,
    approval_request: dict[str, JSONValue],
    session_id: str,
    source_endpoint: str,
    resume_payload: dict[str, JSONValue],
    trace_id: str | None = None,
    workflow_context: dict[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    prompt_raw = approval_request.get("prompt")
    prompt = prompt_raw if isinstance(prompt_raw, dict) else {}
    what_raw = prompt.get("what")
    why_raw = prompt.get("why")
    category_raw = approval_request.get("category")
    tool_raw = approval_request.get("tool")
    details_raw = approval_request.get("details")
    required_raw = approval_request.get("required_categories")
    required_categories: list[str] = []
    if isinstance(required_raw, list):
        for item in required_raw:
            if isinstance(item, str) and item in ALL_CATEGORIES:
                required_categories.append(item)

    summary = (
        what_raw.strip()
        if isinstance(what_raw, str) and what_raw.strip()
        else "Требуется подтверждение действия."
    )
    reason = (
        why_raw.strip() if isinstance(why_raw, str) and why_raw.strip() else "approval_required"
    )
    proposed_action: dict[str, JSONValue] = {
        "category": category_raw if isinstance(category_raw, str) else "",
        "required_categories": required_categories,
        "tool": tool_raw if isinstance(tool_raw, str) else "",
        "details": _normalize_json_value(details_raw) if isinstance(details_raw, dict) else {},
    }
    now = _utc_now_iso()
    context_payload: dict[str, JSONValue] = {
        "session_id": session_id,
        "trace_id": trace_id,
        "source_endpoint": source_endpoint,
        "resume_payload": resume_payload,
    }
    if isinstance(workflow_context, dict):
        for key, value in workflow_context.items():
            context_payload[str(key)] = _normalize_json_value(value)
    return {
        "id": f"decision-{uuid.uuid4().hex}",
        "kind": "approval",
        "status": "pending",
        "blocking": True,
        "reason": reason,
        "summary": summary,
        "proposed_action": proposed_action,
        "options": [
            {
                "id": "approve",
                "title": "Approve",
                "action": "approve",
                "payload": {},
                "risk": "medium",
            },
            {
                "id": "reject",
                "title": "Reject",
                "action": "reject",
                "payload": {},
                "risk": "low",
            },
        ],
        "default_option_id": "approve",
        "context": context_payload,
        "created_at": now,
        "updated_at": now,
        "resolved_at": None,
    }


def _decision_is_pending_blocking(decision: dict[str, JSONValue] | None) -> bool:
    if not isinstance(decision, dict):
        return False
    status = decision.get("status")
    blocking = decision.get("blocking")
    return status == "pending" and blocking is True


def _decision_with_status(
    decision: dict[str, JSONValue],
    *,
    status: str,
    resolved: bool = False,
) -> dict[str, JSONValue]:
    updated = dict(decision)
    updated["status"] = status
    updated["updated_at"] = _utc_now_iso()
    if resolved:
        updated["resolved_at"] = _utc_now_iso()
        updated["blocking"] = False
    return updated


def _normalize_json_value(value: object) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JSONValue] = {}
        for key, item in value.items():
            normalized[str(key)] = _normalize_json_value(item)
        return normalized
    return str(value)


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


def _artifact_type_from_ext(ext: str | None) -> str:
    if not ext:
        return "TXT"
    return _EXT_TO_TYPE.get(ext.lower(), "TXT")


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


def _load_embeddings_model_setting() -> str:
    payload = _load_ui_settings_blob()
    memory_raw = payload.get("memory")
    if not isinstance(memory_raw, dict):
        return DEFAULT_EMBEDDINGS_MODEL
    model_raw = memory_raw.get("embeddings_model")
    if not isinstance(model_raw, str):
        return DEFAULT_EMBEDDINGS_MODEL
    normalized = model_raw.strip()
    return normalized or DEFAULT_EMBEDDINGS_MODEL


def _save_embeddings_model_setting(model_name: str) -> None:
    payload = _load_ui_settings_blob()
    memory_raw = payload.get("memory")
    memory_payload: dict[str, object]
    if isinstance(memory_raw, dict):
        memory_payload = dict(memory_raw)
    else:
        memory_payload = {}
    memory_payload["embeddings_model"] = model_name.strip() or DEFAULT_EMBEDDINGS_MODEL
    payload["memory"] = memory_payload
    _save_ui_settings_blob(payload)


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


def _forbidden_settings_update_key(payload: dict[str, object]) -> str | None:
    for key in UI_SETTINGS_FORBIDDEN_TOP_LEVEL_KEYS:
        if key in payload:
            return key
    tools_raw = payload.get("tools")
    if not isinstance(tools_raw, dict):
        return None
    state_raw = tools_raw.get("state")
    if not isinstance(state_raw, dict):
        return None
    for key in UI_SETTINGS_FORBIDDEN_TOOL_KEYS:
        if key in state_raw:
            return f"tools.state.{key}"
    return None


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
    return [
        {
            "provider": "xai",
            "api_key_env": "XAI_API_KEY",
            "api_key_set": _resolve_provider_api_key("xai", settings_api_keys=saved_api_keys)
            is not None,
            "api_key_source": _provider_api_key_source("xai", settings_api_keys=saved_api_keys),
            "endpoint": XAI_MODELS_ENDPOINT,
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
        },
    ]


def _build_settings_payload() -> dict[str, JSONValue]:
    tone, system_prompt = _load_personalization_settings()
    long_paste_to_file_enabled, long_paste_threshold_chars = _load_composer_settings()
    policy_profile, yolo_armed, yolo_armed_at = _load_policy_settings()
    memory_config = load_memory_config()
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
                "embeddings_model": _load_embeddings_model_setting(),
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


def _normalize_import_status(raw: object) -> str:
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"ok", "busy", "error"}:
            return normalized
    return "ok"


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


def _parse_imported_session(raw: object) -> PersistedSession | None:
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
    return PersistedSession(
        session_id=session_id,
        created_at=created_at,
        updated_at=updated_at,
        status="ok",
        decision=None,
        messages=messages,
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


def _parse_github_import_args(args_raw: str) -> tuple[str, str | None]:
    try:
        parts = shlex.split(args_raw)
    except ValueError as exc:
        raise ValueError(f"Некорректные аргументы github_import: {exc}") from exc
    if not parts:
        raise ValueError("Укажи ссылку на GitHub репозиторий.")
    repo_url = parts[0].strip()
    if not repo_url:
        raise ValueError("Укажи ссылку на GitHub репозиторий.")
    branch = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    return repo_url, branch


def _validate_github_segment(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Некорректный сегмент пути GitHub.")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(char not in allowed for char in normalized):
        raise ValueError("Допустимы только owner/repo с символами [a-zA-Z0-9-_.].")
    return normalized


def _resolve_github_target(repo_url: str) -> tuple[Path, str]:
    parsed = urlparse(repo_url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("Поддерживаются только http(s) GitHub URL.")
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError("Поддерживаются только github.com репозитории.")
    path_parts = [part for part in parsed.path.split("/") if part.strip()]
    if len(path_parts) < 2:
        raise ValueError("URL должен содержать owner/repo.")
    owner = _validate_github_segment(path_parts[0])
    repo_name_raw = path_parts[1].removesuffix(".git")
    repo_name = _validate_github_segment(repo_name_raw)
    UI_GITHUB_ROOT.mkdir(parents=True, exist_ok=True)
    target_root = UI_GITHUB_ROOT / owner
    target_root.mkdir(parents=True, exist_ok=True)
    candidate = target_root / repo_name
    suffix = 1
    while candidate.exists():
        candidate = target_root / f"{repo_name}-{suffix}"
        suffix += 1
    sandbox_root = Path("sandbox/project").resolve()
    relative_target = candidate.resolve().relative_to(sandbox_root)
    return candidate, str(relative_target)


def _build_github_import_approval_request(
    *,
    session_id: str,
    repo_url: str,
    branch: str | None,
    required_categories: list[ApprovalCategory],
) -> ApprovalRequest:
    branch_text = f", branch={branch}" if branch else ""
    return ApprovalRequest(
        category=required_categories[0],
        required_categories=required_categories,
        prompt=ApprovalPrompt(
            what=f"GitHub import: {repo_url}{branch_text}",
            why="Нужно скачать внешний репозиторий и проиндексировать код.",
            risk="Сетевой доступ и выполнение git clone.",
            changes=[
                "Создание каталога в sandbox/project/github/*",
                "Скачивание внешнего репозитория",
                "Индексация файлов в memory/vectors.db",
            ],
        ),
        tool="project",
        details={
            "command": "github_import",
            "repo_url": repo_url,
            "branch": branch,
        },
        session_id=session_id,
    )


async def _clone_github_repository(
    *,
    repo_url: str,
    branch: str | None,
    target_path: Path,
) -> tuple[bool, str]:
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([repo_url, str(target_path)])
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Git clone failed: {exc}"
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "unknown error"
        try:
            if target_path.exists():
                shutil.rmtree(target_path, ignore_errors=True)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to cleanup clone target after error", exc_info=True)
        return False, f"Git clone failed: {details}"
    return True, "ok"


def _index_imported_project(relative_path: str) -> tuple[bool, str]:
    try:
        tool_result = handle_project_request(
            ToolRequest(
                name="project",
                args={"cmd": "index", "args": [relative_path]},
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"project index exception: {exc}"
    if not tool_result.ok:
        message = tool_result.error or "project index failed"
        return False, message
    data = tool_result.data if isinstance(tool_result.data, dict) else {}
    indexed_code = data.get("indexed_code")
    indexed_docs = data.get("indexed_docs")
    return True, f"Code={indexed_code}, Docs={indexed_docs}"


async def handle_ui_redirect(request: web.Request) -> web.StreamResponse:
    raise web.HTTPFound("/ui/")


async def handle_ui_index(request: web.Request) -> web.FileResponse:
    dist_path: Path = request.app["ui_dist_path"]
    index_path = dist_path / "index.html"
    return web.FileResponse(path=index_path)


async def handle_workspace_index(request: web.Request) -> web.FileResponse:
    return await handle_ui_index(request)


async def handle_ui_workspace_root_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    root_path = await _workspace_root_for_session(hub, session_id)
    policy = await hub.get_session_policy(session_id)
    response = _json_response(
        {
            "session_id": session_id,
            "root_path": str(root_path),
            "policy": policy,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_root_select(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_store: SessionApprovalStore = request.app["session_store"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    path_raw = payload.get("root_path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        return _error_response(
            status=400,
            message="root_path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    policy = await hub.get_session_policy(session_id)
    profile_raw = policy.get("profile")
    profile = profile_raw if isinstance(profile_raw, str) else DEFAULT_POLICY_PROFILE
    try:
        target_root = _resolve_workspace_root_candidate(path_raw.strip(), policy_profile=profile)
    except ValueError as exc:
        return _error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    current_root = await _workspace_root_for_session(hub, session_id)
    if current_root == target_root:
        response = _json_response(
            {
                "session_id": session_id,
                "root_path": str(current_root),
                "applied": True,
            }
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response

    required_category: ApprovalCategory = "FS_OUTSIDE_WORKSPACE"
    approved = await session_store.get_categories(session_id)
    if required_category in approved:
        await hub.set_workspace_root(session_id, str(target_root))
        response = _json_response(
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
    response = _json_response(
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
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    root_path = await _workspace_root_for_session(hub, session_id)
    stats = _index_workspace_root(root_path)
    response = _json_response(
        {
            "session_id": session_id,
            "ok": True,
            **stats,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_git_diff(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    root_path = await _workspace_root_for_session(hub, session_id)
    diff, error = _workspace_git_diff(root_path)
    response = _json_response(
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
) -> tuple[AgentProtocol | None, str, set[ApprovalCategory]]:
    hub: UIHub = request.app["ui_hub"]
    session_store: SessionApprovalStore = request.app["session_store"]
    agent = await _resolve_agent(request)
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    approved_categories = await session_store.get_categories(session_id)
    return agent, session_id, approved_categories


async def _workspace_root_for_session(hub: UIHub, session_id: str) -> Path:
    stored_root = await hub.get_workspace_root(session_id)
    if isinstance(stored_root, str) and stored_root.strip():
        candidate = Path(stored_root).expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
    return WORKSPACE_ROOT


def _resolve_workspace_root_candidate(path_raw: str, *, policy_profile: str) -> Path:
    candidate = Path(path_raw).expanduser().resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"Директория не найдена: {candidate}")
    if policy_profile != "yolo":
        try:
            candidate.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError("Root должен быть внутри директории проекта.") from exc
    return candidate


def _index_workspace_root(root: Path) -> dict[str, JSONValue]:
    vector_index = VectorIndex("memory/vectors.db", model_name=_load_embeddings_model_setting())
    indexed_code = 0
    indexed_docs = 0
    skipped = 0
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in WORKSPACE_INDEX_IGNORED_DIRS]
        current = Path(current_root)
        for filename in files:
            full_path = current / filename
            if filename.endswith(".sqlite"):
                skipped += 1
                continue
            suffix = full_path.suffix.lower()
            if suffix not in WORKSPACE_INDEX_ALLOWED_EXTENSIONS:
                skipped += 1
                continue
            try:
                if full_path.stat().st_size > WORKSPACE_INDEX_MAX_FILE_BYTES:
                    skipped += 1
                    continue
                content = full_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            namespace = "code" if suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".sh"} else "docs"
            vector_index.upsert_text(str(full_path), content, namespace=namespace)
            if namespace == "code":
                indexed_code += 1
            else:
                indexed_docs += 1
    return {
        "root_path": str(root),
        "indexed_code": indexed_code,
        "indexed_docs": indexed_docs,
        "skipped": skipped,
    }


def _workspace_git_diff(root: Path) -> tuple[str, str | None]:
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--no-ext-diff", "--", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "not a git repository" in stderr.lower():
            return "", None
        return "", stderr or "git diff failed"
    return result.stdout, None


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
    updated = dict(plan)
    steps_raw = updated.get("steps")
    next_steps: list[dict[str, JSONValue]] = []
    if isinstance(steps_raw, list):
        for item in steps_raw:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            current_id = candidate.get("step_id")
            if isinstance(current_id, str) and current_id == step_id:
                candidate["status"] = status if status in PLAN_STEP_STATUSES else "blocked"
                candidate["evidence"] = evidence
            next_steps.append(candidate)
    updated["steps"] = next_steps
    updated["updated_at"] = _utc_now_iso()
    updated["plan_hash"] = _plan_hash_payload(updated)
    return updated


def _find_next_todo_step(plan: dict[str, JSONValue]) -> dict[str, JSONValue] | None:
    steps_raw = plan.get("steps")
    if not isinstance(steps_raw, list):
        return None
    for item in steps_raw:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status == "todo":
            return item
    return None


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
                await _apply_agent_runtime_state(agent=agent, hub=hub, session_id=session_id)
                agent.set_session_context(session_id, approved_categories)
            except Exception:  # noqa: BLE001
                logger.debug(
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
                approval_payload = _serialize_approval_request(exc.request)
                workflow = await hub.get_session_workflow(session_id)
                mode = _normalize_mode_value(workflow.get("mode"), default="ask")
                active_plan = _normalize_plan_payload(workflow.get("active_plan"))
                active_task = _normalize_task_payload(workflow.get("active_task"))
                decision = _build_ui_approval_decision(
                    approval_request=approval_payload or {},
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
                response = _json_response(
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
    try:
        agent, session_id, approved_categories = await _resolve_workspace_session(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_list",
        raw_input="ui:workspace_list",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return _error_response(
            status=400,
            message=tool_result.error or "Не удалось получить структуру workspace.",
            error_type="invalid_request_error",
            code="workspace_list_failed",
        )
    tree_raw = tool_result.data.get("tree")
    tree: list[JSONValue] = tree_raw if isinstance(tree_raw, list) else []
    root_path = await _workspace_root_for_session(hub, session_id)
    response = _json_response({"session_id": session_id, "tree": tree, "root_path": str(root_path)})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_file_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    try:
        agent, session_id, approved_categories = await _resolve_workspace_session(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    path_raw = request.query.get("path", "")
    path_value = path_raw.strip()
    if not path_value:
        return _error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    tool_result_or_error = await _call_workspace_tool(
        request=request,
        agent=agent,
        session_id=session_id,
        approved_categories=approved_categories,
        tool_name="workspace_read",
        args={"path": path_value},
        raw_input=f"ui:workspace_read {path_value}",
    )
    if isinstance(tool_result_or_error, web.Response):
        return tool_result_or_error
    tool_result = tool_result_or_error
    if not tool_result.ok:
        return _error_response(
            status=400,
            message=tool_result.error or "Не удалось прочитать файл.",
            error_type="invalid_request_error",
            code="workspace_read_failed",
        )
    content_raw = tool_result.data.get("output")
    if not isinstance(content_raw, str):
        content_raw = ""
    response = _json_response(
        {
            "session_id": session_id,
            "path": path_value,
            "content": content_raw,
            "root_path": str(await _workspace_root_for_session(hub, session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_file_put(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    try:
        agent, session_id, approved_categories = await _resolve_workspace_session(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    path_raw = payload.get("path")
    content_raw = payload.get("content")
    if not isinstance(path_raw, str) or not path_raw.strip():
        return _error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(content_raw, str):
        return _error_response(
            status=400,
            message="content должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    path_value = path_raw.strip()
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
        return _error_response(
            status=400,
            message=tool_result.error or "Не удалось сохранить файл.",
            error_type="invalid_request_error",
            code="workspace_write_failed",
        )
    response = _json_response(
        {
            "session_id": session_id,
            "path": path_value,
            "saved": True,
            "root_path": str(await _workspace_root_for_session(hub, session_id)),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_run(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    try:
        agent, session_id, approved_categories = await _resolve_workspace_session(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    path_raw = payload.get("path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        return _error_response(
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
        return _error_response(
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
    response = _json_response(
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


async def handle_ui_status(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    workspace_root = await _workspace_root_for_session(hub, session_id)
    policy = await hub.get_session_policy(session_id)
    workflow = await hub.get_session_workflow(session_id)
    decision = _normalize_ui_decision(
        await hub.get_session_decision(session_id),
        session_id=session_id,
    )
    selected_model = await hub.get_session_model(session_id)
    response = _json_response(
        {
            "ok": True,
            "session_id": session_id,
            "decision": decision,
            "selected_model": selected_model,
            "workspace_root": str(workspace_root),
            "policy": policy,
            "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
            "active_task": _normalize_task_payload(workflow.get("active_task")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_state(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    workflow = await hub.get_session_workflow(session_id)
    decision = _normalize_ui_decision(
        await hub.get_session_decision(session_id),
        session_id=session_id,
    )
    payload: dict[str, JSONValue] = {
        "ok": True,
        "session_id": session_id,
        "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
        "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
        "active_task": _normalize_task_payload(workflow.get("active_task")),
        "pending_decision": decision,
    }
    response = _json_response(payload)
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_mode(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    mode_raw = payload.get("mode")
    if not isinstance(mode_raw, str) or mode_raw.strip().lower() not in SESSION_MODES:
        return _error_response(
            status=400,
            message="mode должен быть ask|plan|act.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    next_mode = _normalize_mode_value(mode_raw, default="ask")
    workflow = await hub.get_session_workflow(session_id)
    current_mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    active_plan = _normalize_plan_payload(workflow.get("active_plan"))
    if next_mode == "act" and current_mode != "plan":
        return _error_response(
            status=409,
            message="В act можно перейти только из plan-режима.",
            error_type="invalid_request_error",
            code="mode_transition_not_allowed",
        )
    if current_mode == "plan" and next_mode == "act":
        confirm = payload.get("confirm") is True
        if not confirm:
            return _error_response(
                status=409,
                message="Для перехода plan->act нужен confirm=true.",
                error_type="invalid_request_error",
                code="mode_confirm_required",
            )
        if active_plan is None or active_plan.get("status") != "approved":
            return _error_response(
                status=409,
                message="Нужен approved план для перехода в act.",
                error_type="invalid_request_error",
                code="plan_not_approved",
            )
    if next_mode == "ask":
        await hub.set_session_workflow(
            session_id,
            mode="ask",
            active_task=None,
        )
    else:
        await hub.set_session_workflow(session_id, mode=next_mode)
    updated = await hub.get_session_workflow(session_id)
    response = _json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_plan_draft(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    if mode != "plan":
        return _error_response(
            status=409,
            message="Draft доступен только в plan-режиме.",
            error_type="invalid_request_error",
            code="mode_not_plan",
        )
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    goal_raw = payload.get("goal")
    if not isinstance(goal_raw, str) or not goal_raw.strip():
        return _error_response(
            status=400,
            message="goal обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    root = await _workspace_root_for_session(hub, session_id)
    audit_log, usage = _run_plan_readonly_audit(root=root)
    if (
        usage["read_files"] >= PLAN_AUDIT_MAX_READ_FILES
        or usage["total_bytes"] >= PLAN_AUDIT_MAX_TOTAL_BYTES
        or usage["search_calls"] >= PLAN_AUDIT_MAX_SEARCH_CALLS
    ):
        return _error_response(
            status=409,
            message="Достигнут лимит read-only аудита.",
            error_type="invalid_request_error",
            code="PLAN_AUDIT_LIMIT_REACHED",
        )

    draft = _build_plan_draft(goal=goal_raw.strip(), audit_log=audit_log)
    await hub.set_session_workflow(session_id, active_plan=draft, active_task=None)
    updated = await hub.get_session_workflow(session_id)
    response = _json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="plan"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
            "audit_usage": usage,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_plan_approve(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    if mode != "plan":
        return _error_response(
            status=409,
            message="Approve доступен только в plan-режиме.",
            error_type="invalid_request_error",
            code="mode_not_plan",
        )
    plan = _normalize_plan_payload(workflow.get("active_plan"))
    if plan is None:
        return _error_response(
            status=404,
            message="Draft plan не найден.",
            error_type="invalid_request_error",
            code="plan_not_found",
        )
    if plan.get("status") != "draft":
        return _error_response(
            status=409,
            message="План должен быть в статусе draft.",
            error_type="invalid_request_error",
            code="plan_not_draft",
        )
    approved = _plan_with_status(plan, status="approved")
    await hub.set_session_workflow(session_id, active_plan=approved)
    updated = await hub.get_session_workflow(session_id)
    response = _json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="plan"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_plan_execute(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    workflow = await hub.get_session_workflow(session_id)
    plan = _normalize_plan_payload(workflow.get("active_plan"))
    if plan is None:
        return _error_response(
            status=404,
            message="План не найден.",
            error_type="invalid_request_error",
            code="plan_not_found",
        )
    if plan.get("status") != "approved":
        return _error_response(
            status=409,
            message="Сначала approve план.",
            error_type="invalid_request_error",
            code="plan_not_approved",
        )
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    expected_hash = payload.get("plan_hash")
    actual_hash = plan.get("plan_hash")
    if (
        isinstance(expected_hash, str)
        and expected_hash.strip()
        and isinstance(actual_hash, str)
        and expected_hash.strip() != actual_hash
    ):
        return _error_response(
            status=409,
            message="plan_hash mismatch.",
            error_type="invalid_request_error",
            code="plan_hash_mismatch",
        )

    task: dict[str, JSONValue] = {
        "task_id": f"task-{uuid.uuid4().hex}",
        "plan_id": plan.get("plan_id"),
        "plan_hash": plan.get("plan_hash"),
        "current_step_id": None,
        "status": "running",
        "started_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    running_plan = _plan_with_status(plan, status="running")
    await hub.set_session_workflow(
        session_id,
        mode="act",
        active_plan=running_plan,
        active_task=task,
    )
    plan_id_raw = task.get("plan_id")
    task_id_raw = task.get("task_id")
    if isinstance(plan_id_raw, str) and isinstance(task_id_raw, str):
        asyncio.create_task(
            _run_plan_runner(
                app=request.app,
                session_id=session_id,
                plan_id=plan_id_raw,
                task_id=task_id_raw,
            )
        )
    updated = await hub.get_session_workflow(session_id)
    response = _json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="act"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_plan_cancel(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    workflow = await hub.get_session_workflow(session_id)
    plan = _normalize_plan_payload(workflow.get("active_plan"))
    task = _normalize_task_payload(workflow.get("active_task"))
    if plan is not None:
        plan = _plan_with_status(plan, status="cancelled")
    if task is not None:
        task = _task_with_status(task, status="cancelled", current_step_id=None)
    await hub.set_session_workflow(session_id, active_plan=plan, active_task=task)
    updated = await hub.get_session_workflow(session_id)
    response = _json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_settings(request: web.Request) -> web.Response:
    del request
    try:
        return _json_response(_build_settings_payload())
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=500,
            message=f"Не удалось загрузить settings: {exc}",
            error_type="internal_error",
            code="settings_load_failed",
        )


async def handle_ui_settings_update(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    forbidden_key = _forbidden_settings_update_key(payload)
    if forbidden_key is not None:
        return _error_response(
            status=403,
            message=f"Изменение security-поля запрещено: {forbidden_key}",
            error_type="forbidden",
            code="security_fields_forbidden",
        )

    next_embeddings_model: str | None = None
    next_policy_profile: str | None = None
    next_yolo_armed: bool | None = None
    next_yolo_armed_at: str | None = None

    personalization_raw = payload.get("personalization")
    if personalization_raw is not None:
        if not isinstance(personalization_raw, dict):
            return _error_response(
                status=400,
                message="personalization должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        tone, system_prompt = _load_personalization_settings()
        if "tone" in personalization_raw:
            tone_raw = personalization_raw.get("tone")
            if not isinstance(tone_raw, str):
                return _error_response(
                    status=400,
                    message="personalization.tone должен быть строкой.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            normalized_tone = tone_raw.strip()
            if not normalized_tone:
                return _error_response(
                    status=400,
                    message="personalization.tone не должен быть пустым.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            tone = normalized_tone
        if "system_prompt" in personalization_raw:
            prompt_raw = personalization_raw.get("system_prompt")
            if not isinstance(prompt_raw, str):
                return _error_response(
                    status=400,
                    message="personalization.system_prompt должен быть строкой.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            system_prompt = prompt_raw
        _save_personalization_settings(tone=tone, system_prompt=system_prompt)

    composer_raw = payload.get("composer")
    if composer_raw is not None:
        if not isinstance(composer_raw, dict):
            return _error_response(
                status=400,
                message="composer должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        long_paste_to_file_enabled, long_paste_threshold_chars = _load_composer_settings()
        if "long_paste_to_file_enabled" in composer_raw:
            enabled_raw = composer_raw.get("long_paste_to_file_enabled")
            if not isinstance(enabled_raw, bool):
                return _error_response(
                    status=400,
                    message="composer.long_paste_to_file_enabled должен быть bool.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            long_paste_to_file_enabled = enabled_raw
        if "long_paste_threshold_chars" in composer_raw:
            threshold_raw = composer_raw.get("long_paste_threshold_chars")
            if isinstance(threshold_raw, bool) or not isinstance(threshold_raw, int):
                return _error_response(
                    status=400,
                    message="composer.long_paste_threshold_chars должен быть int.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            if (
                threshold_raw < MIN_LONG_PASTE_THRESHOLD_CHARS
                or threshold_raw > MAX_LONG_PASTE_THRESHOLD_CHARS
            ):
                return _error_response(
                    status=400,
                    message=(
                        "composer.long_paste_threshold_chars вне диапазона "
                        f"{MIN_LONG_PASTE_THRESHOLD_CHARS}..{MAX_LONG_PASTE_THRESHOLD_CHARS}."
                    ),
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            long_paste_threshold_chars = threshold_raw
        _save_composer_settings(
            long_paste_to_file_enabled=long_paste_to_file_enabled,
            long_paste_threshold_chars=long_paste_threshold_chars,
        )

    memory_raw = payload.get("memory")
    if memory_raw is not None:
        if not isinstance(memory_raw, dict):
            return _error_response(
                status=400,
                message="memory должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        current_memory = load_memory_config()
        auto_save_dialogue = current_memory.auto_save_dialogue
        inbox_max_items = current_memory.inbox_max_items
        inbox_ttl_days = current_memory.inbox_ttl_days
        inbox_writes_per_minute = current_memory.inbox_writes_per_minute
        if "auto_save_dialogue" in memory_raw:
            raw_auto_save = memory_raw.get("auto_save_dialogue")
            if not isinstance(raw_auto_save, bool):
                return _error_response(
                    status=400,
                    message="memory.auto_save_dialogue должен быть bool.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            auto_save_dialogue = raw_auto_save
        if "inbox_max_items" in memory_raw:
            raw_value = memory_raw.get("inbox_max_items")
            if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
                return _error_response(
                    status=400,
                    message="memory.inbox_max_items должен быть положительным int.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            inbox_max_items = raw_value
        if "inbox_ttl_days" in memory_raw:
            raw_value = memory_raw.get("inbox_ttl_days")
            if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
                return _error_response(
                    status=400,
                    message="memory.inbox_ttl_days должен быть положительным int.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            inbox_ttl_days = raw_value
        if "inbox_writes_per_minute" in memory_raw:
            raw_value = memory_raw.get("inbox_writes_per_minute")
            if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
                return _error_response(
                    status=400,
                    message="memory.inbox_writes_per_minute должен быть положительным int.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            inbox_writes_per_minute = raw_value
        if "embeddings_model" in memory_raw:
            raw_model = memory_raw.get("embeddings_model")
            if not isinstance(raw_model, str):
                return _error_response(
                    status=400,
                    message="memory.embeddings_model должен быть строкой.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            normalized_model = raw_model.strip()
            if not normalized_model:
                return _error_response(
                    status=400,
                    message="memory.embeddings_model не должен быть пустым.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            _save_embeddings_model_setting(normalized_model)
            next_embeddings_model = normalized_model
        save_memory_config(
            MemoryConfig(
                auto_save_dialogue=auto_save_dialogue,
                inbox_max_items=inbox_max_items,
                inbox_ttl_days=inbox_ttl_days,
                inbox_writes_per_minute=inbox_writes_per_minute,
            ),
        )

    policy_raw = payload.get("policy")
    if policy_raw is not None:
        if not isinstance(policy_raw, dict):
            return _error_response(
                status=400,
                message="policy должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        profile, yolo_armed, yolo_armed_at = _load_policy_settings()
        if "profile" in policy_raw:
            profile = _normalize_policy_profile(policy_raw.get("profile"))
        if "yolo_armed" in policy_raw:
            yolo_armed_raw = policy_raw.get("yolo_armed")
            if not isinstance(yolo_armed_raw, bool):
                return _error_response(
                    status=400,
                    message="policy.yolo_armed должен быть bool.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            yolo_armed = yolo_armed_raw
        if profile == "yolo" or yolo_armed:
            confirm_raw = policy_raw.get("yolo_confirm")
            confirm_text_raw = policy_raw.get("yolo_confirm_text")
            confirm_ok = (
                confirm_raw is True
                and isinstance(confirm_text_raw, str)
                and confirm_text_raw.strip().upper() == "YOLO"
            )
            if not confirm_ok:
                return _error_response(
                    status=400,
                    message=(
                        "Для включения YOLO требуется подтверждение "
                        "(yolo_confirm=true, yolo_confirm_text='YOLO')."
                    ),
                    error_type="invalid_request_error",
                    code="yolo_confirmation_required",
                )
        if yolo_armed:
            yolo_armed_at = _utc_now_iso()
        else:
            yolo_armed_at = None
        _save_policy_settings(
            profile=profile,
            yolo_armed=yolo_armed,
            yolo_armed_at=yolo_armed_at,
        )
        next_policy_profile = profile
        next_yolo_armed = yolo_armed
        next_yolo_armed_at = yolo_armed_at

    next_tools_state: dict[str, bool] | None = None
    if next_policy_profile is not None:
        profile_base = _tools_state_for_profile(next_policy_profile, _load_tools_state())
        _save_tools_state(profile_base)
        next_tools_state = dict(profile_base)

    tools_raw = payload.get("tools")
    if tools_raw is not None:
        if not isinstance(tools_raw, dict):
            return _error_response(
                status=400,
                message="tools должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        state_raw = tools_raw.get("state", tools_raw)
        if not isinstance(state_raw, dict):
            return _error_response(
                status=400,
                message="tools.state должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        current_state = next_tools_state if next_tools_state is not None else _load_tools_state()
        next_state = dict(current_state)
        for key, raw_value in state_raw.items():
            if not isinstance(key, str) or key not in DEFAULT_TOOLS_STATE:
                return _error_response(
                    status=400,
                    message=f"Неизвестный tools ключ: {key}",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            if not isinstance(raw_value, bool):
                return _error_response(
                    status=400,
                    message=f"tools.{key} должен быть bool.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            next_state[key] = raw_value
        _save_tools_state(next_state)
        next_tools_state = dict(next_state)

    providers_raw = payload.get("providers")
    if providers_raw is not None:
        if not isinstance(providers_raw, dict):
            return _error_response(
                status=400,
                message="providers должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        next_api_keys = _load_provider_api_keys()
        for provider, provider_payload in providers_raw.items():
            if provider not in API_KEY_SETTINGS_PROVIDERS:
                return _error_response(
                    status=400,
                    message=f"Неизвестный provider: {provider}",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            if provider_payload is None:
                next_api_keys.pop(provider, None)
                continue
            api_key_raw: object | None = None
            if isinstance(provider_payload, dict):
                api_key_raw = provider_payload.get("api_key")
            elif isinstance(provider_payload, str):
                api_key_raw = provider_payload
            else:
                return _error_response(
                    status=400,
                    message=f"providers.{provider} должен быть объектом или строкой.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            if api_key_raw is None:
                next_api_keys.pop(provider, None)
                continue
            if not isinstance(api_key_raw, str):
                return _error_response(
                    status=400,
                    message=f"providers.{provider}.api_key должен быть строкой.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            normalized_key = api_key_raw.strip()
            if normalized_key:
                next_api_keys[provider] = normalized_key
            else:
                next_api_keys.pop(provider, None)
        _save_provider_api_keys(next_api_keys)

    session_hint = _extract_ui_session_id(request)
    if session_hint and (next_policy_profile is not None or next_yolo_armed is not None):
        hub: UIHub = request.app["ui_hub"]
        session_id = await hub.get_or_create_session(session_hint)
        await hub.set_session_policy(
            session_id,
            profile=next_policy_profile,
            yolo_armed=next_yolo_armed,
            yolo_armed_at=next_yolo_armed_at,
        )

    if next_tools_state is not None:
        agent_lock: asyncio.Lock = request.app["agent_lock"]
        try:
            agent = await _resolve_agent(request)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to resolve agent for live tools update",
                exc_info=True,
                extra={"error": str(exc)},
            )
            agent = None
        if agent is not None:
            update_tools_enabled = getattr(agent, "update_tools_enabled", None)
            set_embeddings_model = getattr(agent, "set_embeddings_model", None)
            if callable(update_tools_enabled) or (
                callable(set_embeddings_model) and next_embeddings_model is not None
            ):
                async with agent_lock:
                    if callable(set_embeddings_model) and next_embeddings_model is not None:
                        set_embeddings_model(next_embeddings_model)
                    if callable(update_tools_enabled):
                        update_tools_enabled(next_tools_state)
    elif next_embeddings_model is not None:
        agent_lock = request.app["agent_lock"]
        try:
            agent = await _resolve_agent(request)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to resolve agent for embeddings update",
                exc_info=True,
                extra={"error": str(exc)},
            )
            agent = None
        if agent is not None:
            set_embeddings_model = getattr(agent, "set_embeddings_model", None)
            if callable(set_embeddings_model):
                async with agent_lock:
                    set_embeddings_model(next_embeddings_model)

    return _json_response(_build_settings_payload())


def _openai_error_message(response: requests.Response) -> str | None:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    error_raw = payload.get("error")
    if not isinstance(error_raw, dict):
        return None
    message_raw = error_raw.get("message")
    if not isinstance(message_raw, str):
        return None
    normalized = message_raw.strip()
    return normalized or None


async def handle_ui_stt_transcribe(request: web.Request) -> web.Response:
    api_key = _resolve_provider_api_key("openai")
    if not api_key:
        return _error_response(
            status=409,
            message="Не задан OpenAI API key для STT (settings.providers.openai.api_key).",
            error_type="configuration_error",
            code="stt_api_key_missing",
        )

    try:
        reader = await request.multipart()
    except Exception:  # noqa: BLE001
        return _error_response(
            status=400,
            message="Ожидался multipart/form-data с полем audio.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    audio_bytes: bytes | None = None
    audio_filename = "recording.webm"
    audio_content_type = "application/octet-stream"
    language = "ru"
    while True:
        part = await reader.next()
        if part is None:
            break
        if not isinstance(part, BodyPartReader):
            continue
        name = str(getattr(part, "name", "") or "").strip()
        if not name:
            continue
        if name == "language":
            try:
                language_raw = await part.text()
            except Exception:  # noqa: BLE001
                language_raw = ""
            normalized_language = language_raw.strip()
            if normalized_language:
                language = normalized_language
            continue
        if name != "audio":
            continue
        audio_filename_raw = getattr(part, "filename", None)
        if isinstance(audio_filename_raw, str) and audio_filename_raw.strip():
            audio_filename = audio_filename_raw.strip()
        part_content_type = part.headers.get("Content-Type")
        if isinstance(part_content_type, str) and part_content_type.strip():
            audio_content_type = part_content_type.strip()
        chunks: list[bytes] = []
        total_size = 0
        while True:
            chunk = await part.read_chunk()
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_STT_AUDIO_BYTES:
                return _error_response(
                    status=413,
                    message="Аудиофайл слишком большой.",
                    error_type="invalid_request_error",
                    code="payload_too_large",
                )
            chunks.append(chunk)
        if chunks:
            audio_bytes = b"".join(chunks)

    if audio_bytes is None:
        return _error_response(
            status=400,
            message="Поле audio обязательно.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    try:
        response = requests.post(
            OPENAI_STT_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": "whisper-1",
                "language": language,
                "response_format": "json",
            },
            files={"file": (audio_filename, audio_bytes, audio_content_type)},
            timeout=MODEL_FETCH_TIMEOUT,
        )
    except Exception:  # noqa: BLE001
        return _error_response(
            status=502,
            message="Не удалось связаться с STT-провайдером.",
            error_type="upstream_error",
            code="upstream_error",
        )

    if response.status_code >= 400:
        upstream_message = _openai_error_message(response)
        if response.status_code in {400, 415, 422}:
            return _error_response(
                status=400,
                message=upstream_message or "Неподдерживаемый формат аудио.",
                error_type="invalid_request_error",
                code="unsupported_audio_format",
            )
        return _error_response(
            status=502,
            message=upstream_message or "STT-провайдер вернул ошибку.",
            error_type="upstream_error",
            code="upstream_error",
        )

    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = None
    if not isinstance(payload, dict):
        return _error_response(
            status=502,
            message="STT-провайдер вернул неожиданный ответ.",
            error_type="upstream_error",
            code="upstream_error",
        )
    text_raw = payload.get("text")
    if not isinstance(text_raw, str) or not text_raw.strip():
        return _error_response(
            status=502,
            message="STT-провайдер не вернул текст распознавания.",
            error_type="upstream_error",
            code="upstream_error",
        )
    return _json_response(
        {
            "text": text_raw.strip(),
            "model": "whisper-1",
            "language": language,
        }
    )


async def handle_ui_chats_export(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    sessions = await hub.export_sessions()
    payload_sessions = [_serialize_persisted_session(item) for item in sessions]
    return _json_response(
        {
            "exported_at": _utc_iso(),
            "count": len(payload_sessions),
            "sessions": payload_sessions,
        },
    )


async def handle_ui_chats_import(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    sessions_raw = payload.get("sessions")
    if not isinstance(sessions_raw, list):
        return _error_response(
            status=400,
            message="sessions должен быть списком.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    mode: Literal["replace", "merge"] = "replace"
    raw_mode = payload.get("mode")
    if isinstance(raw_mode, str):
        normalized_mode = raw_mode.strip().lower()
        if normalized_mode in {"replace", "merge"}:
            mode = cast(Literal["replace", "merge"], normalized_mode)
        else:
            return _error_response(
                status=400,
                message="mode должен быть replace или merge.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )

    sessions: list[PersistedSession] = []
    for index, item in enumerate(sessions_raw):
        parsed = _parse_imported_session(item)
        if parsed is None:
            return _error_response(
                status=400,
                message=f"sessions[{index}] имеет некорректный формат.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        sessions.append(parsed)
    imported = await hub.import_sessions(sessions, mode=mode)
    return _json_response(
        {
            "imported": imported,
            "mode": mode,
        },
    )


async def handle_ui_models(request: web.Request) -> web.Response:
    provider_query = request.query.get("provider", "").strip().lower()
    providers: list[str]
    if provider_query:
        normalized = _normalize_provider(provider_query)
        if normalized is None:
            return _error_response(
                status=400,
                message=f"Неизвестный провайдер: {provider_query}",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        providers = [normalized]
    else:
        providers = sorted(SUPPORTED_MODEL_PROVIDERS)
    payload_items: list[dict[str, JSONValue]] = []
    for provider in providers:
        models, error_text = _fetch_provider_models(provider)
        payload_items.append({"provider": provider, "models": models, "error": error_text})
    return _json_response({"providers": payload_items})


async def handle_ui_session_model(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    provider_raw = str(payload.get("provider") or "").strip()
    model_raw = str(payload.get("model") or "").strip()
    if not provider_raw or not model_raw:
        return _error_response(
            status=400,
            message="Нужны provider и model.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    provider = _normalize_provider(provider_raw)
    if provider is None:
        return _error_response(
            status=400,
            message=f"Неизвестный провайдер: {provider_raw}",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    models, fetch_error = _fetch_provider_models(provider)
    if fetch_error:
        return _error_response(
            status=502,
            message=fetch_error,
            error_type="provider_error",
            code="provider_models_unavailable",
        )
    if model_raw not in models:
        suggestion = _closest_model_suggestion(model_raw, models)
        details: dict[str, JSONValue] = {
            "provider": provider,
            "model": model_raw,
            "suggestion": suggestion,
            "available_count": len(models),
        }
        message = (
            f"сам придумал, сам и страдай. "
            f"Модель '{model_raw}' не найдена у провайдера '{provider}'."
        )
        if suggestion:
            message = f"{message} Возможно, вы имели в виду '{suggestion}'."
        return _error_response(
            status=404,
            message=message,
            error_type="invalid_request_error",
            code="model_not_found",
            details=details,
        )
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    await hub.set_session_model(session_id, provider, model_raw)
    response = _json_response(
        {"session_id": session_id, "selected_model": {"provider": provider, "model": model_raw}}
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_sessions_list(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    sessions = await hub.list_sessions()
    serialized_sessions: list[dict[str, JSONValue]] = [
        {
            "session_id": item["session_id"],
            "title": item["title"],
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "message_count": item["message_count"],
            "title_override": item["title_override"],
            "folder_id": item["folder_id"],
        }
        for item in sessions
    ]
    return _json_response({"sessions": serialized_sessions})


async def handle_ui_folders_list(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    folders = await hub.list_folders()
    return _json_response({"folders": folders})


async def handle_ui_folders_create(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    name_raw = payload.get("name")
    if not isinstance(name_raw, str) or not name_raw.strip():
        return _error_response(
            status=400,
            message="name должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        folder = await hub.create_folder(name_raw)
    except ValueError:
        return _error_response(
            status=400,
            message="name должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    return _json_response({"folder": folder})


async def handle_ui_sessions_create(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.create_session()
    session = await hub.get_session(session_id)
    if session is None:
        return _error_response(
            status=500,
            message="Failed to create session.",
            error_type="internal_error",
            code="session_create_failed",
        )
    response = _json_response({"session": session})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    session = await hub.get_session(session_id)
    if session is None:
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    raw_decision = session.get("decision")
    session["decision"] = _normalize_ui_decision(raw_decision, session_id=session_id)
    response = _json_response({"session": session})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_history_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    messages = await hub.get_session_history(session_id)
    if messages is None:
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return _json_response({"session_id": session_id, "messages": messages})


async def handle_ui_session_output_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    output = await hub.get_session_output(session_id)
    if output is None:
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return _json_response({"session_id": session_id, "output": output})


async def handle_ui_session_files_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    files = await hub.get_session_files(session_id)
    if files is None:
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return _json_response({"session_id": session_id, "files": files})


async def handle_ui_session_file_download(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    files = await hub.get_session_files(session_id)
    if files is None:
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    path_raw = request.query.get("path", "").strip()
    if not path_raw:
        return _error_response(
            status=400,
            message="path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if path_raw not in files:
        return _error_response(
            status=404,
            message="File not found in session.",
            error_type="invalid_request_error",
            code="session_file_not_found",
        )
    try:
        file_path = _resolve_workspace_file(path_raw)
        content = file_path.read_bytes()
    except FileNotFoundError:
        return _error_response(
            status=404,
            message="File not found.",
            error_type="invalid_request_error",
            code="session_file_not_found",
        )
    except ValueError as exc:
        return _error_response(
            status=400,
            message=f"Invalid path: {exc}",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    safe_name = _sanitize_download_filename(path_raw)
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    mime = _artifact_mime_from_ext(ext)
    response = web.Response(body=content, content_type=mime)
    response.headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_artifact_download(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    artifact_id = request.match_info.get("artifact_id", "").strip()
    if not session_id or not artifact_id:
        return _error_response(
            status=400,
            message="session_id и artifact_id обязательны.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    artifacts = await hub.get_session_artifacts(session_id)
    if artifacts is None:
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    target = next(
        (
            artifact
            for artifact in artifacts
            if isinstance(artifact.get("id"), str) and str(artifact.get("id")) == artifact_id
        ),
        None,
    )
    if target is None:
        return _error_response(
            status=404,
            message="Artifact not found.",
            error_type="invalid_request_error",
            code="artifact_not_found",
        )
    try:
        file_name, file_content, mime = _artifact_file_payload(target)
    except ValueError as exc:
        return _error_response(
            status=400,
            message=f"Artifact is not downloadable file: {exc}",
            error_type="invalid_request_error",
            code="artifact_not_file",
        )

    response = web.Response(body=file_content.encode("utf-8"), content_type=mime)
    response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_artifacts_download_all(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    artifacts = await hub.get_session_artifacts(session_id)
    if artifacts is None:
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )

    file_items: list[tuple[str, str, str]] = []
    for artifact in artifacts:
        try:
            file_items.append(_artifact_file_payload(artifact))
        except ValueError:
            continue
    if not file_items:
        return _error_response(
            status=404,
            message="No file artifacts for download.",
            error_type="invalid_request_error",
            code="artifact_not_file",
        )

    force_zip = request.query.get("format", "").strip().lower() == "zip"
    if len(file_items) == 1 and not force_zip:
        file_name, file_content, mime = file_items[0]
        response = web.Response(body=file_content.encode("utf-8"), content_type=mime)
        response.headers["Content-Disposition"] = f'attachment; filename="{file_name}"'
        response.headers[UI_SESSION_HEADER] = session_id
        return response

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        used_names: set[str] = set()
        for file_name, file_content, _ in file_items:
            entry_name = _safe_zip_entry_name(file_name)
            base_name = entry_name
            suffix = 1
            while entry_name in used_names:
                stem, dot, ext = base_name.partition(".")
                if dot:
                    entry_name = f"{stem}_{suffix}.{ext}"
                else:
                    entry_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(entry_name)
            archive.writestr(entry_name, file_content)
    zip_bytes = zip_buffer.getvalue()
    response = web.Response(body=zip_bytes, content_type="application/zip")
    response.headers["Content-Disposition"] = 'attachment; filename="artifacts.zip"'
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_title_update(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    title_raw = payload.get("title")
    if not isinstance(title_raw, str) or not title_raw.strip():
        return _error_response(
            status=400,
            message="title должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        result = await hub.set_session_title(session_id, title_raw)
    except KeyError:
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    except ValueError:
        return _error_response(
            status=400,
            message="title должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    return _json_response(result)


async def handle_ui_session_folder_update(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    folder_raw = payload.get("folder_id")
    folder_id: str | None
    if folder_raw is None:
        folder_id = None
    elif isinstance(folder_raw, str):
        folder_id = folder_raw.strip() or None
    else:
        return _error_response(
            status=400,
            message="folder_id должен быть строкой или null.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        result = await hub.assign_session_folder(session_id, folder_id)
    except KeyError as exc:
        if "folder" in str(exc).lower():
            return _error_response(
                status=404,
                message="Folder not found.",
                error_type="invalid_request_error",
                code="folder_not_found",
            )
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return _json_response(result)


async def handle_ui_session_delete(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = request.match_info.get("session_id", "").strip()
    if not session_id:
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    deleted = await hub.delete_session(session_id)
    if not deleted:
        return _error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return _json_response({"session_id": session_id, "deleted": True})


async def handle_ui_decision_respond(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    allowed_keys = {"session_id", "decision_id", "choice"}
    rejected_control_keys = {
        "edited_action",
        "context",
        "proposed_action",
        "resume_payload",
        "source_endpoint",
    }
    forbidden_keys = sorted(
        {
            str(key).strip()
            for key in payload
            if str(key).strip() in rejected_control_keys or str(key).strip() not in allowed_keys
        }
    )
    if forbidden_keys:
        return _error_response(
            status=400,
            message=(
                "Допустимы только session_id, decision_id, choice. "
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
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(decision_id_raw, str) or not decision_id_raw.strip():
        return _error_response(
            status=400,
            message="decision_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(choice_raw, str) or choice_raw not in UI_DECISION_RESPONSES:
        return _error_response(
            status=400,
            message="choice должен быть approve|reject.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    session_id = session_id_raw.strip()
    decision_id = decision_id_raw.strip()
    choice = choice_raw

    current_decision = _normalize_ui_decision(
        await hub.get_session_decision(session_id),
        session_id=session_id,
    )
    if current_decision is None:
        return _error_response(
            status=404,
            message="Pending decision not found.",
            error_type="invalid_request_error",
            code="decision_not_found",
        )
    current_id = current_decision.get("id")
    if not isinstance(current_id, str) or current_id != decision_id:
        return _error_response(
            status=409,
            message="Decision id mismatch.",
            error_type="invalid_request_error",
            code="decision_id_mismatch",
        )
    current_status = current_decision.get("status")
    if current_status != "pending":
        status_value = current_status if isinstance(current_status, str) else "unknown"
        return _json_response(
            {
                "ok": True,
                "decision": current_decision,
                "status": status_value,
                "resume_started": False,
                "already_resolved": status_value in {"resolved", "rejected"},
                "resume": None,
            }
        )

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
            if normalized_latest is not None:
                latest_status_raw = normalized_latest.get("status")
                latest_status = (
                    latest_status_raw if isinstance(latest_status_raw, str) else "unknown"
                )
                return _json_response(
                    {
                        "ok": True,
                        "decision": normalized_latest,
                        "status": latest_status,
                        "resume_started": False,
                        "already_resolved": latest_status in {"resolved", "rejected"},
                        "resume": None,
                    }
                )
            return _error_response(
                status=409,
                message="Decision is not pending.",
                error_type="invalid_request_error",
                code="decision_not_pending",
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
        return _json_response(
            {
                "ok": True,
                "decision": normalized,
                "status": resolved_status,
                "resume_started": False,
                "already_resolved": resolved_status in {"resolved", "rejected"},
                "resume": None,
            }
        )

    return _error_response(
        status=409,
        message="Decision resume disabled by emergency lockdown.",
        error_type="invalid_request_error",
        code="decision_resume_disabled_emergency",
    )


async def handle_ui_chat_send(request: web.Request) -> web.Response:
    agent_lock = request.app["agent_lock"]
    session_store = request.app["session_store"]
    hub: UIHub = request.app["ui_hub"]

    session_id: str | None = None
    status_opened = False
    error = False
    try:
        try:
            agent = await _resolve_agent(request)
        except ModelNotAllowedError as exc:
            return _model_not_allowed_response(exc.model_id)
        if agent is None:
            return _model_not_selected_response()

        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            return _error_response(
                status=400,
                message=f"Некорректный JSON: {exc}",
                error_type="invalid_request_error",
                code="invalid_json",
            )
        if not isinstance(payload, dict):
            return _error_response(
                status=400,
                message="JSON должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_json",
            )

        content_raw = payload.get("content")
        if not isinstance(content_raw, str):
            return _error_response(
                status=400,
                message="content должен быть строкой.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        if len(content_raw) > MAX_CONTENT_CHARS:
            return _error_response(
                status=413,
                message="content слишком длинный.",
                error_type="invalid_request_error",
                code="payload_too_large",
            )
        force_canvas_raw = payload.get("force_canvas")
        if force_canvas_raw is None:
            force_canvas = False
        elif isinstance(force_canvas_raw, bool):
            force_canvas = force_canvas_raw
        else:
            return _error_response(
                status=400,
                message="force_canvas должен быть boolean.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        attachments_raw = payload.get("attachments")
        try:
            attachments, attachments_chars = _parse_ui_chat_attachments(attachments_raw)
        except ValueError as exc:
            return _error_response(
                status=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        except OverflowError as exc:
            return _error_response(
                status=413,
                message=str(exc),
                error_type="invalid_request_error",
                code="payload_too_large",
            )
        if not content_raw.strip() and not attachments:
            return _error_response(
                status=400,
                message="Нужно передать content или attachments.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        total_payload_chars = len(content_raw) + attachments_chars
        if total_payload_chars > MAX_TOTAL_PAYLOAD_CHARS:
            return _error_response(
                status=413,
                message="Запрос слишком большой.",
                error_type="invalid_request_error",
                code="payload_too_large",
            )

        session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
        selected_model = await hub.get_session_model(session_id)
        if selected_model is None:
            return _model_not_selected_response()
        workflow = await hub.get_session_workflow(session_id)
        mode = _normalize_mode_value(workflow.get("mode"), default="ask")
        active_plan = _normalize_plan_payload(workflow.get("active_plan"))
        active_task = _normalize_task_payload(workflow.get("active_task"))

        await hub.set_session_status(session_id, "busy")
        status_opened = True
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="request.received",
            detail="chat",
        )

        approved_categories = await session_store.get_categories(session_id)
        user_message = hub.create_message(
            role="user",
            content=content_raw.strip(),
            attachments=attachments,
        )
        await hub.append_message(session_id, user_message)
        user_message_id_raw = user_message.get("message_id")
        user_message_id = (
            user_message_id_raw
            if isinstance(user_message_id_raw, str) and user_message_id_raw
            else None
        )

        if mode == "ask" and _request_likely_action_intent(content_raw):
            stop_text = _build_stop_text(
                happened="Запрос на действие заблокирован в ask-режиме.",
                why="В ask режиме запрещены tool actions и исполнение команд.",
                steps=[
                    "Переключитесь в Plan для read-only аудита.",
                    "Сформируйте и approve план.",
                    "Перейдите в Act для выполнения плана.",
                ],
                code="ASK_MODE_NO_ACTIONS",
            )
            chat_stream_id = uuid.uuid4().hex
            await _publish_chat_stream_from_text(
                hub,
                session_id=session_id,
                stream_id=chat_stream_id,
                content=stop_text,
            )
            await hub.set_session_output(session_id, stop_text)
            assistant_message = hub.create_message(
                role="assistant",
                content=stop_text,
                trace_id=None,
                parent_user_message_id=user_message_id,
            )
            await hub.append_message(session_id, assistant_message)
            messages = await hub.get_messages(session_id)
            output_payload = await hub.get_session_output(session_id)
            files_payload = await hub.get_session_files(session_id)
            artifacts_payload = await hub.get_session_artifacts(session_id)
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response = _json_response(
                {
                    "session_id": session_id,
                    "messages": messages,
                    "output": output_payload,
                    "files": files_payload or [],
                    "artifacts": artifacts_payload or [],
                    "display": {
                        "target": "chat",
                        "artifact_id": None,
                        "forced": force_canvas,
                    },
                    "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                    "selected_model": current_model,
                    "trace_id": None,
                    "approval_request": None,
                    "mwv_report": None,
                    "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                    "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                    "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                }
            )
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="chat",
            )
            return response

        if not attachments and _request_likely_web_intent(content_raw):
            guidance_text = (
                "Для веб-поиска используй команду `/web <запрос>` в чате. "
                "После этого подтвердите approval, если он потребуется."
            )
            chat_stream_id = uuid.uuid4().hex
            await _publish_chat_stream_from_text(
                hub,
                session_id=session_id,
                stream_id=chat_stream_id,
                content=guidance_text,
            )
            await hub.set_session_output(session_id, guidance_text)
            assistant_message = hub.create_message(
                role="assistant",
                content=guidance_text,
                trace_id=None,
                parent_user_message_id=user_message_id,
            )
            await hub.append_message(session_id, assistant_message)
            messages = await hub.get_messages(session_id)
            output_payload = await hub.get_session_output(session_id)
            files_payload = await hub.get_session_files(session_id)
            artifacts_payload = await hub.get_session_artifacts(session_id)
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response = _json_response(
                {
                    "session_id": session_id,
                    "messages": messages,
                    "output": output_payload,
                    "files": files_payload or [],
                    "artifacts": artifacts_payload or [],
                    "display": {
                        "target": "chat",
                        "artifact_id": None,
                        "forced": force_canvas,
                    },
                    "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                    "selected_model": current_model,
                    "trace_id": None,
                    "approval_request": None,
                    "mwv_report": None,
                    "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
                    "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
                    "active_task": _normalize_task_payload(workflow.get("active_task")),
                }
            )
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="chat",
            )
            return response

        llm_messages = _ui_messages_to_llm(await hub.get_messages(session_id))
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="context.prepared",
            detail="chat",
        )

        mwv_report: dict[str, JSONValue] | None = None
        trace_id: str | None = None
        approval_request: dict[str, JSONValue] | None = None
        ui_decision: dict[str, JSONValue] | None = None
        chat_stream_id = uuid.uuid4().hex
        live_stream_sent = False
        async with agent_lock:
            previous_trace_id = _normalize_trace_id(
                getattr(agent, "last_chat_interaction_id", None)
            )
            try:
                model_config = _build_model_config(
                    selected_model["provider"],
                    selected_model["model"],
                )
                api_key = _resolve_provider_api_key(selected_model["provider"])
                agent.reconfigure_models(model_config, main_api_key=api_key, persist=False)
            except Exception as exc:  # noqa: BLE001
                return _error_response(
                    status=400,
                    message=f"Не удалось применить модель сессии: {exc}",
                    error_type="configuration_error",
                    code="model_config_invalid",
                )
            try:
                await _apply_agent_runtime_state(agent=agent, hub=hub, session_id=session_id)
                agent.set_session_context(session_id, approved_categories)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to set session context for ui",
                    exc_info=True,
                    extra={
                        "session_id": session_id,
                        "approved_categories": sorted(approved_categories),
                        "error": str(exc),
                    },
                )
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="agent.respond.start",
                detail="chat",
            )
            request_prefers_canvas = force_canvas or _request_likely_canvas(content_raw)
            response_raw: str
            respond_stream_method = getattr(agent, "respond_stream", None)
            if callable(respond_stream_method):
                stream_chunks: list[str] = []
                pending_chat_chunks: list[str] = []
                chat_stream_mode: Literal["pending", "chat", "canvas"] = (
                    "canvas" if request_prefers_canvas else "pending"
                )
                chat_content_stream_open = False
                canvas_status_stream_open = False
                canvas_status_chars = 0
                next_canvas_status_at = CANVAS_STATUS_CHARS_STEP
                try:
                    for delta in respond_stream_method(llm_messages):
                        if not isinstance(delta, str) or not delta:
                            continue
                        stream_chunks.append(delta)
                        if chat_stream_mode == "canvas":
                            canvas_status_chars += len(delta)
                            if not canvas_status_stream_open:
                                await _publish_chat_stream_start(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                )
                                await _publish_chat_stream_delta(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    delta="Статус: формирую результат в Canvas...",
                                )
                                live_stream_sent = True
                                canvas_status_stream_open = True
                                continue
                            if canvas_status_chars >= next_canvas_status_at:
                                await _publish_chat_stream_delta(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    delta="\nСтатус: обновляю содержимое Canvas...",
                                )
                                live_stream_sent = True
                                next_canvas_status_at += CANVAS_STATUS_CHARS_STEP
                            continue
                        if chat_stream_mode == "chat":
                            for chunk in _split_chat_stream_chunks(delta):
                                await _publish_chat_stream_delta(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    delta=chunk,
                                )
                                live_stream_sent = True
                                await asyncio.sleep(0.005)
                            continue
                        pending_chat_chunks.append(delta)
                        pending_preview = "".join(pending_chat_chunks)
                        if _stream_preview_indicates_canvas(pending_preview):
                            chat_stream_mode = "canvas"
                            continue
                        if not _stream_preview_ready_for_chat(pending_preview):
                            continue
                        chat_stream_mode = "chat"
                        await _publish_chat_stream_start(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                        )
                        chat_content_stream_open = True
                        for chunk in _split_chat_stream_chunks(pending_preview):
                            await _publish_chat_stream_delta(
                                hub,
                                session_id=session_id,
                                stream_id=chat_stream_id,
                                delta=chunk,
                            )
                            live_stream_sent = True
                            await asyncio.sleep(0.005)
                        pending_chat_chunks = []
                    if chat_content_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                        )
                        chat_content_stream_open = False
                    if canvas_status_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                        )
                        canvas_status_stream_open = False
                    response_raw_candidate = getattr(agent, "last_stream_response_raw", None)
                    if isinstance(response_raw_candidate, str) and response_raw_candidate.strip():
                        response_raw = response_raw_candidate
                    else:
                        response_raw = "".join(stream_chunks)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Live stream failed; fallback to regular respond",
                        exc_info=True,
                        extra={"session_id": session_id, "error": str(exc)},
                    )
                    if chat_content_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                        )
                        chat_content_stream_open = False
                    if canvas_status_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                        )
                        canvas_status_stream_open = False
                    response_raw = agent.respond(llm_messages)
            else:
                response_raw = agent.respond(llm_messages)
            response_text, mwv_report = _split_response_and_report(response_raw)
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="agent.respond.end",
                detail="chat",
            )
            decision = _extract_decision_payload(response_text)
            trace_id = _normalize_trace_id(getattr(agent, "last_chat_interaction_id", None))
            if trace_id == previous_trace_id:
                # Не используем trace предыдущего ответа для вывода файлов текущего ответа.
                trace_id = None
            approval_request = _serialize_approval_request(
                getattr(agent, "last_approval_request", None),
            )
            ui_decision = _normalize_ui_decision(
                decision,
                session_id=session_id,
                trace_id=trace_id,
            )
            if approval_request is not None:
                ui_decision = _build_ui_approval_decision(
                    approval_request=approval_request,
                    session_id=session_id,
                    source_endpoint="chat.send",
                    resume_payload={
                        "source_request": {
                            "content": content_raw,
                            "force_canvas": force_canvas,
                            "attachments": attachments,
                        },
                        "user_message_id": user_message_id,
                        "selected_model_snapshot": {
                            "provider": selected_model["provider"],
                            "model": selected_model["model"],
                        },
                    },
                    trace_id=trace_id,
                    workflow_context=_decision_workflow_context(
                        mode=mode,
                        active_plan=active_plan,
                        active_task=active_task,
                    ),
                )

        if _decision_is_pending_blocking(ui_decision):
            await _set_current_plan_step_status(
                hub=hub,
                session_id=session_id,
                status="waiting_approval",
            )
            await hub.set_session_decision(session_id, ui_decision)
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="approval.required",
                detail="chat",
            )
            messages = await hub.get_messages(session_id)
            output_payload = await hub.get_session_output(session_id)
            files_payload = await hub.get_session_files(session_id)
            artifacts_payload = await hub.get_session_artifacts(session_id)
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response = _json_response(
                {
                    "session_id": session_id,
                    "messages": messages,
                    "output": output_payload,
                    "files": files_payload or [],
                    "artifacts": artifacts_payload or [],
                    "display": {
                        "target": "chat",
                        "artifact_id": None,
                        "forced": force_canvas,
                    },
                    "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                    "selected_model": current_model,
                    "trace_id": trace_id,
                    "approval_request": approval_request,
                    "mwv_report": mwv_report,
                    "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                    "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                    "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                }
            )
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="chat",
            )
            return response

        await hub.set_session_output(session_id, response_text)
        files_from_tools: list[str] = []
        if trace_id:
            tool_calls = _tool_calls_for_trace_id(trace_id) or []
            files_from_tools = _extract_files_from_tool_calls(tool_calls)
            if files_from_tools:
                await hub.merge_session_files(session_id, files_from_tools)
        named_files_count = len(_extract_named_files_from_output(response_text))
        display_target: Literal["chat", "canvas"]
        if approval_request is not None:
            display_target = "chat"
        else:
            display_target = (
                "canvas"
                if _should_render_result_in_canvas(
                    response_text=response_text,
                    files_from_tools=files_from_tools,
                    named_files_count=named_files_count,
                    force_canvas=force_canvas,
                )
                else "chat"
            )
        artifact_payloads = _build_output_artifacts(
            response_text=response_text,
            display_target=display_target,
            files_from_tools=files_from_tools,
        )
        for artifact in artifact_payloads:
            await hub.append_session_artifact(session_id, artifact)
        first_artifact = artifact_payloads[0] if artifact_payloads else None
        artifact_id_raw = first_artifact.get("id") if first_artifact is not None else None
        artifact_id = artifact_id_raw if isinstance(artifact_id_raw, str) else None
        artifact_title = _canvas_summary_title_from_artifact(first_artifact)
        if display_target == "canvas" and artifact_id is not None:
            stream_source_raw = (
                first_artifact.get("content") if first_artifact is not None else response_text
            )
            stream_source = (
                stream_source_raw if isinstance(stream_source_raw, str) else response_text
            )
            asyncio.create_task(
                _publish_canvas_stream(
                    hub,
                    session_id=session_id,
                    artifact_id=artifact_id,
                    content=stream_source,
                )
            )
        if display_target == "chat" and not live_stream_sent:
            await _publish_chat_stream_from_text(
                hub,
                session_id=session_id,
                stream_id=chat_stream_id,
                content=response_text,
            )
        chat_summary = (
            response_text
            if approval_request is not None or display_target == "chat"
            else _build_canvas_chat_summary(artifact_title=artifact_title)
        )
        assistant_message = hub.create_message(
            role="assistant",
            content=chat_summary,
            trace_id=trace_id,
            parent_user_message_id=user_message_id,
        )
        await hub.append_message(session_id, assistant_message)
        await hub.set_session_decision(session_id, ui_decision)
        messages = await hub.get_messages(session_id)
        output_payload = await hub.get_session_output(session_id)
        files_payload = await hub.get_session_files(session_id)
        artifacts_payload = await hub.get_session_artifacts(session_id)
        current_decision = await hub.get_session_decision(session_id)
        current_model = await hub.get_session_model(session_id)
        current_workflow = await hub.get_session_workflow(session_id)
        response = _json_response(
            {
                "session_id": session_id,
                "messages": messages,
                "output": output_payload,
                "files": files_payload or [],
                "artifacts": artifacts_payload or [],
                "display": {
                    "target": display_target,
                    "artifact_id": artifact_id,
                    "forced": force_canvas,
                },
                "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                "selected_model": current_model,
                "trace_id": trace_id,
                "approval_request": approval_request,
                "mwv_report": mwv_report,
                "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                "active_task": _normalize_task_payload(current_workflow.get("active_task")),
            },
        )
        response.headers[UI_SESSION_HEADER] = session_id
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="response.ready",
            detail="chat",
        )
        return response
    except Exception as exc:  # noqa: BLE001
        error = True
        if status_opened and session_id is not None:
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="error",
                detail=f"chat: {exc}",
            )
            await hub.set_session_status(session_id, "error")
        return _error_response(
            status=500,
            message=f"Agent error: {exc}",
            error_type="internal_error",
            code="agent_error",
        )
    finally:
        if status_opened and session_id is not None and not error:
            await hub.set_session_status(session_id, "ok")


async def handle_ui_project_command(request: web.Request) -> web.Response:
    agent_lock = request.app["agent_lock"]
    session_store = request.app["session_store"]
    hub: UIHub = request.app["ui_hub"]

    session_id: str | None = None
    status_opened = False
    error = False
    try:
        try:
            agent = await _resolve_agent(request)
        except ModelNotAllowedError as exc:
            return _model_not_allowed_response(exc.model_id)
        if agent is None:
            return _model_not_selected_response()

        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001
            return _error_response(
                status=400,
                message=f"Некорректный JSON: {exc}",
                error_type="invalid_request_error",
                code="invalid_json",
            )
        if not isinstance(payload, dict):
            return _error_response(
                status=400,
                message="JSON должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_json",
            )

        command = str(payload.get("command") or "").strip().lower()
        args_raw = str(payload.get("args") or "").strip()
        if command not in UI_PROJECT_COMMANDS:
            return _error_response(
                status=400,
                message="Поддерживаются project команды: find, index, github_import.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        user_command = f"/project {command}"
        if args_raw:
            user_command = f"{user_command} {args_raw}"

        session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
        selected_model = await hub.get_session_model(session_id)
        if selected_model is None:
            return _model_not_selected_response()
        workflow = await hub.get_session_workflow(session_id)
        mode = _normalize_mode_value(workflow.get("mode"), default="ask")
        active_plan = _normalize_plan_payload(workflow.get("active_plan"))
        active_task = _normalize_task_payload(workflow.get("active_task"))
        if mode == "ask":
            return _error_response(
                status=409,
                message=(
                    "ASK_MODE_NO_ACTIONS: project команды отключены в ask-режиме. "
                    "Переключитесь в plan/act."
                ),
                error_type="invalid_request_error",
                code="ASK_MODE_NO_ACTIONS",
            )
        await hub.set_session_status(session_id, "busy")
        status_opened = True
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="request.received",
            detail="project",
        )

        approved_categories = await session_store.get_categories(session_id)
        user_message = hub.create_message(role="user", content=user_command)
        await hub.append_message(session_id, user_message)
        user_message_id_raw = user_message.get("message_id")
        user_message_id = (
            user_message_id_raw
            if isinstance(user_message_id_raw, str) and user_message_id_raw
            else None
        )
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="context.prepared",
            detail="project",
        )

        if command == "github_import":
            try:
                repo_url, branch = _parse_github_import_args(args_raw)
            except ValueError as exc:
                return _error_response(
                    status=400,
                    message=str(exc),
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            missing_categories = [
                category
                for category in UI_GITHUB_REQUIRED_CATEGORIES
                if category not in approved_categories
            ]
            if missing_categories:
                approval_request_obj = _build_github_import_approval_request(
                    session_id=session_id,
                    repo_url=repo_url,
                    branch=branch,
                    required_categories=missing_categories,
                )
                approval_payload = _serialize_approval_request(approval_request_obj)
                approval_decision = _build_ui_approval_decision(
                    approval_request=approval_payload or {},
                    session_id=session_id,
                    source_endpoint="project.command",
                    resume_payload={
                        "source_request": {
                            "command": command,
                            "args": args_raw,
                        },
                        "user_message_id": user_message_id,
                        "selected_model_snapshot": {
                            "provider": selected_model["provider"],
                            "model": selected_model["model"],
                        },
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
                await hub.set_session_decision(session_id, approval_decision)
                messages = await hub.get_messages(session_id)
                current_decision = await hub.get_session_decision(session_id)
                current_model = await hub.get_session_model(session_id)
                current_workflow = await hub.get_session_workflow(session_id)
                response = _json_response(
                    {
                        "session_id": session_id,
                        "messages": messages,
                        "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                        "selected_model": current_model,
                        "trace_id": None,
                        "approval_request": approval_payload,
                        "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                        "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                        "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                    },
                )
                response.headers[UI_SESSION_HEADER] = session_id
                await _publish_agent_activity(
                    hub,
                    session_id=session_id,
                    phase="approval.required",
                    detail="project/github_import",
                )
                return response

            try:
                target_path, relative_target = _resolve_github_target(repo_url)
            except ValueError as exc:
                return _error_response(
                    status=400,
                    message=str(exc),
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="github.clone.start",
                detail=repo_url,
            )
            cloned, clone_result = await _clone_github_repository(
                repo_url=repo_url,
                branch=branch,
                target_path=target_path,
            )
            if not cloned:
                response_text = f"Командный режим (без MWV)\n{clone_result}"
            else:
                await _publish_agent_activity(
                    hub,
                    session_id=session_id,
                    phase="github.index.start",
                    detail=relative_target,
                )
                indexed, index_result = _index_imported_project(relative_target)
                await _publish_agent_activity(
                    hub,
                    session_id=session_id,
                    phase="github.index.end",
                    detail=index_result,
                )
                if indexed:
                    response_text = (
                        "Командный режим (без MWV)\n"
                        f"GitHub import completed: {repo_url}\n"
                        f"Path: {relative_target}\n"
                        f"Index: {index_result}"
                    )
                else:
                    response_text = (
                        "Командный режим (без MWV)\n"
                        f"GitHub import completed with indexing errors: {repo_url}\n"
                        f"Path: {relative_target}\n"
                        f"Index error: {index_result}"
                    )
            await hub.append_message(
                session_id,
                hub.create_message(
                    role="assistant",
                    content=response_text,
                    parent_user_message_id=user_message_id,
                ),
            )
            await hub.set_session_decision(session_id, None)
            messages = await hub.get_messages(session_id)
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response = _json_response(
                {
                    "session_id": session_id,
                    "messages": messages,
                    "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                    "selected_model": current_model,
                    "trace_id": None,
                    "approval_request": None,
                    "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                    "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                    "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                },
            )
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="project/github_import",
            )
            return response

        llm_messages = _ui_messages_to_llm(await hub.get_messages(session_id))

        mwv_report: dict[str, JSONValue] | None = None
        ui_decision: dict[str, JSONValue] | None = None
        async with agent_lock:
            try:
                model_config = _build_model_config(
                    selected_model["provider"],
                    selected_model["model"],
                )
                api_key = _resolve_provider_api_key(selected_model["provider"])
                agent.reconfigure_models(model_config, main_api_key=api_key, persist=False)
            except Exception as exc:  # noqa: BLE001
                return _error_response(
                    status=400,
                    message=f"Не удалось применить модель сессии: {exc}",
                    error_type="configuration_error",
                    code="model_config_invalid",
                )
            try:
                await _apply_agent_runtime_state(agent=agent, hub=hub, session_id=session_id)
                agent.set_session_context(session_id, approved_categories)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to set session context for ui project command",
                    exc_info=True,
                    extra={
                        "session_id": session_id,
                        "approved_categories": sorted(approved_categories),
                        "error": str(exc),
                    },
                )
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="agent.respond.start",
                detail="project",
            )
            response_raw = agent.respond(llm_messages)
            response_text, mwv_report = _split_response_and_report(response_raw)
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="agent.respond.end",
                detail="project",
            )
            trace_id = getattr(agent, "last_chat_interaction_id", None)
            approval_request = _serialize_approval_request(
                getattr(agent, "last_approval_request", None),
            )
            decision_raw = _extract_decision_payload(response_text)
            ui_decision = _normalize_ui_decision(
                decision_raw,
                session_id=session_id,
                trace_id=_normalize_trace_id(trace_id),
            )
            if approval_request is not None:
                ui_decision = _build_ui_approval_decision(
                    approval_request=approval_request,
                    session_id=session_id,
                    source_endpoint="project.command",
                    resume_payload={
                        "source_request": {
                            "command": command,
                            "args": args_raw,
                        },
                        "user_message_id": user_message_id,
                        "selected_model_snapshot": {
                            "provider": selected_model["provider"],
                            "model": selected_model["model"],
                        },
                    },
                    trace_id=_normalize_trace_id(trace_id),
                    workflow_context=_decision_workflow_context(
                        mode=mode,
                        active_plan=active_plan,
                        active_task=active_task,
                    ),
                )

        if _decision_is_pending_blocking(ui_decision):
            await _set_current_plan_step_status(
                hub=hub,
                session_id=session_id,
                status="waiting_approval",
            )
            await hub.set_session_decision(session_id, ui_decision)
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="approval.required",
                detail="project",
            )
            messages = await hub.get_messages(session_id)
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response = _json_response(
                {
                    "session_id": session_id,
                    "messages": messages,
                    "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                    "selected_model": current_model,
                    "trace_id": trace_id,
                    "approval_request": approval_request,
                    "mwv_report": mwv_report,
                    "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                    "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                    "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                }
            )
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="project",
            )
            return response

        await hub.append_message(
            session_id,
            hub.create_message(
                role="assistant",
                content=response_text,
                trace_id=trace_id if isinstance(trace_id, str) else None,
                parent_user_message_id=user_message_id,
            ),
        )
        await hub.set_session_decision(session_id, ui_decision)
        messages = await hub.get_messages(session_id)
        current_decision = await hub.get_session_decision(session_id)
        current_model = await hub.get_session_model(session_id)
        current_workflow = await hub.get_session_workflow(session_id)
        response = _json_response(
            {
                "session_id": session_id,
                "messages": messages,
                "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                "selected_model": current_model,
                "trace_id": trace_id,
                "approval_request": approval_request,
                "mwv_report": mwv_report,
                "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                "active_task": _normalize_task_payload(current_workflow.get("active_task")),
            },
        )
        response.headers[UI_SESSION_HEADER] = session_id
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="response.ready",
            detail="project",
        )
        return response
    except Exception as exc:  # noqa: BLE001
        error = True
        if status_opened and session_id is not None:
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="error",
                detail=f"project: {exc}",
            )
            await hub.set_session_status(session_id, "error")
        return _error_response(
            status=500,
            message=f"Project command error: {exc}",
            error_type="internal_error",
            code="agent_error",
        )
    finally:
        if status_opened and session_id is not None and not error:
            await hub.set_session_status(session_id, "ok")


def _normalize_conflict_action(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower()
    if normalized in {"activate", "deprecate", "set_value"}:
        return normalized
    return None


async def handle_ui_memory_conflicts(request: web.Request) -> web.Response:
    try:
        agent = await _resolve_agent(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _json_response({"conflicts": []})

    limit_raw = request.query.get("limit", "").strip()
    limit = 50
    if limit_raw:
        try:
            limit = int(limit_raw)
        except ValueError:
            return _error_response(
                status=400,
                message="limit должен быть int.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
    if limit <= 0 or limit > 200:
        return _error_response(
            status=400,
            message="limit должен быть в диапазоне 1..200.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    list_conflicts = getattr(agent, "list_memory_conflicts", None)
    if not callable(list_conflicts):
        return _json_response({"conflicts": []})
    try:
        conflicts_raw = list_conflicts(limit)
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=500,
            message=f"Не удалось загрузить конфликты памяти: {exc}",
            error_type="internal_error",
            code="memory_conflicts_load_failed",
        )
    conflicts: list[JSONValue] = []
    if isinstance(conflicts_raw, list):
        for item in conflicts_raw:
            if isinstance(item, dict):
                normalized: dict[str, JSONValue] = {}
                for key, value in item.items():
                    normalized[str(key)] = _normalize_json_value(value)
                conflicts.append(normalized)
    return _json_response({"conflicts": conflicts})


async def handle_ui_memory_conflicts_resolve(request: web.Request) -> web.Response:
    try:
        agent = await _resolve_agent(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()

    resolve_conflict = getattr(agent, "resolve_memory_conflict", None)
    if not callable(resolve_conflict):
        return _error_response(
            status=501,
            message="Memory conflict resolver недоступен.",
            error_type="not_supported",
            code="memory_conflict_resolve_not_supported",
        )

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    stable_key_raw = payload.get("stable_key")
    action = _normalize_conflict_action(payload.get("action"))
    if not isinstance(stable_key_raw, str) or not stable_key_raw.strip():
        return _error_response(
            status=400,
            message="stable_key обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if action is None:
        return _error_response(
            status=400,
            message="action должен быть activate | deprecate | set_value.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    value_json_raw = payload.get("value_json")
    value_json = _normalize_json_value(value_json_raw)
    if action != "set_value":
        value_json = None

    agent_lock: asyncio.Lock = request.app["agent_lock"]
    try:
        async with agent_lock:
            resolved_raw = resolve_conflict(
                stable_key=stable_key_raw.strip(),
                action=action,
                value_json=value_json,
            )
    except ValueError as exc:
        return _error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=500,
            message=f"Не удалось разрешить конфликт памяти: {exc}",
            error_type="internal_error",
            code="memory_conflict_resolve_failed",
        )
    if resolved_raw is None:
        return _error_response(
            status=404,
            message="Конфликт не найден.",
            error_type="invalid_request_error",
            code="memory_conflict_not_found",
        )
    normalized: dict[str, JSONValue] = {}
    if isinstance(resolved_raw, dict):
        for key, value in resolved_raw.items():
            normalized[str(key)] = _normalize_json_value(value)
    return _json_response({"resolved": normalized})


async def handle_ui_events_stream(request: web.Request) -> web.StreamResponse:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    queue = await hub.subscribe(session_id)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            UI_SESSION_HEADER: session_id,
        },
    )
    await response.prepare(request)
    initial_status_event = await hub.get_session_status_event(session_id)
    initial_status_payload = json.dumps(initial_status_event, ensure_ascii=False)
    await response.write(f"data: {initial_status_payload}\n\n".encode())

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                payload = json.dumps(event, ensure_ascii=False)
                await response.write(f"data: {payload}\n\n".encode())
            except asyncio.TimeoutError:  # noqa: UP041
                await response.write(b": keep-alive\n\n")
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        await hub.unsubscribe(session_id, queue)
    return response


async def handle_chat_completions(request: web.Request) -> web.Response:
    try:
        agent = await _resolve_agent(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    agent_lock = request.app["agent_lock"]
    session_store = request.app["session_store"]

    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )

    parsed, error_text = _parse_chat_request(payload)
    if parsed is None:
        return _error_response(
            status=400,
            message=error_text,
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    session_id = _extract_session_id(request, payload) or parsed.session_id
    if session_id is None:
        session_id = str(uuid.uuid4())

    approved_categories = await session_store.get_categories(session_id)
    try:
        agent.set_session_context(session_id, approved_categories)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to set session context",
            exc_info=True,
            extra={
                "session_id": session_id,
                "approved_categories": sorted(approved_categories),
                "error": str(exc),
            },
        )

    if parsed.stream:
        return _error_response(
            status=400,
            message="Streaming is not supported.",
            error_type="not_supported",
            code="streaming_not_supported",
        )

    if parsed.tool_calling_present and not TOOL_PIPELINE_ENABLED:
        return _error_response(
            status=400,
            message="Tool calling is not supported.",
            error_type="not_supported",
            code="tool_calling_not_supported",
        )

    trace_id: str | None = None
    mwv_report: dict[str, JSONValue] | None = None
    try:
        async with agent_lock:
            response_raw = agent.respond(parsed.messages)
            response_text, mwv_report = _split_response_and_report(response_raw)
            trace_id = agent.last_chat_interaction_id
    except Exception as exc:  # noqa: BLE001
        return _error_response(
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
        return _error_response(
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

    if trace_id is None:
        return _error_response(
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

    response_payload: dict[str, JSONValue] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": parsed.model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": response_text},
            }
        ],
        "slavik_meta": slavik_meta,
    }
    return _json_response(response_payload)


async def handle_trace(request: web.Request) -> web.Response:
    trace_id = request.match_info.get("trace_id", "").strip()
    if not trace_id:
        return _error_response(
            status=400,
            message="trace_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    records = _parse_trace_log(TRACE_LOG)
    groups = _build_trace_groups(records)
    for group in groups:
        if group.interaction_id == trace_id:
            return _json_response(
                {"trace_id": trace_id, "events": _serialize_trace_events(group.events)},
            )
    return _error_response(
        status=404,
        message="Trace not found.",
        error_type="invalid_request_error",
        code="trace_not_found",
    )


async def handle_tool_calls(request: web.Request) -> web.Response:
    trace_id = request.match_info.get("trace_id", "").strip()
    if not trace_id:
        return _error_response(
            status=400,
            message="trace_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    tool_calls = _tool_calls_for_trace_id(trace_id)
    if tool_calls is None:
        return _error_response(
            status=404,
            message="Trace not found.",
            error_type="invalid_request_error",
            code="trace_not_found",
        )
    return _json_response({"trace_id": trace_id, "tool_calls": tool_calls})


async def handle_feedback(request: web.Request) -> web.Response:
    try:
        agent = await _resolve_agent(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )

    interaction_id = payload.get("interaction_id")
    rating_raw = payload.get("rating")
    labels_raw = payload.get("labels", [])
    free_text_raw = payload.get("free_text")

    if not isinstance(interaction_id, str) or not interaction_id.strip():
        return _error_response(
            status=400,
            message="interaction_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(rating_raw, str):
        return _error_response(
            status=400,
            message="rating должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        rating = FeedbackRating(rating_raw)
    except ValueError as exc:
        return _error_response(
            status=400,
            message=f"Некорректный rating: {exc}",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(labels_raw, list):
        return _error_response(
            status=400,
            message="labels должен быть списком.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    labels: list[FeedbackLabel] = []
    for label_raw in labels_raw:
        if not isinstance(label_raw, str):
            return _error_response(
                status=400,
                message="labels содержит нестроковое значение.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        try:
            labels.append(FeedbackLabel(label_raw))
        except ValueError as exc:
            return _error_response(
                status=400,
                message=f"Некорректный label: {exc}",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
    free_text = None
    if free_text_raw is not None:
        if not isinstance(free_text_raw, str):
            return _error_response(
                status=400,
                message="free_text должен быть строкой или null.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        free_text = free_text_raw

    try:
        agent.record_feedback_event(
            interaction_id=interaction_id,
            rating=rating,
            labels=labels,
            free_text=free_text,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=500,
            message=f"Feedback error: {exc}",
            error_type="internal_error",
            code="feedback_error",
        )

    return _json_response({"ok": True})


async def handle_approve_session(request: web.Request) -> web.Response:
    session_store = request.app["session_store"]
    agent = request.app.get("agent")
    admin_token = os.environ.get("SLAVIK_ADMIN_TOKEN", "").strip()
    if not admin_token:
        return _error_response(
            status=503,
            message="Admin token is not configured.",
            error_type="configuration_error",
            code="admin_token_not_configured",
        )
    presented_token = _extract_bearer_token(request)
    if presented_token is None or not hmac.compare_digest(presented_token, admin_token):
        return _error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return _error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return _error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    session_id = payload.get("session_id")
    categories_raw = payload.get("categories", [])
    if not isinstance(session_id, str) or not session_id.strip():
        return _error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    categories: set[ApprovalCategory] = set()
    if categories_raw is None:
        categories_raw = []
    if not isinstance(categories_raw, list):
        return _error_response(
            status=400,
            message="categories должен быть списком.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    for item in categories_raw:
        if not isinstance(item, str):
            return _error_response(
                status=400,
                message="categories содержит нестроковое значение.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        category = _CATEGORY_MAP.get(item)
        if category is None:
            return _error_response(
                status=400,
                message=f"Неизвестная категория: {item}",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        categories.add(category)

    approved_categories: set[ApprovalCategory] = set()
    if categories:
        approved_categories = await session_store.approve(session_id.strip(), categories)
        if agent is not None:
            try:
                agent.tracer.log(
                    "approval_granted",
                    "Session approval granted",
                    {
                        "session_id": session_id.strip(),
                        "categories": sorted(approved_categories),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to log approval_granted",
                    exc_info=True,
                    extra={
                        "session_id": session_id.strip(),
                        "categories": sorted(approved_categories),
                        "error": str(exc),
                    },
                )
    return _json_response(
        {
            "session_id": session_id.strip(),
            "session_approved": bool(approved_categories),
            "approved_categories": sorted(approved_categories),
        },
    )


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
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/slavik/trace/{trace_id}", handle_trace)
    app.router.add_get("/slavik/tool-calls/{trace_id}", handle_tool_calls)
    app.router.add_post("/slavik/feedback", handle_feedback)
    app.router.add_post("/slavik/approve-session", handle_approve_session)
    app.router.add_get("/", handle_ui_index)
    app.router.add_get("/workspace", handle_workspace_index)
    app.router.add_get("/ui", handle_ui_redirect)
    app.router.add_get("/ui/", handle_ui_index)
    app.router.add_get("/ui/workspace", handle_workspace_index)
    app.router.add_get("/ui/api/status", handle_ui_status)
    app.router.add_get("/ui/api/state", handle_ui_state)
    app.router.add_post("/ui/api/mode", handle_ui_mode)
    app.router.add_post("/ui/api/plan/draft", handle_ui_plan_draft)
    app.router.add_post("/ui/api/plan/approve", handle_ui_plan_approve)
    app.router.add_post("/ui/api/plan/execute", handle_ui_plan_execute)
    app.router.add_post("/ui/api/plan/cancel", handle_ui_plan_cancel)
    app.router.add_get("/ui/api/settings", handle_ui_settings)
    app.router.add_post("/ui/api/settings", handle_ui_settings_update)
    app.router.add_get("/ui/api/memory/conflicts", handle_ui_memory_conflicts)
    app.router.add_post("/ui/api/memory/conflicts/resolve", handle_ui_memory_conflicts_resolve)
    app.router.add_post("/ui/api/stt/transcribe", handle_ui_stt_transcribe)
    app.router.add_get("/ui/api/settings/chats/export", handle_ui_chats_export)
    app.router.add_post("/ui/api/settings/chats/import", handle_ui_chats_import)
    app.router.add_get("/ui/api/models", handle_ui_models)
    app.router.add_get("/ui/api/folders", handle_ui_folders_list)
    app.router.add_post("/ui/api/folders", handle_ui_folders_create)
    app.router.add_get("/ui/api/sessions", handle_ui_sessions_list)
    app.router.add_post("/ui/api/sessions", handle_ui_sessions_create)
    app.router.add_get("/ui/api/sessions/{session_id}", handle_ui_session_get)
    app.router.add_get("/ui/api/sessions/{session_id}/history", handle_ui_session_history_get)
    app.router.add_get("/ui/api/sessions/{session_id}/output", handle_ui_session_output_get)
    app.router.add_get("/ui/api/sessions/{session_id}/files", handle_ui_session_files_get)
    app.router.add_post("/ui/api/decision/respond", handle_ui_decision_respond)
    app.router.add_get(
        "/ui/api/sessions/{session_id}/files/download",
        handle_ui_session_file_download,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/artifacts/download-all",
        handle_ui_session_artifacts_download_all,
    )
    app.router.add_get(
        "/ui/api/sessions/{session_id}/artifacts/{artifact_id}/download",
        handle_ui_session_artifact_download,
    )
    app.router.add_delete("/ui/api/sessions/{session_id}", handle_ui_session_delete)
    app.router.add_patch("/ui/api/sessions/{session_id}/title", handle_ui_session_title_update)
    app.router.add_put("/ui/api/sessions/{session_id}/folder", handle_ui_session_folder_update)
    app.router.add_post("/ui/api/session-model", handle_ui_session_model)
    app.router.add_get("/ui/api/workspace/root", handle_ui_workspace_root_get)
    app.router.add_post("/ui/api/workspace/root/select", handle_ui_workspace_root_select)
    app.router.add_post("/ui/api/workspace/index", handle_ui_workspace_index_run)
    app.router.add_get("/ui/api/workspace/git-diff", handle_ui_workspace_git_diff)
    app.router.add_get("/ui/api/workspace/tree", handle_ui_workspace_tree)
    app.router.add_get("/ui/api/workspace/file", handle_ui_workspace_file_get)
    app.router.add_put("/ui/api/workspace/file", handle_ui_workspace_file_put)
    app.router.add_post("/ui/api/workspace/run", handle_ui_workspace_run)
    app.router.add_post("/ui/api/chat/send", handle_ui_chat_send)
    app.router.add_post("/ui/api/tools/project", handle_ui_project_command)
    app.router.add_get("/ui/api/events/stream", handle_ui_events_stream)
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
