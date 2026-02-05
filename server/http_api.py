from __future__ import annotations

import asyncio
import difflib
import importlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import requests
from aiohttp import web

from config.http_server_config import (
    DEFAULT_MAX_REQUEST_BYTES,
    HttpServerConfig,
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
from core.approval_policy import ALL_CATEGORIES, ApprovalCategory, ApprovalRequest
from core.tracer import TRACE_LOG, TraceRecord
from llm.local_http_brain import DEFAULT_LOCAL_ENDPOINT
from llm.types import ModelConfig
from server.lazy_agent import LazyAgentProvider
from server.ui_hub import UIHub
from server.ui_session_storage import PersistedSession, SQLiteUISessionStorage, UISessionStorage
from shared.memory_companion_models import FeedbackLabel, FeedbackRating
from shared.models import JSONValue, LLMMessage
from shared.sanitize import safe_json_loads
from tools.tool_logger import DEFAULT_LOG_PATH as TOOL_CALLS_LOG

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
SUPPORTED_MODEL_PROVIDERS: Final[set[str]] = {"xai", "openrouter", "local"}
XAI_MODELS_ENDPOINT: Final[str] = "https://api.x.ai/v1/models"
OPENROUTER_MODELS_ENDPOINT: Final[str] = "https://openrouter.ai/api/v1/models"
MODEL_FETCH_TIMEOUT: Final[int] = 20
UI_PROJECT_COMMANDS: Final[set[str]] = {"find", "index"}
UI_SETTINGS_PATH: Final[Path] = Path(__file__).resolve().parent.parent / ".run" / "ui_settings.json"
DEFAULT_UI_TONE: Final[str] = "balanced"
DEFAULT_EMBEDDINGS_MODEL: Final[str] = "all-MiniLM-L6-v2"


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
        api_key = os.getenv("XAI_API_KEY", "").strip()
        if not api_key:
            return {}, "Не задан XAI_API_KEY."
        return {"Authorization": f"Bearer {api_key}"}, None
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            return {}, "Не задан OPENROUTER_API_KEY."
        return {"Authorization": f"Bearer {api_key}"}, None
    if provider == "local":
        api_key = os.getenv("LOCAL_LLM_API_KEY", "").strip()
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


def _ui_messages_to_llm(messages: list[dict[str, str]]) -> list[LLMMessage]:
    parsed: list[LLMMessage] = []
    for item in messages:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            parsed.append(LLMMessage(role="user", content=content))
        elif role == "assistant":
            parsed.append(LLMMessage(role="assistant", content=content))
        elif role == "system":
            parsed.append(LLMMessage(role="system", content=content))
    return parsed


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


def _load_tools_state() -> dict[str, bool]:
    try:
        return load_tools_config().to_dict()
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_TOOLS_STATE)


def _save_tools_state(state: dict[str, bool]) -> None:
    payload: dict[str, object] = {key: value for key, value in state.items()}
    save_tools_config(ToolsConfig.from_dict(payload))


def _provider_settings_payload() -> list[dict[str, JSONValue]]:
    local_endpoint = (
        os.getenv("LOCAL_LLM_URL", DEFAULT_LOCAL_ENDPOINT).strip() or DEFAULT_LOCAL_ENDPOINT
    )
    return [
        {
            "provider": "xai",
            "api_key_env": "XAI_API_KEY",
            "api_key_set": bool(os.getenv("XAI_API_KEY", "").strip()),
            "endpoint": XAI_MODELS_ENDPOINT,
        },
        {
            "provider": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY",
            "api_key_set": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
            "endpoint": OPENROUTER_MODELS_ENDPOINT,
        },
        {
            "provider": "local",
            "api_key_env": "LOCAL_LLM_API_KEY",
            "api_key_set": bool(os.getenv("LOCAL_LLM_API_KEY", "").strip()),
            "endpoint": local_endpoint,
        },
    ]


def _build_settings_payload() -> dict[str, JSONValue]:
    tone, system_prompt = _load_personalization_settings()
    memory_config = load_memory_config()
    tools_state = _load_tools_state()
    tools_registry = {key: value for key, value in tools_state.items() if key != "safe_mode"}
    return {
        "settings": {
            "personalization": {"tone": tone, "system_prompt": system_prompt},
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
            "providers": _provider_settings_payload(),
        },
    }


def _normalize_import_status(raw: object) -> str:
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"ok", "busy", "error"}:
            return normalized
    return "ok"


def _parse_imported_message(raw: object) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    role_raw = raw.get("role")
    content_raw = raw.get("content")
    if not isinstance(role_raw, str) or role_raw not in {"user", "assistant", "system"}:
        return None
    if not isinstance(content_raw, str):
        return None
    return {"role": role_raw, "content": content_raw}


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
    messages: list[dict[str, str]] = []
    for item in messages_raw:
        parsed_message = _parse_imported_message(item)
        if parsed_message is None:
            return None
        messages.append(parsed_message)
    decision_raw = raw.get("decision")
    decision: dict[str, JSONValue] | None = None
    if isinstance(decision_raw, dict):
        parsed_decision: dict[str, JSONValue] = {}
        for key, item in decision_raw.items():
            parsed_decision[str(key)] = _normalize_json_value(item)
        decision = parsed_decision
    selected_model_raw = raw.get("selected_model")
    model_provider: str | None = None
    model_id: str | None = None
    if isinstance(selected_model_raw, dict):
        provider_raw = selected_model_raw.get("provider")
        model_raw = selected_model_raw.get("model")
        if isinstance(provider_raw, str) and provider_raw.strip():
            model_provider = provider_raw.strip()
        if isinstance(model_raw, str) and model_raw.strip():
            model_id = model_raw.strip()
    return PersistedSession(
        session_id=session_id,
        created_at=created_at,
        updated_at=updated_at,
        status=_normalize_import_status(raw.get("status")),
        decision=decision,
        messages=messages,
        model_provider=model_provider,
        model_id=model_id,
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
        "decision": dict(session.decision) if session.decision is not None else None,
        "selected_model": selected_model,
    }
    return payload


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


async def handle_ui_redirect(request: web.Request) -> web.StreamResponse:
    raise web.HTTPFound("/ui/")


async def handle_ui_index(request: web.Request) -> web.FileResponse:
    dist_path: Path = request.app["ui_dist_path"]
    index_path = dist_path / "index.html"
    return web.FileResponse(path=index_path)


async def handle_ui_status(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
    decision = await hub.get_session_decision(session_id)
    selected_model = await hub.get_session_model(session_id)
    response = _json_response(
        {
            "ok": True,
            "session_id": session_id,
            "decision": decision,
            "selected_model": selected_model,
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
        save_memory_config(
            MemoryConfig(
                auto_save_dialogue=auto_save_dialogue,
                inbox_max_items=inbox_max_items,
                inbox_ttl_days=inbox_ttl_days,
                inbox_writes_per_minute=inbox_writes_per_minute,
            ),
        )

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
        current_state = _load_tools_state()
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

    return _json_response(_build_settings_payload())


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
            "created_at": item["created_at"],
            "updated_at": item["updated_at"],
            "message_count": item["message_count"],
        }
        for item in sessions
    ]
    return _json_response({"sessions": serialized_sessions})


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
    response = _json_response({"session": session})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


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
        if not isinstance(content_raw, str) or not content_raw.strip():
            return _error_response(
                status=400,
                message="content должен быть непустой строкой.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )

        session_id = await hub.get_or_create_session(_extract_ui_session_id(request))
        selected_model = await hub.get_session_model(session_id)
        if selected_model is None:
            return _model_not_selected_response()

        await hub.set_session_status(session_id, "busy")
        status_opened = True

        approved_categories = await session_store.get_categories(session_id)
        await hub.append_message(session_id, "user", content_raw.strip())
        llm_messages = _ui_messages_to_llm(await hub.get_messages(session_id))

        async with agent_lock:
            try:
                model_config = _build_model_config(
                    selected_model["provider"],
                    selected_model["model"],
                )
                agent.reconfigure_models(model_config, persist=False)
            except Exception as exc:  # noqa: BLE001
                return _error_response(
                    status=400,
                    message=f"Не удалось применить модель сессии: {exc}",
                    error_type="configuration_error",
                    code="model_config_invalid",
                )
            try:
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
            response_text = agent.respond(llm_messages)
            decision = _extract_decision_payload(response_text)
            trace_id = getattr(agent, "last_chat_interaction_id", None)
            approval_request = _serialize_approval_request(
                getattr(agent, "last_approval_request", None),
            )

        await hub.append_message(session_id, "assistant", response_text)
        await hub.set_session_decision(session_id, decision)
        messages = await hub.get_messages(session_id)
        current_decision = await hub.get_session_decision(session_id)
        current_model = await hub.get_session_model(session_id)
        response = _json_response(
            {
                "session_id": session_id,
                "messages": messages,
                "decision": current_decision,
                "selected_model": current_model,
                "trace_id": trace_id,
                "approval_request": approval_request,
            },
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response
    except Exception as exc:  # noqa: BLE001
        error = True
        if status_opened and session_id is not None:
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
                message="Поддерживаются только project команды: find, index.",
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
        await hub.set_session_status(session_id, "busy")
        status_opened = True

        approved_categories = await session_store.get_categories(session_id)
        await hub.append_message(session_id, "user", user_command)
        llm_messages = _ui_messages_to_llm(await hub.get_messages(session_id))

        async with agent_lock:
            try:
                model_config = _build_model_config(
                    selected_model["provider"],
                    selected_model["model"],
                )
                agent.reconfigure_models(model_config, persist=False)
            except Exception as exc:  # noqa: BLE001
                return _error_response(
                    status=400,
                    message=f"Не удалось применить модель сессии: {exc}",
                    error_type="configuration_error",
                    code="model_config_invalid",
                )
            try:
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
            response_text = agent.respond(llm_messages)
            trace_id = getattr(agent, "last_chat_interaction_id", None)
            approval_request = _serialize_approval_request(
                getattr(agent, "last_approval_request", None),
            )

        await hub.append_message(session_id, "assistant", response_text)
        messages = await hub.get_messages(session_id)
        current_decision = await hub.get_session_decision(session_id)
        current_model = await hub.get_session_model(session_id)
        response = _json_response(
            {
                "session_id": session_id,
                "messages": messages,
                "decision": current_decision,
                "selected_model": current_model,
                "trace_id": trace_id,
                "approval_request": approval_request,
            },
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response
    except Exception as exc:  # noqa: BLE001
        error = True
        if status_opened and session_id is not None:
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
    try:
        async with agent_lock:
            response_text = agent.respond(parsed.messages)
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
    records = _parse_trace_log(TRACE_LOG)
    groups = _build_trace_groups(records)
    target: TraceGroup | None = None
    for group in groups:
        if group.interaction_id == trace_id:
            target = group
            break
    if target is None:
        return _error_response(
            status=404,
            message="Trace not found.",
            error_type="invalid_request_error",
            code="trace_not_found",
        )

    tool_calls = _filter_tool_calls(
        path=TOOL_CALLS_LOG,
        start_ts=target.start_ts,
        end_ts=target.end_ts,
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
) -> web.Application:
    config_max_bytes = max_request_bytes or DEFAULT_MAX_REQUEST_BYTES
    app = web.Application(client_max_size=config_max_bytes)
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
    app.router.add_get("/ui", handle_ui_redirect)
    app.router.add_get("/ui/", handle_ui_index)
    app.router.add_get("/ui/api/status", handle_ui_status)
    app.router.add_get("/ui/api/settings", handle_ui_settings)
    app.router.add_post("/ui/api/settings", handle_ui_settings_update)
    app.router.add_get("/ui/api/settings/chats/export", handle_ui_chats_export)
    app.router.add_post("/ui/api/settings/chats/import", handle_ui_chats_import)
    app.router.add_get("/ui/api/models", handle_ui_models)
    app.router.add_get("/ui/api/sessions", handle_ui_sessions_list)
    app.router.add_post("/ui/api/sessions", handle_ui_sessions_create)
    app.router.add_get("/ui/api/sessions/{session_id}", handle_ui_session_get)
    app.router.add_post("/ui/api/session-model", handle_ui_session_model)
    app.router.add_post("/ui/api/chat/send", handle_ui_chat_send)
    app.router.add_post("/ui/api/tools/project", handle_ui_project_command)
    app.router.add_get("/ui/api/events/stream", handle_ui_events_stream)
    app.router.add_static("/ui/assets/", dist_path / "assets")
    return app


def run_server(config: HttpServerConfig) -> None:
    app = create_app(max_request_bytes=config.max_request_bytes)
    web.run_app(app, host=config.host, port=config.port)


def main() -> None:
    config = resolve_http_server_config()
    run_server(config)


__all__ = ["create_app", "main", "run_server"]
