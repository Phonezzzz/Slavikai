from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from shared.models import LLMMessage

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


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[LLMMessage]
    stream: bool
    session_id: str | None
    sampling_warnings: list[str]
    tool_calling_present: bool


def _is_sampling_key(key: str) -> bool:
    if key in EXTRA_SAMPLING_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in SAMPLING_PREFIXES)


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
