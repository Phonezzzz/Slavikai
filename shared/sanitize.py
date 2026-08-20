from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from shared.models import JSONValue

SECRET_KEYS = {
    "api_key",
    "authorization",
    "x-api-key",
    "token",
    "secret",
    "voice_id",
}
PAYLOAD_KEYS = {"content", "patch", "base64", "audio", "bytes"}
SENSITIVE_PAYLOAD_KEYS = {"text", "value", "body"}
MAX_FIELD_PREVIEW = 256
MAX_RECORD_BYTES = 4096
SENSITIVE_COMMAND_FLAGS = {
    "--api-key",
    "--api_key",
    "--authorization",
    "--passwd",
    "--password",
    "--secret",
    "--token",
}
_COMMAND_SECRET_RE = re.compile(
    r"(?i)(--(?:api[-_]?key|authorization|passwd|password|secret|token)(?:=|\s+))([^\s]+)"
)
_COMMAND_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([a-z0-9_]*(?:api[-_]?key|authorization|passwd|password|secret|token))=([^\s]+)"
)
_URL_CREDENTIAL_RE = re.compile(r"(?i)(https?://[^\s:/@]+:)([^\s@]+)(@)")
_URL_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[-_]?key|authorization|password|secret|token)=)([^&\s#]+)"
)


def _mask_value(value: Any) -> str:
    return "[secret]"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace")
    try:
        return json.dumps(value, ensure_ascii=False).encode("utf-8", errors="replace")
    except (TypeError, ValueError):
        return str(value).encode("utf-8", errors="replace")


def _truncate_payload(value: Any) -> dict[str, JSONValue]:
    raw_bytes = _to_bytes(value)
    preview_bytes = raw_bytes[:MAX_FIELD_PREVIEW]
    preview = preview_bytes.decode("utf-8", errors="replace")
    if len(raw_bytes) > MAX_FIELD_PREVIEW:
        preview += "…[truncated]"
    return {
        "preview": preview,
        "bytes_count": len(raw_bytes),
        "sha256": _sha256_bytes(raw_bytes),
    }


def _fingerprint_payload(value: Any) -> dict[str, JSONValue]:
    raw_bytes = _to_bytes(value)
    return {
        "redacted": True,
        "bytes_count": len(raw_bytes),
        "sha256": _sha256_bytes(raw_bytes),
    }


def _sanitize_value(key: str | None, value: Any) -> JSONValue:
    key_lower = key.lower() if isinstance(key, str) else ""
    if key_lower in SECRET_KEYS:
        return _mask_value(value)

    if isinstance(value, (bool, int, float)) or value is None:
        return value

    if isinstance(value, dict):
        return {k: _sanitize_value(k, v) for k, v in value.items()}
    if key_lower == "argv" and isinstance(value, list):
        return _redact_argv(value)
    if isinstance(value, list):
        return [_sanitize_value(key, v) for v in value]

    if isinstance(value, (str, bytes)):
        if key_lower in {"command", "url"}:
            return _redact_command_text(
                value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
            )
        raw_bytes = _to_bytes(value)
        if key_lower in SENSITIVE_PAYLOAD_KEYS:
            return _fingerprint_payload(value)
        if key_lower in PAYLOAD_KEYS or len(raw_bytes) > MAX_FIELD_PREVIEW:
            return _truncate_payload(value)
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value

    return str(value)


def _redact_argv(value: list[Any]) -> list[JSONValue]:
    redacted: list[JSONValue] = []
    mask_next = False
    for item in value:
        if not isinstance(item, str):
            redacted.append(_sanitize_value(None, item))
            mask_next = False
            continue
        if mask_next:
            redacted.append("[secret]")
            mask_next = False
            continue
        lowered = item.lower()
        if lowered in SENSITIVE_COMMAND_FLAGS:
            redacted.append(item)
            mask_next = True
            continue
        flag, separator, _secret = item.partition("=")
        if separator and flag.lower() in SENSITIVE_COMMAND_FLAGS:
            redacted.append(f"{flag}=[secret]")
            continue
        if "authorization:" in lowered:
            redacted.append("Authorization: [secret]")
            continue
        redacted.append(_redact_command_text(item))
    return redacted


def _redact_command_text(value: str) -> str:
    redacted = _COMMAND_SECRET_RE.sub(r"\1[secret]", value)
    redacted = _COMMAND_ASSIGNMENT_RE.sub(r"\1=[secret]", redacted)
    redacted = _URL_CREDENTIAL_RE.sub(r"\1[secret]\3", redacted)
    return _URL_QUERY_SECRET_RE.sub(r"\1[secret]", redacted)


def sanitize_record(
    record: Mapping[str, JSONValue],
    *,
    max_bytes: int = MAX_RECORD_BYTES,
) -> dict[str, JSONValue]:
    sanitized = {k: _sanitize_value(k, v) for k, v in record.items()}
    encoded = json.dumps(sanitized, ensure_ascii=False).encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return sanitized

    # Drop heavy fields and replace with summary
    for heavy_key in ("args", "meta"):
        if heavy_key in sanitized:
            sanitized[heavy_key] = _truncate_payload(sanitized[heavy_key])
    encoded = json.dumps(sanitized, ensure_ascii=False).encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return sanitized

    return _truncate_payload(sanitized)


def safe_json_loads(raw: str) -> object | None:
    try:
        parsed: object = json.loads(raw)
        return parsed
    except json.JSONDecodeError:
        return None
