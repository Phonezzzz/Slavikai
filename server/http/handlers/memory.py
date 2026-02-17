from __future__ import annotations

import asyncio

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from server.http.common.responses import error_response, json_response
from server.http_api import (
    _model_not_allowed_response,
    _model_not_selected_response,
    _normalize_json_value,
    _resolve_agent,
)
from shared.models import JSONValue


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
        return json_response({"conflicts": []})

    limit_raw = request.query.get("limit", "").strip()
    limit = 50
    if limit_raw:
        try:
            limit = int(limit_raw)
        except ValueError:
            return error_response(
                status=400,
                message="limit должен быть int.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
    if limit <= 0 or limit > 200:
        return error_response(
            status=400,
            message="limit должен быть в диапазоне 1..200.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    list_conflicts = getattr(agent, "list_memory_conflicts", None)
    if not callable(list_conflicts):
        return json_response({"conflicts": []})
    try:
        conflicts_raw = list_conflicts(limit)
    except Exception as exc:  # noqa: BLE001
        return error_response(
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
    return json_response({"conflicts": conflicts})


async def handle_ui_memory_conflicts_resolve(request: web.Request) -> web.Response:
    try:
        agent = await _resolve_agent(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()

    resolve_conflict = getattr(agent, "resolve_memory_conflict", None)
    if not callable(resolve_conflict):
        return error_response(
            status=501,
            message="Memory conflict resolver недоступен.",
            error_type="not_supported",
            code="memory_conflict_resolve_not_supported",
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
    stable_key_raw = payload.get("stable_key")
    action = _normalize_conflict_action(payload.get("action"))
    if not isinstance(stable_key_raw, str) or not stable_key_raw.strip():
        return error_response(
            status=400,
            message="stable_key обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if action is None:
        return error_response(
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
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=500,
            message=f"Не удалось разрешить конфликт памяти: {exc}",
            error_type="internal_error",
            code="memory_conflict_resolve_failed",
        )
    if resolved_raw is None:
        return error_response(
            status=404,
            message="Конфликт не найден.",
            error_type="invalid_request_error",
            code="memory_conflict_not_found",
        )
    normalized: dict[str, JSONValue] = {}
    if isinstance(resolved_raw, dict):
        for key, value in resolved_raw.items():
            normalized[str(key)] = _normalize_json_value(value)
    return json_response({"resolved": normalized})
