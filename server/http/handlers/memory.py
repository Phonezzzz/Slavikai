from __future__ import annotations

import asyncio
from typing import Protocol, cast

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from memory.categorized_memory_store import ListPage
from memory.triage import TriagePlan, TriageSuggestion, triage_apply, triage_preview, triage_undo
from server.http.common.responses import error_response, json_response
from server.http_api import (
    _model_not_allowed_response,
    _model_not_selected_response,
    _normalize_json_value,
    _resolve_agent,
)
from shared.memory_category_models import MemoryCategory, MemoryItem
from shared.models import JSONValue


def _normalize_conflict_action(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower()
    if normalized in {"activate", "deprecate", "set_value"}:
        return normalized
    return None


class MemoryStoreProtocol(Protocol):
    def list_items(
        self,
        category: MemoryCategory,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> ListPage: ...

    def get_item(self, item_id: str) -> MemoryItem | None: ...

    def update_item(self, item_id: str, **kwargs: object) -> MemoryItem | None: ...


def _resolve_memory_store(agent: object) -> MemoryStoreProtocol | None:
    store = getattr(agent, "_memory_inbox_store", None)
    if store is None:
        return None
    list_items = getattr(store, "list_items", None)
    get_item = getattr(store, "get_item", None)
    update_item = getattr(store, "update_item", None)
    if not callable(list_items) or not callable(get_item) or not callable(update_item):
        return None
    return cast(MemoryStoreProtocol, store)


def _parse_triage_suggestions(raw: object) -> list[TriageSuggestion]:
    if not isinstance(raw, list):
        raise ValueError("plan.suggestions должен быть списком.")
    suggestions: list[TriageSuggestion] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        item_id_raw = item.get("item_id")
        category_raw = item.get("proposed_category")
        confidence_raw = item.get("confidence")
        reason_raw = item.get("reason")
        dangerous_raw = item.get("dangerous")
        if not isinstance(item_id_raw, str) or not item_id_raw.strip():
            continue
        if not isinstance(category_raw, str):
            continue
        if category_raw not in {category.value for category in MemoryCategory}:
            continue
        if confidence_raw not in {"low", "medium", "high"}:
            continue
        if not isinstance(reason_raw, str) or not reason_raw.strip():
            continue
        dangerous = bool(dangerous_raw)
        suggestions.append(
            TriageSuggestion(
                item_id=item_id_raw.strip(),
                proposed_category=MemoryCategory(category_raw),
                confidence=confidence_raw,
                reason=reason_raw.strip(),
                dangerous=dangerous,
            )
        )
    return suggestions


def _parse_triage_plan(raw: object) -> TriagePlan:
    if not isinstance(raw, dict):
        raise ValueError("plan должен быть объектом.")
    plan_id_raw = raw.get("plan_id")
    if not isinstance(plan_id_raw, str) or not plan_id_raw.strip():
        raise ValueError("plan.plan_id обязателен.")
    suggestions = _parse_triage_suggestions(raw.get("suggestions"))
    counts_raw = raw.get("counts")
    counts: dict[str, int] = {}
    if isinstance(counts_raw, dict):
        for key, value in counts_raw.items():
            if isinstance(key, str) and isinstance(value, int):
                counts[key] = value
    return TriagePlan(
        plan_id=plan_id_raw.strip(),
        created_at=triage_preview([], plan_id=plan_id_raw.strip()).created_at,
        source_category=MemoryCategory.INBOX,
        suggestions=suggestions,
        counts=counts,
    )


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


async def handle_ui_memory_triage_preview(request: web.Request) -> web.Response:
    try:
        agent = await _resolve_agent(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    store = _resolve_memory_store(agent)
    if store is None:
        return error_response(
            status=501,
            message="Memory triage store недоступен.",
            error_type="not_supported",
            code="memory_triage_not_supported",
        )
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    limit_raw = payload.get("limit", 50)
    limit = 50
    if isinstance(limit_raw, int) and limit_raw > 0:
        limit = min(limit_raw, 500)
    plan_id_raw = payload.get("plan_id")
    plan_id = plan_id_raw.strip() if isinstance(plan_id_raw, str) and plan_id_raw.strip() else None
    inbox_page = store.list_items(MemoryCategory.INBOX, limit=limit)
    plan = triage_preview(inbox_page.items, plan_id=plan_id)
    return json_response(
        {
            "ok": True,
            "plan": plan.to_dict(),
            "next_cursor": inbox_page.next_cursor,
            "inbox_count": len(inbox_page.items),
        }
    )


async def handle_ui_memory_triage_apply(request: web.Request) -> web.Response:
    try:
        agent = await _resolve_agent(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    store = _resolve_memory_store(agent)
    if store is None:
        return error_response(
            status=501,
            message="Memory triage store недоступен.",
            error_type="not_supported",
            code="memory_triage_not_supported",
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
    plan_payload = payload.get("plan")
    if plan_payload is None:
        return error_response(
            status=400,
            message="plan обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        plan = _parse_triage_plan(plan_payload)
    except ValueError as exc:
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    allow_dangerous = payload.get("allow_dangerous") is True
    tx_id_raw = payload.get("tx_id")
    tx_id = tx_id_raw.strip() if isinstance(tx_id_raw, str) and tx_id_raw.strip() else None
    result = triage_apply(
        store=store,
        plan=plan,
        allow_dangerous=allow_dangerous,
        tx_id=tx_id,
    )
    return json_response({"ok": True, "result": result.to_dict()})


async def handle_ui_memory_triage_undo(request: web.Request) -> web.Response:
    try:
        agent = await _resolve_agent(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    store = _resolve_memory_store(agent)
    if store is None:
        return error_response(
            status=501,
            message="Memory triage store недоступен.",
            error_type="not_supported",
            code="memory_triage_not_supported",
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
    tx_id_raw = payload.get("tx_id")
    if not isinstance(tx_id_raw, str) or not tx_id_raw.strip():
        return error_response(
            status=400,
            message="tx_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        result = triage_undo(store=store, tx_id=tx_id_raw.strip())
    except ValueError as exc:
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    return json_response({"ok": True, "result": result.to_dict()})


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
