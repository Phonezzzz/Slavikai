from __future__ import annotations

from aiohttp import web

from core.desktop_policy import (
    DesktopApprovalRule,
    DesktopApprovalScope,
    DesktopPolicyStore,
    PolicyEffect,
)
from server.http.common.auth import _request_principal_id, _require_owner
from server.http.common.responses import error_response, json_response


async def handle_desktop_approval_rules_list(request: web.Request) -> web.Response:
    owner_error = _require_owner(request)
    if owner_error is not None:
        return owner_error
    principal_id = _request_principal_id(request)
    if principal_id is None:
        raise RuntimeError("Owner request has no principal")
    store: DesktopPolicyStore = request.app["desktop_policy_store"]
    try:
        rules = store.list_rules(subject_principal_id=principal_id)
    except (OSError, ValueError) as exc:
        return error_response(
            status=409,
            message=f"Не удалось прочитать Desktop approval rules: {exc}",
            error_type="invalid_request_error",
            code="desktop_approval_store_invalid",
            details={
                "path": str(store.path),
                "load_errors": store.list_load_errors(),
                "recovery_endpoint": "/ui/api/desktop/approvals/reset-invalid",
            },
        )
    return json_response(
        {
            "ok": True,
            "rules": [rule.to_dict() for rule in rules],
            "load_warnings": store.list_load_errors(),
        }
    )


async def handle_desktop_approval_rule_create(request: web.Request) -> web.Response:
    owner_error = _require_owner(request)
    if owner_error is not None:
        return owner_error
    principal_id = _request_principal_id(request)
    if principal_id is None:
        raise RuntimeError("Owner request has no principal")
    payload, error = await _json_object(request)
    if error is not None:
        return error
    if payload is None:
        raise RuntimeError("JSON helper returned no payload and no error")
    try:
        effect = _effect(payload.get("effect"))
        scope = DesktopApprovalScope.from_dict(payload.get("scope"))
        description_raw = payload.get("description")
        rule = DesktopApprovalRule.create(
            effect=effect,
            scope=scope,
            source="persistent",
            description=description_raw if isinstance(description_raw, str) else "",
            subject_principal_id=principal_id,
        )
        store: DesktopPolicyStore = request.app["desktop_policy_store"]
        store.add_rule(rule)
    except (OSError, ValueError) as exc:
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_desktop_approval_rule",
        )
    return json_response({"ok": True, "rule": rule.to_dict()}, status=201)


async def handle_desktop_approval_rule_update(request: web.Request) -> web.Response:
    owner_error = _require_owner(request)
    if owner_error is not None:
        return owner_error
    payload, error = await _json_object(request)
    if error is not None:
        return error
    if payload is None:
        raise RuntimeError("JSON helper returned no payload and no error")
    rule_id = request.match_info.get("rule_id", "").strip()
    try:
        effect = _effect(payload.get("effect")) if "effect" in payload else None
        scope = DesktopApprovalScope.from_dict(payload.get("scope")) if "scope" in payload else None
        description_raw = payload.get("description")
        description = description_raw if isinstance(description_raw, str) else None
        store: DesktopPolicyStore = request.app["desktop_policy_store"]
        updated = store.update_rule(
            rule_id,
            effect=effect,
            scope=scope,
            description=description,
        )
    except (OSError, ValueError) as exc:
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_desktop_approval_rule",
        )
    if updated is None:
        return error_response(
            status=404,
            message="Desktop approval rule не найден.",
            error_type="invalid_request_error",
            code="desktop_approval_rule_not_found",
        )
    return json_response({"ok": True, "rule": updated.to_dict()})


async def handle_desktop_approval_rule_delete(request: web.Request) -> web.Response:
    owner_error = _require_owner(request)
    if owner_error is not None:
        return owner_error
    rule_id = request.match_info.get("rule_id", "").strip()
    store: DesktopPolicyStore = request.app["desktop_policy_store"]
    try:
        removed = store.remove_rule(rule_id)
    except (OSError, ValueError) as exc:
        return error_response(
            status=500,
            message=f"Не удалось удалить Desktop approval rule: {exc}",
            error_type="internal_error",
            code="desktop_approval_store_error",
        )
    if not removed:
        return error_response(
            status=404,
            message="Desktop approval rule не найден.",
            error_type="invalid_request_error",
            code="desktop_approval_rule_not_found",
        )
    return json_response({"ok": True, "removed_rule_id": rule_id})


async def handle_desktop_approval_rules_reset_invalid(request: web.Request) -> web.Response:
    owner_error = _require_owner(request)
    if owner_error is not None:
        return owner_error
    payload, error = await _json_object(request)
    if error is not None:
        return error
    if payload is None:
        raise RuntimeError("JSON helper returned no payload and no error")
    if payload.get("confirm") is not True:
        return error_response(
            status=400,
            message="Для reset повреждённого Desktop approval store требуется confirm=true.",
            error_type="invalid_request_error",
            code="desktop_approval_reset_confirmation_required",
        )
    store: DesktopPolicyStore = request.app["desktop_policy_store"]
    try:
        discarded_errors = store.reset_invalid_store()
    except (OSError, ValueError) as exc:
        return error_response(
            status=409,
            message=str(exc),
            error_type="invalid_request_error",
            code="desktop_approval_store_not_invalid",
        )
    return json_response(
        {
            "ok": True,
            "rules": [],
            "discarded_load_errors": discarded_errors,
        }
    )


async def _json_object(
    request: web.Request,
) -> tuple[dict[str, object] | None, web.Response | None]:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return None, error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return None, error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    return {str(key): value for key, value in payload.items()}, None


def _effect(value: object) -> PolicyEffect:
    if value == "allow":
        return "allow"
    if value == "ask":
        return "ask"
    if value == "deny":
        return "deny"
    raise ValueError("effect должен быть allow|ask|deny")
