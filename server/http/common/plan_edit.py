from __future__ import annotations

import json
from collections.abc import Callable

from shared.models import JSONValue


def plan_mark_step(
    plan: dict[str, JSONValue],
    *,
    step_id: str,
    status: str,
    evidence: dict[str, JSONValue] | None,
    plan_step_statuses: set[str],
    utc_now_iso: Callable[[], str],
    increment_plan_revision_fn: Callable[[dict[str, JSONValue]], int],
    plan_hash_payload_fn: Callable[[dict[str, JSONValue]], str],
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
                candidate["status"] = status if status in plan_step_statuses else "blocked"
                candidate["evidence"] = evidence
            next_steps.append(candidate)
    updated["steps"] = next_steps
    updated["updated_at"] = utc_now_iso()
    increment_plan_revision_fn(updated)
    updated["plan_hash"] = plan_hash_payload_fn(updated)
    return updated


def validate_text_limit(
    value: str,
    *,
    field: str,
    max_chars: int,
) -> None:
    if len(value) > max_chars:
        raise ValueError(f"{field} превышает лимит {max_chars} символов.")


def find_forbidden_plan_key(
    value: JSONValue,
    *,
    forbidden_keys: set[str] | None = None,
) -> str | None:
    forbidden = forbidden_keys or {
        "args",
        "command",
        "exec",
        "tool_name",
        "source_endpoint",
        "resume_payload",
        "edited_action",
    }
    stack: list[tuple[JSONValue, str]] = [(value, "$")]
    while stack:
        current, path = stack.pop()
        if isinstance(current, dict):
            for key, nested in current.items():
                lowered = key.strip().lower()
                next_path = f"{path}.{key}"
                if lowered in forbidden:
                    return next_path
                stack.append((nested, next_path))
        elif isinstance(current, list):
            for index, nested in enumerate(current):
                stack.append((nested, f"{path}[{index}]"))
    return None


def normalize_plan_step_insert(
    raw: object,
    *,
    plan_step_statuses: set[str],
    normalize_string_list_fn: Callable[[object], list[str]],
    normalize_json_value_fn: Callable[[object], JSONValue],
    validate_text_limit_fn: Callable[[str, str], None],
) -> dict[str, JSONValue]:
    if not isinstance(raw, dict):
        raise ValueError("step должен быть объектом.")
    allowed_fields = {
        "step_id",
        "title",
        "description",
        "intent",
        "allowed_tool_kinds",
        "acceptance_checks",
        "status",
        "evidence",
    }
    unexpected = sorted({str(key) for key in raw if str(key) not in allowed_fields})
    if unexpected:
        raise ValueError(f"step содержит неожиданные поля: {', '.join(unexpected)}.")
    step_id_raw = raw.get("step_id")
    title_raw = raw.get("title")
    description_raw = raw.get("description")
    intent_raw = raw.get("intent")
    if not isinstance(step_id_raw, str) or not step_id_raw.strip():
        raise ValueError("step.step_id обязателен.")
    if not isinstance(title_raw, str) or not title_raw.strip():
        raise ValueError("step.title обязателен.")
    if isinstance(description_raw, str):
        description = description_raw
    elif isinstance(intent_raw, str):
        description = intent_raw
    else:
        description = ""
    status_raw = raw.get("status")
    status = (
        status_raw if isinstance(status_raw, str) and status_raw in plan_step_statuses else "todo"
    )
    validate_text_limit_fn(step_id_raw.strip(), "step.step_id")
    validate_text_limit_fn(title_raw.strip(), "step.title")
    validate_text_limit_fn(description, "step.description")
    return {
        "step_id": step_id_raw.strip(),
        "title": title_raw.strip(),
        "description": description,
        "allowed_tool_kinds": normalize_string_list_fn(raw.get("allowed_tool_kinds")),
        "acceptance_checks": normalize_string_list_fn(raw.get("acceptance_checks")),
        "status": status,
        "evidence": (
            normalize_json_value_fn(raw.get("evidence"))
            if isinstance(raw.get("evidence"), dict)
            else None
        ),
    }


def normalize_plan_step_changes(
    raw: object,
    *,
    plan_step_statuses: set[str],
    normalize_string_list_fn: Callable[[object], list[str]],
    normalize_json_value_fn: Callable[[object], JSONValue],
    validate_text_limit_fn: Callable[[str, str], None],
) -> dict[str, JSONValue]:
    if not isinstance(raw, dict):
        raise ValueError("changes должен быть объектом.")
    allowed_fields = {
        "title",
        "description",
        "intent",
        "allowed_tool_kinds",
        "acceptance_checks",
        "status",
        "evidence",
    }
    unexpected = sorted({str(key) for key in raw if str(key) not in allowed_fields})
    if unexpected:
        raise ValueError(f"changes содержит неожиданные поля: {', '.join(unexpected)}.")
    normalized: dict[str, JSONValue] = {}
    if "title" in raw:
        title_raw = raw.get("title")
        if not isinstance(title_raw, str) or not title_raw.strip():
            raise ValueError("changes.title должен быть непустой строкой.")
        validate_text_limit_fn(title_raw.strip(), "changes.title")
        normalized["title"] = title_raw.strip()
    if "description" in raw or "intent" in raw:
        source = raw.get("description") if "description" in raw else raw.get("intent")
        if not isinstance(source, str):
            raise ValueError("changes.description/intent должен быть строкой.")
        validate_text_limit_fn(source, "changes.description")
        normalized["description"] = source
    if "allowed_tool_kinds" in raw:
        normalized["allowed_tool_kinds"] = normalize_string_list_fn(raw.get("allowed_tool_kinds"))
    if "acceptance_checks" in raw:
        normalized["acceptance_checks"] = normalize_string_list_fn(raw.get("acceptance_checks"))
    if "status" in raw:
        status_raw = raw.get("status")
        if not isinstance(status_raw, str) or status_raw not in plan_step_statuses:
            raise ValueError(
                "changes.status должен быть todo|doing|waiting_approval|blocked|done|failed."
            )
        normalized["status"] = status_raw
    if "evidence" in raw:
        evidence_raw = raw.get("evidence")
        if evidence_raw is None:
            normalized["evidence"] = None
        elif isinstance(evidence_raw, dict):
            normalized["evidence"] = normalize_json_value_fn(evidence_raw)
        else:
            raise ValueError("changes.evidence должен быть объектом или null.")
    return normalized


def validate_plan_document(
    plan: dict[str, JSONValue],
    *,
    plan_max_payload_bytes: int,
    plan_max_steps: int,
    validate_text_limit_fn: Callable[[str, str], None],
    find_forbidden_plan_key_fn: Callable[[JSONValue], str | None],
    normalize_plan_step_insert_fn: Callable[[object], dict[str, JSONValue]],
) -> None:
    encoded = json.dumps(plan, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > plan_max_payload_bytes:
        raise ValueError(f"plan payload превышает лимит {plan_max_payload_bytes} bytes.")
    allowed_top = {
        "plan_id",
        "plan_hash",
        "plan_revision",
        "status",
        "goal",
        "scope_in",
        "scope_out",
        "assumptions",
        "inputs_needed",
        "audit_log",
        "steps",
        "exit_criteria",
        "created_at",
        "updated_at",
        "approved_at",
        "approved_by",
    }
    unexpected_top = sorted({str(key) for key in plan if str(key) not in allowed_top})
    if unexpected_top:
        raise ValueError(f"plan содержит неожиданные поля: {', '.join(unexpected_top)}.")
    forbidden_path = find_forbidden_plan_key_fn(plan)
    if forbidden_path is not None:
        raise ValueError(f"plan содержит запрещённый ключ: {forbidden_path}.")
    goal_raw = plan.get("goal")
    if isinstance(goal_raw, str):
        validate_text_limit_fn(goal_raw, "plan.goal")
    steps_raw = plan.get("steps")
    if not isinstance(steps_raw, list):
        raise ValueError("plan.steps должен быть списком.")
    if len(steps_raw) > plan_max_steps:
        raise ValueError(f"steps превышает лимит {plan_max_steps}.")
    seen_ids: set[str] = set()
    for item in steps_raw:
        if not isinstance(item, dict):
            raise ValueError("steps содержит не-объект.")
        step = normalize_plan_step_insert_fn(item)
        step_id = step["step_id"]
        if isinstance(step_id, str):
            if step_id in seen_ids:
                raise ValueError(f"step_id должен быть уникальным: {step_id}")
            seen_ids.add(step_id)


def plan_apply_edit_operation(
    *,
    plan: dict[str, JSONValue],
    operation: dict[str, JSONValue],
    normalize_plan_step_insert_fn: Callable[[object], dict[str, JSONValue]],
    normalize_plan_step_changes_fn: Callable[[object], dict[str, JSONValue]],
    utc_now_iso: Callable[[], str],
    increment_plan_revision_fn: Callable[[dict[str, JSONValue]], int],
    validate_plan_document_fn: Callable[[dict[str, JSONValue]], None],
    plan_hash_payload_fn: Callable[[dict[str, JSONValue]], str],
) -> dict[str, JSONValue]:
    op_raw = operation.get("op")
    if not isinstance(op_raw, str) or not op_raw.strip():
        raise ValueError("operation.op обязателен.")
    op = op_raw.strip()
    steps_raw = plan.get("steps")
    if not isinstance(steps_raw, list):
        raise ValueError("plan.steps должен быть списком.")
    steps: list[dict[str, JSONValue]] = [dict(item) for item in steps_raw if isinstance(item, dict)]

    if op == "insert_step":
        step = normalize_plan_step_insert_fn(operation.get("step"))
        index_raw = operation.get("index")
        index = len(steps)
        if isinstance(index_raw, int):
            index = max(0, min(index_raw, len(steps)))
        steps.insert(index, step)
    elif op == "delete_step":
        step_id_raw = operation.get("step_id")
        if not isinstance(step_id_raw, str) or not step_id_raw.strip():
            raise ValueError("operation.step_id обязателен.")
        filtered_steps = [item for item in steps if item.get("step_id") != step_id_raw.strip()]
        if len(filtered_steps) == len(steps):
            raise ValueError("step_id не найден.")
        steps = filtered_steps
    elif op == "reorder_steps":
        order_raw = operation.get("step_ids")
        if not isinstance(order_raw, list) or not order_raw:
            raise ValueError("operation.step_ids должен быть непустым списком.")
        order = [item.strip() for item in order_raw if isinstance(item, str) and item.strip()]
        if len(order) != len(steps):
            raise ValueError("operation.step_ids должен содержать все шаги.")
        by_id = {
            item.get("step_id"): item for item in steps if isinstance(item.get("step_id"), str)
        }
        if len(by_id) != len(steps):
            raise ValueError("Некорректные шаги в плане.")
        reordered: list[dict[str, JSONValue]] = []
        for step_id in order:
            candidate = by_id.get(step_id)
            if candidate is None:
                raise ValueError("operation.step_ids содержит неизвестный step_id.")
            reordered.append(dict(candidate))
        steps = reordered
    elif op == "update_step":
        step_id_raw = operation.get("step_id")
        if not isinstance(step_id_raw, str) or not step_id_raw.strip():
            raise ValueError("operation.step_id обязателен.")
        changes = normalize_plan_step_changes_fn(operation.get("changes"))
        updated = False
        next_steps: list[dict[str, JSONValue]] = []
        for item in steps:
            candidate = dict(item)
            if candidate.get("step_id") == step_id_raw.strip():
                candidate.update(changes)
                updated = True
            next_steps.append(candidate)
        if not updated:
            raise ValueError("step_id не найден.")
        steps = next_steps
    else:
        raise ValueError(
            "operation.op должен быть insert_step|delete_step|reorder_steps|update_step."
        )

    updated_plan = dict(plan)
    updated_plan["steps"] = steps
    updated_plan["status"] = "draft"
    updated_plan["approved_at"] = None
    updated_plan["approved_by"] = None
    updated_plan["updated_at"] = utc_now_iso()
    increment_plan_revision_fn(updated_plan)
    validate_plan_document_fn(updated_plan)
    updated_plan["plan_hash"] = plan_hash_payload_fn(updated_plan)
    return updated_plan


def find_next_todo_step(plan: dict[str, JSONValue]) -> dict[str, JSONValue] | None:
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
