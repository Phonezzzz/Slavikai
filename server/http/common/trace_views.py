from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.tracer import TRACE_LOG as _TRACE_LOG
from core.tracer import TraceRecord
from server.http.common.workflow_state import normalize_json_value
from shared.models import JSONValue
from shared.sanitize import safe_json_loads
from tools.tool_logger import DEFAULT_LOG_PATH as _TOOL_CALLS_LOG

TRACE_LOG: Path = _TRACE_LOG
TOOL_CALLS_LOG: Path = _TOOL_CALLS_LOG


@dataclass(frozen=True)
class TraceGroup:
    start_ts: str
    end_ts: str | None
    interaction_id: str | None
    events: list[TraceRecord]


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


def _tool_calls_for_trace_id(
    trace_id: str,
    *,
    trace_log: Path = TRACE_LOG,
    tool_calls_log: Path = TOOL_CALLS_LOG,
) -> list[dict[str, JSONValue]] | None:
    records = _parse_trace_log(trace_log)
    groups = _build_trace_groups(records)
    target: TraceGroup | None = None
    for group in groups:
        if group.interaction_id == trace_id:
            target = group
            break
    if target is None:
        return None
    return _filter_tool_calls(
        path=tool_calls_log,
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
                args[str(key)] = normalize_json_value(value)
        for path in _extract_paths_from_tool_call(tool=tool_raw, args=args):
            if path in seen:
                continue
            seen.add(path)
            files.append(path)
    return files
