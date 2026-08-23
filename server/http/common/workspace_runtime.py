from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from shared.models import JSONValue, ToolRequest, ToolResult


def _resolve_workspace_file(
    path_raw: str,
    *,
    workspace_root: Path,
    max_download_bytes: int,
) -> Path:
    normalized = path_raw.strip()
    if not normalized:
        raise ValueError("path required")
    candidate = (workspace_root / normalized).resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("path outside workspace") from exc
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError("file not found")
    if candidate.stat().st_size > max_download_bytes:
        raise ValueError("file too large")
    return candidate


def _artifact_file_payload(
    artifact: dict[str, JSONValue],
    *,
    sanitize_download_filename_fn: Callable[[str], str],
    artifact_mime_from_ext_fn: Callable[[str | None], str],
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
    file_name = sanitize_download_filename_fn(file_name_raw)
    inferred_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    mime = artifact_mime_from_ext_fn(ext or inferred_ext)
    return file_name, file_content_raw, mime


def _run_plan_readonly_audit(
    *,
    call_tool: Callable[[ToolRequest], ToolResult],
    plan_audit_timeout_seconds: int,
    plan_audit_max_total_bytes: int,
    plan_audit_max_read_files: int,
) -> tuple[list[dict[str, JSONValue]], dict[str, int]]:
    started = time.monotonic()
    audit_entries: list[dict[str, JSONValue]] = []
    read_files = 0
    total_bytes = 0
    search_calls = 1
    listing = call_tool(
        ToolRequest(
            "workspace_list",
            {"path": "", "recursive": True, "max_depth": 32},
        )
    )
    tree = listing.data.get("tree") if listing.ok else None
    pending = list(tree) if isinstance(tree, list) else []
    while pending:
        if time.monotonic() - started > plan_audit_timeout_seconds:
            break
        node = pending.pop(0)
        if not isinstance(node, dict):
            continue
        children = node.get("children")
        if isinstance(children, list):
            pending[0:0] = children
        if node.get("type") != "file":
            continue
        path = node.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        result = call_tool(ToolRequest("workspace_read", {"path": path}))
        content = result.data.get("output") if result.ok else None
        if not isinstance(content, str) or not content:
            continue
        raw = content.encode("utf-8")
        next_size = min(len(raw), 4000)
        if total_bytes + next_size > plan_audit_max_total_bytes:
            break
        preview = raw[:next_size].decode("utf-8", errors="ignore")
        audit_entries.append(
            {
                "kind": "read_file",
                "path": path,
                "bytes": next_size,
                "preview": preview[:240],
            }
        )
        total_bytes += next_size
        read_files += 1
        if read_files >= plan_audit_max_read_files:
            break
    return audit_entries, {
        "read_files": read_files,
        "total_bytes": total_bytes,
        "search_calls": search_calls,
    }
