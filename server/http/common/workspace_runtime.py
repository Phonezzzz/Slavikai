from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from shared.models import JSONValue


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
    root: Path,
    plan_audit_timeout_seconds: int,
    workspace_index_ignored_dirs: set[str],
    workspace_index_allowed_extensions: set[str],
    plan_audit_max_total_bytes: int,
    plan_audit_max_read_files: int,
) -> tuple[list[dict[str, JSONValue]], dict[str, int]]:
    started = time.monotonic()
    audit_entries: list[dict[str, JSONValue]] = []
    read_files = 0
    total_bytes = 0
    search_calls = 0
    for current_root, dirs, files in os.walk(root):
        if time.monotonic() - started > plan_audit_timeout_seconds:
            break
        dirs[:] = [name for name in dirs if name not in workspace_index_ignored_dirs]
        current = Path(current_root)
        for filename in files:
            if read_files >= plan_audit_max_read_files:
                break
            if time.monotonic() - started > plan_audit_timeout_seconds:
                break
            if filename.endswith(".sqlite"):
                continue
            full_path = current / filename
            suffix = full_path.suffix.lower()
            if suffix and suffix not in workspace_index_allowed_extensions:
                continue
            try:
                raw = full_path.read_bytes()
            except Exception:  # noqa: BLE001
                continue
            if not raw:
                continue
            next_size = min(len(raw), 4000)
            if total_bytes + next_size > plan_audit_max_total_bytes:
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
        if read_files >= plan_audit_max_read_files or total_bytes >= plan_audit_max_total_bytes:
            break
    return audit_entries, {
        "read_files": read_files,
        "total_bytes": total_bytes,
        "search_calls": search_calls,
    }
