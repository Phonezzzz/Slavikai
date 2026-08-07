from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from server.ui_hub import UIHub
from shared.models import JSONValue

MAX_BROWSE_ENTRIES: Final[int] = 200


async def workspace_root_for_session(
    *,
    hub: UIHub,
    session_id: str,
    fallback_root: Path,
) -> Path:
    stored_root = await hub.get_workspace_root(session_id)
    if isinstance(stored_root, str) and stored_root.strip():
        candidate = Path(stored_root).expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate
    return fallback_root


def resolve_workspace_root_candidate(
    path_raw: str,
    *,
    policy_profile: str,
    workspace_root: Path,
) -> Path:
    candidate = Path(path_raw).expanduser().resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"Директория не найдена: {candidate}")
    if policy_profile == "sandbox":
        try:
            candidate.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError("Root должен быть внутри sandbox директории.") from exc
        return candidate
    if policy_profile == "index":
        home_dir = Path.home().resolve()
        try:
            candidate.relative_to(home_dir)
        except ValueError:
            return candidate
        raise ValueError("Root не должен быть внутри домашней директории пользователя.")
    if policy_profile == "yolo":
        return candidate
    raise ValueError(f"Неизвестный policy profile: {policy_profile}")


def browse_directories(
    path_raw: str,
    *,
    policy_profile: str,
    workspace_root: Path,
    max_entries: int = MAX_BROWSE_ENTRIES,
) -> dict[str, JSONValue]:
    base: Path
    if policy_profile == "sandbox":
        base = workspace_root
    elif policy_profile == "index":
        base = Path("/").resolve()
    elif policy_profile == "yolo":
        base = Path("/").resolve()
    else:
        raise ValueError(f"Неизвестный policy profile: {policy_profile}")

    normalized = path_raw.strip()
    candidate = (base / normalized).resolve() if normalized else base

    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("Путь вне разрешённой области просмотра.") from exc

    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"Директория не найдена: {candidate}")

    if policy_profile == "index":
        home_dir = Path.home().resolve()
        is_inside_home = False
        try:
            candidate.relative_to(home_dir)
            is_inside_home = True
        except ValueError:
            pass
        if is_inside_home:
            raise ValueError("Просмотр домашней директории запрещён при index policy.")

    entries: list[dict[str, str]] = []
    try:
        raw_names = sorted(os.listdir(candidate), key=lambda name: name.casefold())
    except OSError as exc:
        raise ValueError(f"Нет доступа к директории: {exc}") from exc

    for name in raw_names:
        if len(entries) >= max_entries:
            break
        if name.startswith("."):
            continue
        full_path = candidate / name
        try:
            resolved = full_path.resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        if full_path.is_symlink():
            try:
                resolved.relative_to(base)
            except ValueError:
                continue
        entries.append({"name": name, "path": str(full_path)})

    parent_path = str(candidate.parent) if candidate != base else None

    breadcrumbs: list[dict[str, str]] = []
    current = candidate
    while True:
        label = current.name or str(current) if current != Path("/") else "/"
        breadcrumbs.insert(0, {"name": label, "path": str(current)})
        if current == base:
            break
        current = current.parent

    return {
        "path": str(candidate),
        "parent": parent_path,
        "entries": entries,
        "truncated": len(entries) >= max_entries,
        "breadcrumbs": breadcrumbs,
    }
