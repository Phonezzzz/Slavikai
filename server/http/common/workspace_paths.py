from __future__ import annotations

from pathlib import Path

from server.ui_hub import UIHub


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
