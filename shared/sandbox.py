from __future__ import annotations

import re
from pathlib import Path
from typing import Final

_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:")
_UNC_RE = re.compile(r"^\\\\")

SANDBOX_ROOT: Final[Path] = Path("sandbox").resolve()
WORKSPACE_ROOT: Final[Path] = (SANDBOX_ROOT / "project").resolve()


class SandboxViolationError(ValueError):
    def __init__(self, raw_path: str, normalized_path: Path) -> None:
        self.raw_path = raw_path
        self.normalized_path = normalized_path
        super().__init__(self.__str__())

    def __str__(self) -> str:  # noqa: D401 - сообщение ошибки
        return f"Sandbox violation: raw='{self.raw_path}', normalized='{self.normalized_path}'"


def _is_disallowed_absolute(raw_path: str) -> bool:
    raw = raw_path.strip()
    if not raw:
        return False
    if raw.startswith(("~", "/", "\\")):
        return True
    if raw.startswith("//"):
        return True
    if _WINDOWS_DRIVE_RE.match(raw):
        return True
    if _UNC_RE.match(raw):
        return True
    return False


def _contains_parent_reference(raw_path: str) -> bool:
    normalized = raw_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    return any(part == ".." for part in parts)


def normalize_sandbox_path(raw_path: str, sandbox_root: Path) -> Path:
    """
    Нормализует путь относительно sandbox_root.
    Запрещает абсолютные/tilde/UNC/drive пути и выход за пределы песочницы.
    """
    raw = raw_path.strip() or "."
    root = sandbox_root.resolve()
    if _is_disallowed_absolute(raw):
        normalized = Path(raw).expanduser().resolve()
        raise SandboxViolationError(raw, normalized)
    if _contains_parent_reference(raw):
        candidate = (root / raw).resolve()
        raise SandboxViolationError(raw, candidate)

    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SandboxViolationError(raw, candidate) from exc
    return candidate


def normalize_shell_sandbox_root(
    raw_sandbox_root: str, *, sandbox_root: Path | None = None
) -> Path:
    """
    Нормализует sandbox_root для ShellTool относительно SANDBOX_ROOT.

    Принимает:
    - "" / "." / "sandbox" -> SANDBOX_ROOT
    - "sandbox/<subdir>" -> SANDBOX_ROOT/<subdir> (back-compat)
    - "<subdir>" -> SANDBOX_ROOT/<subdir>

    Запрещает абсолютные пути и любые '..' сегменты.
    """
    root = (sandbox_root or SANDBOX_ROOT).resolve()
    raw = raw_sandbox_root.strip()
    if not raw or raw == ".":
        return root
    if _is_disallowed_absolute(raw):
        normalized = Path(raw).expanduser().resolve()
        raise SandboxViolationError(raw, normalized)

    normalized_raw = raw.replace("\\", "/").strip()
    if normalized_raw == "sandbox":
        relative = "."
    elif normalized_raw.startswith("sandbox/"):
        relative = normalized_raw.removeprefix("sandbox/").lstrip("/") or "."
    else:
        relative = normalized_raw

    return normalize_sandbox_path(relative, root)


def normalize_workspace_path(raw_path: str) -> Path:
    """Нормализует путь относительно sandbox/project."""
    return normalize_sandbox_path(raw_path, WORKSPACE_ROOT)
