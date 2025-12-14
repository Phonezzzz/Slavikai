from __future__ import annotations

import re
from pathlib import Path

_WINDOWS_DRIVE_RE = re.compile(r"^[a-zA-Z]:")
_UNC_RE = re.compile(r"^\\\\")


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

    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SandboxViolationError(raw, candidate) from exc
    return candidate
