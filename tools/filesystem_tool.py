from __future__ import annotations

from pathlib import Path
from typing import Final

from shared.models import ToolRequest, ToolResult

SANDBOX_ROOT: Final[Path] = Path("sandbox").resolve()
MAX_READ_BYTES: Final[int] = 512_000  # ~500 KB
MAX_WRITE_BYTES: Final[int] = 1_000_000  # ~1 MB

SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)


def _normalize_path(raw_path: str) -> Path:
    candidate = (SANDBOX_ROOT / raw_path).resolve()
    if not str(candidate).startswith(str(SANDBOX_ROOT)):
        raise ValueError("Путь вне песочнице (sandbox) запрещён.")
    return candidate


def handle_filesystem(request: ToolRequest) -> ToolResult:
    """
    Файловый инструмент с песочницей.
    Поддержка:
    - list: список файлов/папок в заданном каталоге песочницы;
    - read: чтение файла с ограничением размера;
    - write: запись текста в файл (создаёт каталоги).
    """
    name = str(request.args.get("op") or request.name)
    args = request.args
    path_value = str(args.get("path", "") or "")

    try:
        if name == "list":
            target_dir = _normalize_path(path_value or ".")
            if not target_dir.exists():
                return ToolResult.failure(f"Каталог не найден: {path_value}")
            if not target_dir.is_dir():
                return ToolResult.failure(f"Ожидался каталог, получен файл: {path_value}")
            entries = sorted(p.name for p in target_dir.iterdir())
            return ToolResult.success(
                {"output": "\n".join(entries), "entries": entries},
                meta={"count": len(entries), "path": str(target_dir)},
            )

        if name == "read":
            target_file = _normalize_path(path_value)
            if not target_file.exists() or not target_file.is_file():
                return ToolResult.failure(f"Файл не найден: {path_value}")
            size = target_file.stat().st_size
            if size > MAX_READ_BYTES:
                return ToolResult.failure(
                    f"Файл слишком большой ({size} байт), лимит {MAX_READ_BYTES}."
                )
            content = target_file.read_text(encoding="utf-8")
            return ToolResult.success(
                {"output": content, "path": str(target_file)},
                meta={"bytes_read": size},
            )

        if name == "write":
            target_file = _normalize_path(path_value)
            content = str(args.get("content", "") or "")
            if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
                return ToolResult.failure("Лимит записи превышен (макс 1 МБ).")
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content, encoding="utf-8")
            return ToolResult.success(
                {"output": f"Файл записан: {path_value}", "path": str(target_file)},
                meta={"bytes_written": len(content.encode("utf-8"))},
            )

        return ToolResult.failure("Неизвестная операция FS.")
    except ValueError as err:
        return ToolResult.failure(str(err))
    except Exception as exc:  # noqa: BLE001
        return ToolResult.failure(f"FS ошибка: {exc}")


class FilesystemTool:
    """Класс-обёртка для файлового инструмента (Tool.handle совместимый)."""

    def handle(self, request: ToolRequest) -> ToolResult:  # noqa: D401 - очевидно
        return handle_filesystem(request)
