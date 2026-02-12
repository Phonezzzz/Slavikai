from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

from shared.models import JSONValue, ToolRequest, ToolResult

SANDBOX_ROOT = Path("sandbox").resolve()
WORKSPACE_ROOT = (SANDBOX_ROOT / "project").resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
_workspace_root_current: Path | None = None
ALLOWED_EXTENSIONS = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml"}
MAX_FILE_BYTES = 2_000_000  # 2 MB


def get_workspace_root() -> Path:
    return _workspace_root_current or WORKSPACE_ROOT


def set_workspace_root(root: str | Path | None) -> Path:
    global _workspace_root_current
    if root is None:
        _workspace_root_current = None
        return get_workspace_root()
    raw = Path(root).expanduser()
    candidate = raw.resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"Рабочая директория не найдена: {candidate}")
    _workspace_root_current = candidate
    return _workspace_root_current


def _ensure_in_workspace(raw_path: str) -> Path:
    workspace_root = get_workspace_root()
    candidate = (workspace_root / raw_path).resolve()
    if not str(candidate).startswith(str(workspace_root)):
        raise ValueError("Путь вне рабочей директории запрещён.")
    return candidate


def _read_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError("Файл слишком большой для чтения.")
    return path.read_text(encoding="utf-8")


def _validate_patch_contract(diff_text: str) -> str | None:
    lines = diff_text.splitlines()
    if any(line.startswith(prefix) for line in lines for prefix in ("diff --git ", "--- ", "+++ ")):
        return (
            "workspace_patch поддерживает только single-file hunk patch "
            "(без diff --git/---/+++ заголовков)."
        )
    if not any(line.startswith("@@") for line in lines):
        return "Patch должен содержать хотя бы один hunk (@@ ... @@)."
    return None


def _apply_unified_patch(original: str, diff_text: str) -> str:
    """
    Простейшее применение unified diff к одному файлу.
    Без поддержки переименований/бинарных файлов.
    """
    lines = original.splitlines(keepends=True)
    diff_lines = diff_text.splitlines()

    def parse_hunk_header(header: str) -> tuple[int, int]:
        # @@ -l,s +l,s @@
        parts = header.strip("@ ").split()
        minus = parts[0]  # -l,s
        plus = parts[1]  # +l,s
        start_old = int(minus.split(",")[0].lstrip("-"))
        start_new = int(plus.split(",")[0].lstrip("+"))
        return start_old, start_new

    new_lines: list[str] = []
    idx = 0  # current position in original
    it: Iterator[str] = iter(diff_lines)
    for line in it:
        if not line.startswith("@@"):
            continue
        start_old, _ = parse_hunk_header(line)
        # copy unchanged section before hunk
        target_index = start_old - 1
        if target_index < idx or target_index > len(lines):
            raise ValueError("Hunk не соответствует файлу.")
        new_lines.extend(lines[idx:target_index])
        idx = target_index
        # process hunk lines
        for hunk_line in it:
            if hunk_line.startswith("@@"):
                # next hunk, push back by restarting loop
                # but we already consumed line; use recursion-like approach
                # easiest: process by resetting iterator with current line
                it = _prepend(hunk_line, it)
                break
            if not hunk_line:
                continue
            prefix = hunk_line[0]
            content = hunk_line[1:]
            if prefix == " ":
                if idx >= len(lines) or lines[idx].rstrip("\n") != content.rstrip("\n"):
                    raise ValueError("Hunk не совпадает с оригиналом.")
                new_lines.append(lines[idx])
                idx += 1
            elif prefix == "-":
                if idx >= len(lines) or lines[idx].rstrip("\n") != content.rstrip("\n"):
                    raise ValueError("Hunk не совпадает с оригиналом.")
                idx += 1  # skip (delete)
            elif prefix == "+":
                new_lines.append(content + ("\n" if content and not content.endswith("\n") else ""))
            elif prefix == "\\":
                # "\ No newline" -- ignore
                continue
            else:
                raise ValueError("Неизвестный префикс патча.")
    # append remaining original
    new_lines.extend(lines[idx:])
    return "".join(new_lines)


def _prepend(first: str, iterator: Iterator[str]) -> Iterator[str]:
    yield first
    yield from iterator


class ListFilesTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        root = get_workspace_root()
        tree = self._walk(root)
        return ToolResult.success({"output": "ok", "tree": tree})

    def _walk(self, path: Path) -> list[dict[str, JSONValue]]:
        entries: list[dict[str, JSONValue]] = []
        for item in sorted(path.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                child = self._walk(item)
                if child:
                    entries.append({"name": item.name, "type": "dir", "children": child})
            else:
                if item.suffix.lower() not in ALLOWED_EXTENSIONS:
                    continue
                entries.append(
                    {
                        "name": item.name,
                        "type": "file",
                        "path": str(item.relative_to(get_workspace_root())),
                    }
                )
        return entries


class ReadFileTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        raw_path = str(request.args.get("path") or "").strip()
        if not raw_path:
            return ToolResult.failure("Не указан путь файла.")
        try:
            path = _ensure_in_workspace(raw_path)
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                return ToolResult.failure("Расширение файла запрещено для чтения.")
            content = _read_file(path)
            return ToolResult.success({"output": content, "path": str(path)})
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))


class WriteFileTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        raw_path = str(request.args.get("path") or "").strip()
        content = request.args.get("content", "")
        if not raw_path:
            return ToolResult.failure("Не указан путь файла.")
        if not isinstance(content, str):
            return ToolResult.failure("content должен быть строкой.")
        try:
            path = _ensure_in_workspace(raw_path)
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                return ToolResult.failure("Расширение файла запрещено для записи.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult.success({"output": "Файл сохранён.", "path": str(path)})
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))


class ApplyPatchTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        raw_path = str(request.args.get("path") or "").strip()
        diff_text = str(request.args.get("patch") or "")
        dry_run = bool(request.args.get("dry_run", False))
        if not raw_path or not diff_text.strip():
            return ToolResult.failure("Нужны path и patch.")
        contract_error = _validate_patch_contract(diff_text)
        if contract_error is not None:
            return ToolResult.failure(contract_error)
        try:
            path = _ensure_in_workspace(raw_path)
            original = _read_file(path)
            try:
                patched = _apply_unified_patch(original, diff_text)
            except Exception as exc:  # noqa: BLE001
                return ToolResult.failure(f"Патч не применён: {exc}")
            if not dry_run:
                path.write_text(patched, encoding="utf-8")
            return ToolResult.success(
                {
                    "output": "Патч применён." if not dry_run else "Патч проверен (dry run).",
                    "path": str(path),
                    "content": patched,
                    "dry_run": dry_run,
                },
                meta={"bytes": len(patched.encode("utf-8"))},
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))


class RunCodeTool:
    def __init__(self, timeout: int = 3) -> None:
        self.timeout = timeout

    def handle(self, request: ToolRequest) -> ToolResult:
        raw_path = str(request.args.get("path") or "").strip()
        if not raw_path:
            return ToolResult.failure("Не указан путь скрипта.")
        try:
            path = _ensure_in_workspace(raw_path)
            if path.suffix != ".py":
                return ToolResult.failure("Можно запускать только .py файлы.")
            if not path.exists():
                return ToolResult.failure("Файл не найден.")
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=get_workspace_root(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return ToolResult.success(
                {
                    "output": proc.stdout,
                    "stderr": proc.stderr,
                    "exit_code": proc.returncode,
                }
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"Время выполнения превышено ({self.timeout}с).")
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Ошибка запуска: {exc}")
