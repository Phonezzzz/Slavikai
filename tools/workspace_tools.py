from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from config.shell_config import DEFAULT_SHELL_CONFIG_PATH, load_shell_config
from shared.models import JSONValue, ToolRequest, ToolResult
from tools.shell_tool import _is_unsafe as _is_unsafe_shell_command
from tools.shell_tool import _validate_args as _validate_shell_args

SANDBOX_ROOT = Path("sandbox").resolve()
WORKSPACE_ROOT = (SANDBOX_ROOT / "project").resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
_workspace_root_current: ContextVar[Path | None] = ContextVar(
    "workspace_root_current",
    default=None,
)
ALLOWED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".css",
    ".html",
    ".sql",
    ".sh",
}
IGNORED_DIR_NAMES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "venv",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".cache",
}
MAX_FILE_BYTES = 2_000_000  # 2 MB
MAX_TREE_ENTRIES = 2_500
MAX_TREE_DIRS = 800
MAX_TREE_FILES = 1_700
MAX_CHILDREN_PER_DIR = 300


def get_workspace_root() -> Path:
    return _workspace_root_current.get() or WORKSPACE_ROOT


def set_workspace_root(root: str | Path | None) -> Path:
    if root is None:
        _workspace_root_current.set(None)
        return get_workspace_root()
    candidate = _resolve_workspace_root(root)
    _workspace_root_current.set(candidate)
    return candidate


@contextmanager
def workspace_root_context(root: str | Path | None) -> Iterator[Path]:
    candidate: Path | None
    if root is None:
        candidate = None
    else:
        candidate = _resolve_workspace_root(root)
    token = _workspace_root_current.set(candidate)
    try:
        yield get_workspace_root()
    finally:
        _workspace_root_current.reset(token)


def _resolve_workspace_root(root: str | Path) -> Path:
    raw = Path(root).expanduser()
    candidate = raw.resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise ValueError(f"Рабочая директория не найдена: {candidate}")
    return candidate


def _ensure_in_workspace(raw_path: str) -> Path:
    workspace_root = get_workspace_root()
    candidate = (workspace_root / raw_path).resolve()
    try:
        candidate.relative_to(workspace_root)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Путь вне рабочей директории запрещён.") from exc
    return candidate


def _read_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError("Файл слишком большой для чтения.")
    return path.read_text(encoding="utf-8")


def _is_allowed_workspace_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


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
    class _WalkState:
        def __init__(self, *, max_depth_applied: int) -> None:
            self.returned_entries = 0
            self.returned_dirs = 0
            self.returned_files = 0
            self.max_depth_applied = max_depth_applied
            self.truncated_reasons: set[str] = set()

        def _mark(self, reason: str) -> None:
            self.truncated_reasons.add(reason)

        def can_append_dir(self) -> bool:
            if self.returned_entries >= MAX_TREE_ENTRIES:
                self._mark("max_entries")
                return False
            if self.returned_dirs >= MAX_TREE_DIRS:
                self._mark("max_dirs")
                return False
            return True

        def can_append_file(self) -> bool:
            if self.returned_entries >= MAX_TREE_ENTRIES:
                self._mark("max_entries")
                return False
            if self.returned_files >= MAX_TREE_FILES:
                self._mark("max_files")
                return False
            return True

        def consume_dir(self) -> None:
            self.returned_entries += 1
            self.returned_dirs += 1

        def consume_file(self) -> None:
            self.returned_entries += 1
            self.returned_files += 1

        def global_limit_reached(self) -> bool:
            return (
                self.returned_entries >= MAX_TREE_ENTRIES
                or self.returned_dirs >= MAX_TREE_DIRS
                or self.returned_files >= MAX_TREE_FILES
            )

        def meta(self) -> dict[str, JSONValue]:
            return {
                "returned_entries": self.returned_entries,
                "returned_dirs": self.returned_dirs,
                "returned_files": self.returned_files,
                "truncated": bool(self.truncated_reasons),
                "truncated_reasons": sorted(self.truncated_reasons),
                "max_depth_applied": self.max_depth_applied,
                "max_entries": MAX_TREE_ENTRIES,
                "max_dirs": MAX_TREE_DIRS,
                "max_files": MAX_TREE_FILES,
                "max_children_per_dir": MAX_CHILDREN_PER_DIR,
            }

    def handle(self, request: ToolRequest) -> ToolResult:
        root = get_workspace_root()
        raw_path = str(request.args.get("path") or "").strip()
        recursive = bool(request.args.get("recursive", False))
        max_depth_raw = request.args.get("max_depth")
        max_depth = 12
        if isinstance(max_depth_raw, int) and max_depth_raw >= 0:
            max_depth = max_depth_raw
        max_depth_applied = max_depth if recursive else 0
        state = self._WalkState(max_depth_applied=max_depth_applied)

        try:
            target = root if not raw_path else _ensure_in_workspace(raw_path)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))
        if not target.exists() or not target.is_dir():
            return ToolResult.failure(f"Директория не найдена: {target}")

        tree, _ = self._walk(
            target,
            base_root=root,
            recursive=recursive,
            depth=0,
            max_depth=max_depth,
            state=state,
        )
        return ToolResult.success(
            {
                "output": "ok",
                "tree": tree,
                "path": str(target.relative_to(root)) if target != root else "",
                "tree_meta": state.meta(),
            }
        )

    def _dir_has_visible_children(self, path: Path) -> bool:
        try:
            for item in path.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_dir() and item.name not in IGNORED_DIR_NAMES:
                    return True
                if item.is_file() and _is_allowed_workspace_file(item):
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _walk(
        self,
        path: Path,
        *,
        base_root: Path,
        recursive: bool,
        depth: int,
        max_depth: int,
        state: _WalkState,
    ) -> tuple[list[dict[str, JSONValue]], bool]:
        entries: list[dict[str, JSONValue]] = []
        children_truncated = False
        try:
            iter_items = sorted(
                path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())
            )
        except Exception:  # noqa: BLE001
            return entries, children_truncated

        visible_children = 0
        for item in iter_items:
            if item.name.startswith("."):
                continue
            is_dir = item.is_dir() and item.name not in IGNORED_DIR_NAMES
            is_file = item.is_file() and _is_allowed_workspace_file(item)
            if not is_dir and not is_file:
                continue
            if visible_children >= MAX_CHILDREN_PER_DIR:
                state._mark("max_children_per_dir")
                children_truncated = True
                break
            visible_children += 1
            if is_dir:
                if not state.can_append_dir():
                    children_truncated = True
                    break
                state.consume_dir()
                rel_path = str(item.relative_to(base_root))
                node: dict[str, JSONValue] = {
                    "name": item.name,
                    "type": "dir",
                    "path": rel_path,
                    "has_children": self._dir_has_visible_children(item),
                }
                if recursive and depth < max_depth:
                    children, node_children_truncated = self._walk(
                        item,
                        base_root=base_root,
                        recursive=recursive,
                        depth=depth + 1,
                        max_depth=max_depth,
                        state=state,
                    )
                    node["children"] = children
                    if node_children_truncated:
                        node["children_truncated"] = True
                entries.append(node)
                if state.global_limit_reached() and state.truncated_reasons:
                    children_truncated = True
                    break
                continue

            if not state.can_append_file():
                children_truncated = True
                break
            state.consume_file()
            entries.append(
                {
                    "name": item.name,
                    "type": "file",
                    "path": str(item.relative_to(base_root)),
                }
            )
        return entries, children_truncated


class ReadFileTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        raw_path = str(request.args.get("path") or "").strip()
        if not raw_path:
            return ToolResult.failure("Не указан путь файла.")
        try:
            path = _ensure_in_workspace(raw_path)
            if not _is_allowed_workspace_file(path):
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
            if not _is_allowed_workspace_file(path):
                return ToolResult.failure("Расширение файла запрещено для записи.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult.success({"output": "Файл сохранён.", "path": str(path)})
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))


class CreateFileTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        raw_path = str(request.args.get("path") or "").strip()
        content = request.args.get("content", "")
        overwrite = bool(request.args.get("overwrite", False))
        if not raw_path:
            return ToolResult.failure("Не указан путь файла.")
        if not isinstance(content, str):
            return ToolResult.failure("content должен быть строкой.")
        try:
            path = _ensure_in_workspace(raw_path)
            if not _is_allowed_workspace_file(path):
                return ToolResult.failure("Расширение файла запрещено для создания.")
            if path.exists() and path.is_dir():
                return ToolResult.failure("По указанному пути уже существует директория.")
            if path.exists() and not overwrite:
                return ToolResult.failure("Файл уже существует. Укажите overwrite=true.")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult.success({"output": "Файл создан.", "path": str(path)})
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))


class RenameFileTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        old_path_raw = str(request.args.get("old_path") or "").strip()
        new_path_raw = str(request.args.get("new_path") or "").strip()
        if not old_path_raw or not new_path_raw:
            return ToolResult.failure("Нужны old_path и new_path.")
        try:
            old_path = _ensure_in_workspace(old_path_raw)
            new_path = _ensure_in_workspace(new_path_raw)
            if not old_path.exists():
                return ToolResult.failure("Исходный путь не найден.")
            if old_path.is_file() and not _is_allowed_workspace_file(old_path):
                return ToolResult.failure("Расширение файла запрещено для переименования.")
            if new_path.exists():
                return ToolResult.failure("Целевой путь уже существует.")
            if new_path.suffix and not _is_allowed_workspace_file(new_path):
                return ToolResult.failure("Расширение файла запрещено для переименования.")
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)
            return ToolResult.success(
                {
                    "output": "Файл переименован.",
                    "old_path": str(old_path),
                    "new_path": str(new_path),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))


class MoveFileTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        from_path_raw = str(request.args.get("from_path") or "").strip()
        to_path_raw = str(request.args.get("to_path") or "").strip()
        if not from_path_raw or not to_path_raw:
            return ToolResult.failure("Нужны from_path и to_path.")
        try:
            source = _ensure_in_workspace(from_path_raw)
            target = _ensure_in_workspace(to_path_raw)
            if not source.exists():
                return ToolResult.failure("Исходный путь не найден.")
            if source.is_file() and not _is_allowed_workspace_file(source):
                return ToolResult.failure("Расширение файла запрещено для перемещения.")
            if target.exists():
                return ToolResult.failure("Целевой путь уже существует.")
            if target.suffix and not _is_allowed_workspace_file(target):
                return ToolResult.failure("Расширение файла запрещено для перемещения.")
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            return ToolResult.success(
                {
                    "output": "Файл перемещён.",
                    "from_path": str(source),
                    "to_path": str(target),
                }
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(str(exc))


class DeleteFileTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        raw_path = str(request.args.get("path") or "").strip()
        recursive = bool(request.args.get("recursive", False))
        if not raw_path:
            return ToolResult.failure("Не указан путь.")
        try:
            target = _ensure_in_workspace(raw_path)
            if not target.exists():
                return ToolResult.failure("Путь не найден.")
            if target.is_dir():
                if not recursive:
                    return ToolResult.failure("Удаление директории требует recursive=true.")
                shutil.rmtree(target)
            else:
                target.unlink()
            return ToolResult.success({"output": "Удалено.", "path": str(target)})
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
            if not _is_allowed_workspace_file(path):
                return ToolResult.failure("Расширение файла запрещено для patch.")
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


class WorkspaceTerminalRunTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        command = str(request.args.get("command") or "").strip()
        cwd_mode = str(request.args.get("cwd_mode") or "session_root").strip().lower()
        if not command:
            return ToolResult.failure("Не указана команда.")
        if _is_unsafe_shell_command(command):
            return ToolResult.failure("🚫 Опасная команда заблокирована.")

        try:
            cfg = load_shell_config(DEFAULT_SHELL_CONFIG_PATH)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Ошибка загрузки shell config: {exc}")

        try:
            args = shlex.split(command)
        except ValueError as exc:
            return ToolResult.failure(f"Ошибка парсинга команды: {exc}")
        validation_error = _validate_shell_args(args, set(cfg.allowed_commands))
        if validation_error:
            return ToolResult.failure(validation_error)

        if cwd_mode == "session_root":
            cwd = get_workspace_root()
        elif cwd_mode == "sandbox":
            cwd = SANDBOX_ROOT
        else:
            return ToolResult.failure("cwd_mode должен быть session_root|sandbox.")
        if not cwd.exists() or not cwd.is_dir():
            return ToolResult.failure("Рабочая директория для команды не найдена.")

        try:
            proc = subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=cfg.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(
                f"⏳ Команда превысила лимит времени ({cfg.timeout_seconds}с)."
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Ошибка запуска: {exc}")

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        if len(stdout) > cfg.max_output_chars:
            stdout = stdout[: cfg.max_output_chars] + "\n…[stdout truncated]"
        if len(stderr) > cfg.max_output_chars:
            stderr = stderr[: cfg.max_output_chars] + "\n…[stderr truncated]"
        return ToolResult.success(
            {
                "output": stdout,
                "stderr": stderr,
                "exit_code": proc.returncode,
                "cwd": str(cwd),
                "cwd_mode": cwd_mode,
            }
        )
