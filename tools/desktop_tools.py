from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tarfile
import time
import uuid
import zipfile
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import psutil

from core.desktop_policy import classify_command
from core.desktop_security import DesktopPathSecurity
from shared.models import JSONValue, ToolRequest, ToolResult

MAX_DESKTOP_READ_BYTES = 1_048_576
MAX_SEARCH_RESULTS = 200
MAX_ARCHIVE_ENTRIES = 5_000
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_PROCESS_OUTPUT_CHARS = 64_000


class DesktopFileSearchTool:
    def __init__(self, security: DesktopPathSecurity) -> None:
        self.security = security

    def handle(self, request: ToolRequest) -> ToolResult:
        root_raw = _string_arg(request, "root") or str(self.security.home)
        query = (_string_arg(request, "query") or "").casefold()
        limit = _bounded_int(request.args.get("limit"), default=50, minimum=1, maximum=200)
        try:
            modified_after = _parse_timestamp(request.args.get("modified_after"))
            root = self.security.require_not_denied(root_raw, must_exist=True).canonical
        except (OSError, ValueError, PermissionError) as exc:
            return ToolResult.failure(f"Desktop search path rejected: {exc}")
        if not root.is_dir():
            return ToolResult.failure("Desktop search root is not a directory.")
        matches: list[dict[str, JSONValue]] = []
        try:
            for current, dirs, files in os.walk(root, followlinks=False):
                current_path = Path(current)
                visible_dirs: list[str] = []
                for name in dirs:
                    child = current_path / name
                    try:
                        if self.security.resolve(str(child)).protection != "deny":
                            visible_dirs.append(name)
                    except (OSError, ValueError):
                        continue
                dirs[:] = visible_dirs
                for name in [*visible_dirs, *files]:
                    if query and query not in name.casefold():
                        continue
                    candidate = current_path / name
                    try:
                        stat = candidate.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if modified_after is not None and stat.st_mtime < modified_after:
                        continue
                    matches.append(
                        {
                            "path": str(candidate.resolve(strict=False)),
                            "name": name,
                            "type": "directory" if candidate.is_dir() else "file",
                            "size": stat.st_size,
                            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        }
                    )
                    if len(matches) >= limit:
                        return ToolResult.success(
                            {"output": f"Found {len(matches)} matching paths.", "matches": matches},
                            meta={"truncated": True, "root": str(root)},
                        )
        except OSError as exc:
            return ToolResult.failure(f"Desktop search failed: {exc}")
        matches.sort(key=lambda item: str(item.get("modified_at") or ""), reverse=True)
        return ToolResult.success(
            {"output": f"Found {len(matches)} matching paths.", "matches": matches},
            meta={"truncated": False, "root": str(root)},
        )


class DesktopFileReadTool:
    def __init__(self, security: DesktopPathSecurity) -> None:
        self.security = security

    def handle(self, request: ToolRequest) -> ToolResult:
        path_raw = _string_arg(request, "path")
        if path_raw is None:
            return ToolResult.failure("path is required")
        try:
            path = self.security.require_not_denied(path_raw, must_exist=True).canonical
            if not path.is_file():
                return ToolResult.failure("Desktop path is not a file.")
            size = path.stat().st_size
            if size > MAX_DESKTOP_READ_BYTES:
                return ToolResult.failure(
                    f"Desktop file exceeds read limit ({MAX_DESKTOP_READ_BYTES} bytes)."
                )
            raw = path.read_bytes()
        except (OSError, ValueError, PermissionError) as exc:
            return ToolResult.failure(f"Desktop read failed: {exc}")
        if b"\x00" in raw:
            return ToolResult.failure("Binary file cannot be returned as text.")
        content = raw.decode("utf-8", errors="replace")
        return ToolResult.success(
            {"output": content, "content": content, "path": str(path), "size": len(raw)}
        )


class DesktopFileWriteTool:
    def __init__(self, security: DesktopPathSecurity) -> None:
        self.security = security

    def handle(self, request: ToolRequest) -> ToolResult:
        path_raw = _string_arg(request, "path")
        content = request.args.get("content")
        overwrite = request.args.get("overwrite") is True
        if path_raw is None or not isinstance(content, str):
            return ToolResult.failure("path and string content are required")
        try:
            resolved = self.security.require_not_denied(path_raw, mutation=True)
            path = resolved.canonical
            self.security.require_not_denied(str(path.parent))
            if path.exists() and not overwrite:
                return ToolResult.failure("Target already exists; set overwrite=true explicitly.")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temp.write_text(content, encoding="utf-8")
                os.replace(temp, path)
            finally:
                if temp.exists():
                    temp.unlink()
            written_size = path.stat().st_size
        except (OSError, ValueError, PermissionError) as exc:
            return ToolResult.failure(f"Desktop write failed: {exc}")
        return ToolResult.success(
            {
                "output": f"Wrote {len(content.encode('utf-8'))} bytes to {path}",
                "path": str(path),
                "size": written_size,
            }
        )


class DesktopFileTransferTool:
    def __init__(self, security: DesktopPathSecurity) -> None:
        self.security = security

    def handle(self, request: ToolRequest) -> ToolResult:
        operation = (_string_arg(request, "operation") or "move").lower()
        source_raw = _string_arg(request, "source")
        destination_raw = _string_arg(request, "destination")
        overwrite = request.args.get("overwrite") is True
        if operation not in {"copy", "move", "rename"}:
            return ToolResult.failure("operation must be copy|move|rename")
        if source_raw is None or destination_raw is None:
            return ToolResult.failure("source and destination are required")
        staging: Path | None = None
        try:
            source = self.security.require_not_denied(
                source_raw,
                must_exist=True,
                mutation=operation != "copy",
            ).canonical
            destination = self.security.require_not_denied(
                destination_raw,
                mutation=True,
            ).canonical
            self.security.require_not_denied(str(destination.parent))
            backup: Path | None = None
            if destination.exists():
                if not overwrite:
                    return ToolResult.failure("Destination already exists.")
                backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.bak")
                os.replace(destination, backup)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                if operation == "copy":
                    staging = destination.with_name(
                        f".{destination.name}.{uuid.uuid4().hex}.staging"
                    )
                    if source.is_dir():
                        shutil.copytree(source, staging, symlinks=True)
                    else:
                        shutil.copy2(source, staging, follow_symlinks=False)
                    os.replace(staging, destination)
                else:
                    shutil.move(str(source), str(destination))
            except (OSError, shutil.Error):
                if staging is not None:
                    _remove_path(staging)
                _rollback_transfer(
                    operation=operation,
                    source=source,
                    destination=destination,
                    backup=backup,
                )
                raise
            if backup is not None:
                _remove_path(backup)
        except (OSError, ValueError, PermissionError, shutil.Error) as exc:
            return ToolResult.failure(f"Desktop transfer failed: {exc}")
        return ToolResult.success(
            {
                "output": f"{operation} completed: {source} -> {destination}",
                "source": str(source),
                "destination": str(destination),
                "operation": operation,
            }
        )


class DesktopFileDeleteTool:
    """Recoverable delete: moves a target into the user's freedesktop trash directory."""

    def __init__(self, security: DesktopPathSecurity, trash_root: Path | None = None) -> None:
        self.security = security
        self.trash_root = (
            trash_root.resolve()
            if trash_root is not None
            else (security.home / ".local/share/Trash/files").resolve()
        )

    def handle(self, request: ToolRequest) -> ToolResult:
        path_raw = _string_arg(request, "path")
        if path_raw is None:
            return ToolResult.failure("path is required")
        try:
            source = self.security.require_not_denied(
                path_raw,
                must_exist=True,
                mutation=True,
            ).canonical
            self.security.require_not_denied(str(self.trash_root))
            self.trash_root.mkdir(parents=True, exist_ok=True)
            target = self.trash_root / source.name
            if target.exists():
                target = self.trash_root / f"{source.stem}-{uuid.uuid4().hex[:8]}{source.suffix}"
            shutil.move(str(source), str(target))
        except (OSError, ValueError, PermissionError, shutil.Error) as exc:
            return ToolResult.failure(f"Desktop delete failed: {exc}")
        return ToolResult.success(
            {
                "output": f"Moved to trash: {target}",
                "path": str(source),
                "trash_path": str(target),
                "recoverable": True,
            }
        )


class DesktopArchiveExtractTool:
    def __init__(self, security: DesktopPathSecurity) -> None:
        self.security = security

    def handle(self, request: ToolRequest) -> ToolResult:
        archive_raw = _string_arg(request, "archive")
        destination_raw = _string_arg(request, "destination")
        if archive_raw is None or destination_raw is None:
            return ToolResult.failure("archive and destination are required")
        staging: Path | None = None
        try:
            archive = self.security.require_not_denied(archive_raw, must_exist=True).canonical
            destination = self.security.require_not_denied(
                destination_raw,
                mutation=True,
            ).canonical
            self.security.require_not_denied(str(destination.parent))
            if destination.exists():
                return ToolResult.failure("Archive destination already exists.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.staging")
            staging.mkdir()
            if zipfile.is_zipfile(archive):
                staged_entries = self._extract_zip(archive, staging)
            elif tarfile.is_tarfile(archive):
                staged_entries = self._extract_tar(archive, staging)
            else:
                _remove_path(staging)
                return ToolResult.failure("Unsupported archive format (expected zip or tar).")
            os.replace(staging, destination)
            extracted = [
                str(destination / Path(entry).relative_to(staging)) for entry in staged_entries
            ]
        except (OSError, ValueError, PermissionError, zipfile.BadZipFile, tarfile.TarError) as exc:
            if staging is not None and staging.exists():
                _remove_path(staging)
            return ToolResult.failure(f"Archive extraction failed: {exc}")
        return ToolResult.success(
            {
                "output": f"Extracted {len(extracted)} entries to {destination}",
                "archive": str(archive),
                "destination": str(destination),
                "entries": extracted[:MAX_SEARCH_RESULTS],
            },
            meta={"entries_total": len(extracted)},
        )

    def _extract_zip(self, archive: Path, destination: Path) -> list[str]:
        extracted: list[str] = []
        total_size = 0
        with zipfile.ZipFile(archive) as handle:
            infos = handle.infolist()
            if len(infos) > MAX_ARCHIVE_ENTRIES:
                raise ValueError("Archive contains too many entries")
            for info in infos:
                total_size += max(0, info.file_size)
                if total_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive exceeds extraction size limit")
                target = _safe_archive_target(destination, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                extracted.append(str(target))
        return extracted

    def _extract_tar(self, archive: Path, destination: Path) -> list[str]:
        extracted: list[str] = []
        total_size = 0
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            if len(members) > MAX_ARCHIVE_ENTRIES:
                raise ValueError("Archive contains too many entries")
            for member in members:
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"Unsafe tar entry: {member.name}")
                total_size += max(0, member.size)
                if total_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Archive exceeds extraction size limit")
                target = _safe_archive_target(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    source = handle.extractfile(member)
                    if source is None:
                        raise ValueError(f"Cannot read tar entry: {member.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                extracted.append(str(target))
        return extracted


class DesktopShellTool:
    def __init__(
        self,
        security: DesktopPathSecurity,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.security = security
        self.cancelled = cancelled or (lambda: False)

    def handle(self, request: ToolRequest) -> ToolResult:
        argv = _argv_arg(request)
        cwd_raw = _string_arg(request, "cwd") or str(self.security.home)
        timeout = _bounded_int(
            request.args.get("timeout_seconds"),
            default=30,
            minimum=1,
            maximum=120,
        )
        if argv is None:
            return ToolResult.failure("argv must be a non-empty string array")
        command_class = classify_command(argv)
        if command_class in {"privilege_escalation", "disk_boot", "shell_indirection"}:
            return ToolResult.failure(f"Desktop command denied: {command_class}")
        if command_class in {"package_management", "service_management", "process_management"}:
            return ToolResult.failure(
                f"Desktop command must use the corresponding typed capability: {command_class}"
            )
        try:
            cwd = self.security.require_not_denied(cwd_raw, must_exist=True).canonical
            if not cwd.is_dir():
                return ToolResult.failure("cwd is not a directory")
            _validate_command_paths(
                argv,
                cwd,
                self.security,
                mutation=command_class in {"filesystem_mutation", "network"},
            )
            started = time.monotonic()
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_desktop_subprocess_env(),
                start_new_session=True,
            )
            completed = _communicate_bounded(
                process,
                timeout_seconds=timeout,
                cancelled=self.cancelled,
            )
            duration = time.monotonic() - started
        except DesktopProcessStopped as exc:
            return ToolResult.failure(str(exc), meta={"cwd": str(cwd_raw)})
        except (OSError, ValueError, PermissionError) as exc:
            return ToolResult.failure(f"Desktop command failed: {exc}")
        stdout = _truncate_output(completed.stdout or "")
        stderr = _truncate_output(completed.stderr or "")
        data: dict[str, JSONValue] = {
            "output": stdout,
            "stderr": stderr,
            "exit_code": completed.returncode,
            "cwd": str(cwd),
            "command_class": command_class,
            "duration_seconds": round(duration, 3),
        }
        if completed.returncode != 0:
            return ToolResult.failure(
                f"Desktop command exited with code {completed.returncode}.",
                meta=data,
            )
        return ToolResult.success(data)


class DesktopLaunchTool:
    def __init__(
        self,
        security: DesktopPathSecurity,
        *,
        cancelled: Callable[[], bool] | None = None,
        on_launch: Callable[[int], None] | None = None,
    ) -> None:
        self.security = security
        self.cancelled = cancelled or (lambda: False)
        self.on_launch = on_launch or (lambda _pid: None)

    def handle(self, request: ToolRequest) -> ToolResult:
        result, _process = self.start(request)
        return result

    def start(
        self,
        request: ToolRequest,
    ) -> tuple[ToolResult, subprocess.Popen[bytes] | None]:
        argv = _argv_arg(request)
        cwd_raw = _string_arg(request, "cwd") or str(self.security.home)
        if argv is None:
            return ToolResult.failure("argv must be a non-empty string array"), None
        if self.cancelled():
            return ToolResult.failure("Desktop launch cancelled before execution."), None
        command_class = classify_command(argv)
        if command_class in {
            "privilege_escalation",
            "disk_boot",
            "shell_indirection",
            "package_management",
            "service_management",
            "process_management",
            "filesystem_mutation",
        }:
            return ToolResult.failure(f"Desktop launch denied: {command_class}"), None
        try:
            cwd = self.security.require_not_denied(cwd_raw, must_exist=True).canonical
            _validate_command_paths(argv, cwd, self.security, mutation=False)
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=_desktop_subprocess_env(),
            )
            self.on_launch(process.pid)
        except (OSError, ValueError, PermissionError) as exc:
            return ToolResult.failure(f"Desktop launch failed: {exc}"), None
        if self.cancelled():
            _terminate_process_group(process)
            return ToolResult.failure("Desktop launch cancelled during execution."), None
        return (
            ToolResult.success(
                {
                    "output": f"Started process {process.pid}",
                    "pid": process.pid,
                    "argv": argv,
                    "cwd": str(cwd),
                    "command_class": command_class,
                }
            ),
            process,
        )


class DesktopOpenTool:
    def __init__(
        self,
        security: DesktopPathSecurity,
        runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.security = security
        self._runner = runner or self._run

    def handle(self, request: ToolRequest) -> ToolResult:
        target = _string_arg(request, "target")
        if target is None:
            return ToolResult.failure("target is required")
        parsed = urlparse(target)
        normalized_target = target
        if parsed.scheme:
            if parsed.scheme not in {"http", "https"}:
                return ToolResult.failure("Only http/https URLs may be opened.")
        else:
            try:
                normalized_target = str(
                    self.security.require_not_denied(target, must_exist=True).canonical
                )
            except (OSError, ValueError, PermissionError) as exc:
                return ToolResult.failure(f"Desktop open target rejected: {exc}")
        try:
            opener = "gio" if shutil.which("gio") else "xdg-open"
            argv = (
                [opener, "open", normalized_target]
                if opener == "gio"
                else [opener, normalized_target]
            )
            result = self._runner(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult.failure(f"Desktop open failed: {exc}")
        if result.returncode != 0:
            return ToolResult.failure(
                f"Desktop opener exited with code {result.returncode}",
                meta={"stderr": result.stderr or ""},
            )
        return ToolResult.success(
            {"output": f"Opened {normalized_target}", "target": normalized_target}
        )

    @staticmethod
    def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )


class DesktopVerifyTool:
    def __init__(self, security: DesktopPathSecurity) -> None:
        self.security = security

    def handle(self, request: ToolRequest) -> ToolResult:
        check = (_string_arg(request, "check") or "path_exists").lower()
        if check in {"path_exists", "file_contains", "path_missing"}:
            path_raw = _string_arg(request, "path")
            if path_raw is None:
                return ToolResult.failure("path is required")
            try:
                path = self.security.require_not_denied(path_raw).canonical
                exists = path.exists()
                if check == "path_exists" and not exists:
                    return ToolResult.failure("Verification failed: path does not exist.")
                if check == "path_missing" and exists:
                    return ToolResult.failure("Verification failed: path still exists.")
                if check == "file_contains":
                    expected = _string_arg(request, "expected")
                    if expected is None or not path.is_file():
                        return ToolResult.failure("Verification requires expected text and a file.")
                    if path.stat().st_size > MAX_DESKTOP_READ_BYTES:
                        return ToolResult.failure("Verification file exceeds read limit.")
                    if expected not in path.read_text(encoding="utf-8", errors="replace"):
                        return ToolResult.failure("Verification failed: expected text not found.")
            except (OSError, ValueError, PermissionError) as exc:
                return ToolResult.failure(f"Verification failed: {exc}")
            return ToolResult.success(
                {"output": f"Verification passed: {check}", "verified": True, "path": str(path)}
            )
        if check == "process_running":
            pid_raw = request.args.get("pid")
            if not isinstance(pid_raw, int) or pid_raw <= 0:
                return ToolResult.failure("positive integer pid is required")
            expected_raw = request.args.get("expected_create_time")
            try:
                process = psutil.Process(pid_raw)
                if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                    return ToolResult.failure("Verification failed: process is not running.")
                if isinstance(expected_raw, (int, float)) and not isinstance(expected_raw, bool):
                    actual = process.create_time()
                    if abs(actual - float(expected_raw)) > 0.01:
                        return ToolResult.failure(
                            "Verification failed: stale PID create_time does not match."
                        )
            except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
                return ToolResult.failure(f"Verification failed: process is not running: {exc}")
            return ToolResult.success(
                {
                    "output": f"Process {pid_raw} is running",
                    "verified": True,
                    "pid": pid_raw,
                    "create_time": process.create_time(),
                }
            )
        return ToolResult.failure(
            "check must be path_exists|path_missing|file_contains|process_running"
        )


def _safe_archive_target(destination: Path, member_name: str) -> Path:
    if not member_name or "\x00" in member_name:
        raise ValueError("Archive entry has an invalid name")
    target = (destination / member_name).resolve(strict=False)
    try:
        target.relative_to(destination.resolve())
    except ValueError as exc:
        raise ValueError(f"Archive path traversal blocked: {member_name}") from exc
    return target


class DesktopProcessStopped(RuntimeError):
    pass


def _communicate_bounded(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: int,
    cancelled: Callable[[], bool],
) -> subprocess.CompletedProcess[str]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if cancelled():
            _terminate_process_group(process)
            raise DesktopProcessStopped("Desktop command cancelled during execution.")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(process)
            raise DesktopProcessStopped(f"Desktop command timed out after {timeout_seconds}s.")
        try:
            stdout, stderr = process.communicate(timeout=min(0.25, remaining))
        except subprocess.TimeoutExpired:
            continue
        return subprocess.CompletedProcess(
            args=process.args,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )


def _terminate_process_group(
    process: subprocess.Popen[str] | subprocess.Popen[bytes],
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _validate_command_paths(
    argv: list[str],
    cwd: Path,
    security: DesktopPathSecurity,
    *,
    mutation: bool,
) -> None:
    executable = argv[0]
    if executable.startswith(("~", "/", ".")) or "/" in executable:
        executable_path = executable if executable.startswith(("~", "/")) else str(cwd / executable)
        security.require_not_denied(executable_path)
    for arg in argv[1:]:
        if "://" in arg:
            continue
        candidate_arg = arg
        if "=" in candidate_arg:
            candidate_arg = candidate_arg.split("=", 1)[1]
        candidate_arg = candidate_arg.lstrip("@")
        if candidate_arg.startswith("-"):
            continue
        if (
            candidate_arg.startswith("~")
            or candidate_arg.startswith("/")
            or "/" in candidate_arg
            or candidate_arg.startswith(".")
            or mutation
        ):
            candidate = (
                candidate_arg if candidate_arg.startswith(("~", "/")) else str(cwd / candidate_arg)
            )
            security.require_not_denied(candidate, mutation=mutation)


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _rollback_transfer(
    *,
    operation: str,
    source: Path,
    destination: Path,
    backup: Path | None,
) -> None:
    if operation != "copy" and not source.exists() and destination.exists():
        shutil.move(str(destination), str(source))
    else:
        _remove_path(destination)
    if backup is not None and backup.exists():
        os.replace(backup, destination)


def _desktop_subprocess_env() -> dict[str, str]:
    allowed = {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
        "HOME",
        "USER",
        "LOGNAME",
        "TERM",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _argv_arg(request: ToolRequest) -> list[str] | None:
    raw = request.args.get("argv")
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(item, str) and item.strip() for item in raw):
        return None
    return [str(item) for item in raw]


def _string_arg(request: ToolRequest, key: str) -> str | None:
    raw = request.args.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def _parse_timestamp(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip():
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
    raise ValueError("modified_after must be an ISO timestamp or unix timestamp")


def _truncate_output(value: str) -> str:
    if len(value) <= MAX_PROCESS_OUTPUT_CHARS:
        return value
    return value[:MAX_PROCESS_OUTPUT_CHARS] + "\n...[output truncated]"
