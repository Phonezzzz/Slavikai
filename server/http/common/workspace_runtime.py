from __future__ import annotations

import errno
import os
import stat
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from shared.models import JSONValue, ToolRequest, ToolResult
from shared.sandbox import is_private_runtime_path


def _read_mount_points() -> frozenset[Path]:
    """Mount points of the current mount namespace.

    Parsed from /proc/self/mountinfo. A same-device bind mount keeps
    st_dev, so the walk checks this set as well as st_dev — the same
    discipline as host_bridge/fs.py ``_open_dirs``. On platforms
    without procfs (macOS is a supported server platform) the file is
    absent and the set degrades to empty: the st_dev check in the walk
    still refuses cross-device components, only same-device bind
    mounts escape detection there.
    """
    mount_points: set[Path] = set()
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as handle:
            for line in handle:
                fields = line.split(" - ", 1)[0].split()
                if len(fields) < 5:
                    continue
                mount_points.add(
                    Path(
                        fields[4]
                        .replace("\\040", " ")
                        .replace("\\011", "\t")
                        .replace("\\012", "\n")
                        .replace("\\134", "\\")
                    )
                )
    except FileNotFoundError:
        # No procfs (e.g. macOS): degrade to an empty set, see the
        # docstring. Any other read failure stays loud.
        return frozenset()
    except OSError as exc:
        raise ValueError("mount topology unavailable") from exc
    return frozenset(mount_points)


def _map_component_error(exc: OSError, *, dir_fd: int, name: str) -> Exception:
    """Map an openat failure to the pre-fix resolver's error shape.

    ENOENT/ENOTDIR: the path does not resolve to a file below the root
    (a regular file in the middle of the path is ENOTDIR). ELOOP: a
    symlink was met with O_NOFOLLOW. Linux reports ENOTDIR (not ELOOP)
    when O_DIRECTORY|O_NOFOLLOW meets a symlink, so ENOTDIR is
    disambiguated with lstat. Unreadable or over-long names behave as
    "not found", as the old exists()/is_file() checks did.
    """
    err = exc.errno
    if err == errno.ENOENT:
        return FileNotFoundError("file not found")
    if err == errno.ELOOP:
        return ValueError("path outside workspace")
    if err == errno.ENOTDIR:
        try:
            st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            st = None
        if st is not None and stat.S_ISLNK(st.st_mode):
            return ValueError("path outside workspace")
        return FileNotFoundError("file not found")
    if err in (errno.EACCES, errno.ENAMETOOLONG):
        return FileNotFoundError("file not found")
    return exc


def _read_workspace_file_bytes(
    path_raw: str,
    *,
    workspace_root: Path,
    max_download_bytes: int,
) -> bytes:
    """Read a workspace file, bound to the inode opened, not the path.

    SEC-04: the previous resolver validated the pathname (resolve +
    exists + stat) and the caller then read it with ``Path.read_bytes()``.
    An external writer could swap any component for a symlink — or grow
    the file — in between, leaking content from outside the workspace
    or above the size cap. This helper walks the path component by
    component with ``openat`` + ``O_NOFOLLOW`` from a pinned root fd,
    pins the final file with an open fd, and enforces the size cap on
    the bytes actually read from that fd:

    - every directory component is opened ``O_DIRECTORY|O_NOFOLLOW``;
      a symlink is refused (ELOOP, or ENOTDIR on Linux, disambiguated
      with lstat);
    - every component must stay on the root's device and must not be a
      mount point (same-device bind mounts keep st_dev), mirroring
      ``host_bridge/fs.py`` ``_open_dirs``;
    - the final component is opened ``O_RDONLY|O_NONBLOCK|O_NOFOLLOW``
      (O_NONBLOCK so a FIFO cannot block the worker) and must fstat as
      a regular file;
    - the size cap is checked on fstat and again on the bytes read
      (cap + 1), so growth between fstat and read is caught.

    ``..`` and absolute paths are refused outright; private runtime
    paths (``is_private_runtime_path``) are refused as before.
    """
    normalized = path_raw.strip()
    if not normalized:
        raise ValueError("path required")
    if max_download_bytes <= 0:
        raise ValueError("max_download_bytes must be positive")
    parsed = PurePosixPath(normalized)
    parts = parsed.parts
    # is_absolute() catches "//" too: PurePosixPath("//tmp/x").parts[0]
    # is "//", not "/", so the old parts[0] == "/" check missed it and
    # os.open("//", dir_fd=...) ignored the workspace fd.
    if not parts or parsed.is_absolute() or ".." in parts:
        raise ValueError("path outside workspace")
    canonical = workspace_root.joinpath(*parts)
    if is_private_runtime_path(canonical):
        raise ValueError("path outside workspace")
    mount_points = _read_mount_points()
    root_resolved = workspace_root.resolve()
    dir_fd = os.open(workspace_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        root_dev = os.fstat(dir_fd).st_dev
        lexical = root_resolved
        for part in parts[:-1]:
            lexical = lexical / part
            try:
                child_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=dir_fd,
                )
            except OSError as exc:
                raise _map_component_error(exc, dir_fd=dir_fd, name=part) from exc
            try:
                st = os.fstat(child_fd)
            except OSError:
                os.close(child_fd)
                raise
            if st.st_dev != root_dev or lexical in mount_points:
                os.close(child_fd)
                raise ValueError("path outside workspace")
            os.close(dir_fd)
            dir_fd = child_fd
        try:
            fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=dir_fd,
            )
        except OSError as exc:
            raise _map_component_error(exc, dir_fd=dir_fd, name=parts[-1]) from exc
    finally:
        os.close(dir_fd)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise FileNotFoundError("file not found")
        if st.st_size > max_download_bytes:
            raise ValueError("file too large")
        chunks: list[bytes] = []
        remaining = max_download_bytes + 1
        offset = 0
        while remaining > 0:
            chunk = os.pread(fd, min(65536, remaining), offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_download_bytes:
            raise ValueError("file too large")
        return data
    finally:
        os.close(fd)


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
    pending = deque(tree) if isinstance(tree, list) else deque()
    while pending:
        if time.monotonic() - started > plan_audit_timeout_seconds:
            break
        node = pending.popleft()
        if not isinstance(node, dict):
            continue
        children = node.get("children")
        if isinstance(children, list):
            pending.extendleft(reversed(children))
        if node.get("type") != "file":
            continue
        path = node.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        remaining_bytes = plan_audit_max_total_bytes - total_bytes
        if remaining_bytes <= 0:
            break
        read_limit = min(remaining_bytes, 4000)
        result = call_tool(ToolRequest("workspace_read", {"path": path, "max_bytes": read_limit}))
        content = result.data.get("output") if result.ok else None
        if not isinstance(content, str) or not content:
            continue
        raw = content.encode("utf-8")
        if len(raw) > read_limit:
            continue
        next_size = len(raw)
        audit_entries.append(
            {
                "kind": "read_file",
                "path": path,
                "bytes": next_size,
                "preview": content[:240],
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
