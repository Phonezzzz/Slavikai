"""SEC-04: UI session file download must not leak through TOCTOU swaps.

The handler used to validate the pathname (resolve + exists + stat)
and then read it with ``Path.read_bytes()``. An external writer swapping
a path component for a symlink — or growing the file — between the
check and the read leaked content from outside the workspace or above
the 5 MB cap. The fix reads through a pinned fd
(``_read_workspace_file_bytes``): openat + O_NOFOLLOW per component,
fstat on the open fd, cap enforced on the bytes actually read.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
import threading
from pathlib import Path

import pytest

import server.http_api as http_api
from server.http.handlers.sessions import handle_ui_session_file_download

SAFE = b"SAFE-CONTENT"
LEAKED = b"LEAKED-OUTSIDE-CONTENT"
MAX_DOWNLOAD_BYTES = 5_000_000


class _StubHub:
    def __init__(self, files: list[str]) -> None:
        self._files = files

    async def get_session_files(self, session_id: str) -> list[str]:
        return self._files

    async def get_session_access(self, session_id: str, principal_id: str) -> str:
        return "owned"


class _FakeRequest:
    def __init__(self, files: list[str], path: str) -> None:
        self.app = {"ui_hub": _StubHub(files)}
        self.match_info = {"session_id": "s1"}
        self.query = {"path": path}

    def get(self, key: str, default=None):
        if key == "principal_id":
            return "owner-1"
        return default


def _download(files: list[str], path: str):
    return asyncio.run(handle_ui_session_file_download(_FakeRequest(files, path)))


@pytest.fixture
def download_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(LEAKED)
    big_outside = tmp_path / "big_outside.bin"
    big_outside.write_bytes(b"B" * 6_000_000)
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_bytes(LEAKED)
    monkeypatch.setattr(http_api, "WORKSPACE_ROOT", ws)
    return {
        "ws": ws,
        "outside": outside,
        "big_outside": big_outside,
        "outside_dir": outside_dir,
    }


def _swap_file_for_symlink(target: Path, link_target: Path) -> None:
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    try:
        target.symlink_to(link_target)
    except FileExistsError:
        pass


def _write_atomically(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` so a concurrent reader never sees
    a torn (empty, half-written) file: complete content in a temp file,
    then one atomic rename."""
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def _restore_file(target: Path, data: bytes) -> None:
    """Restore ``target`` to a real file with ``data`` (a leftover
    symlink from a swapper is unlinked first, not followed)."""
    if target.is_symlink():
        target.unlink()
    target.write_bytes(data)


def test_live_race_no_leak(download_env) -> None:
    """500 handler calls while a thread swaps the file for a symlink.

    Red on pre-fix code: the handler leaked on the 1st attempt.
    """
    ws = download_env["ws"]
    outside = download_env["outside"]
    secret = ws / "secret.txt"
    secret.write_bytes(SAFE)

    stop = threading.Event()

    def swapper() -> None:
        while not stop.is_set():
            _swap_file_for_symlink(secret, outside)
            try:
                secret.unlink()
            except FileNotFoundError:
                pass
            _write_atomically(secret, SAFE)

    thread = threading.Thread(target=swapper, daemon=True)
    thread.start()
    try:
        for _ in range(500):
            resp = _download(["secret.txt"], "secret.txt")
            if resp.status == 200:
                assert bytes(resp.body) == SAFE
            else:
                assert resp.status in (400, 404)
    finally:
        stop.set()
        thread.join(timeout=5)
        _restore_file(secret, SAFE)


def test_deterministic_swap_before_read_returns_original(
    download_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swap rigged to fire inside the check->read window.

    Both ``Path.read_bytes`` (the pre-fix read path) and ``os.pread``
    (the fixed read path) are patched to swap the file for a symlink
    to outside content before delegating. Pre-fix code returns the
    leaked content (red); fixed code reads from the pinned fd and
    returns the original content (green).
    """
    ws = download_env["ws"]
    outside = download_env["outside"]
    secret = ws / "secret.txt"
    secret.write_bytes(SAFE)

    orig_read_bytes = pathlib.Path.read_bytes
    orig_pread = os.pread

    def swap() -> None:
        _swap_file_for_symlink(secret, outside)

    def patched_read_bytes(self):
        if self == secret:
            swap()
        return orig_read_bytes(self)

    def patched_pread(fd, length, offset=0):
        swap()
        return orig_pread(fd, length, offset)

    monkeypatch.setattr(pathlib.Path, "read_bytes", patched_read_bytes)
    monkeypatch.setattr(os, "pread", patched_pread)
    try:
        resp = _download(["secret.txt"], "secret.txt")
    finally:
        _restore_file(secret, SAFE)
    assert resp.status == 200
    assert bytes(resp.body) == SAFE


def test_deterministic_growth_before_read_returns_capped_content(
    download_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same rig as above, but the swap grows the file past the cap.

    Pre-fix code stat'ed the small file and then read the swapped-in
    6 MB file (red); fixed code reads from the pinned fd (green).
    """
    ws = download_env["ws"]
    big_outside = download_env["big_outside"]
    secret = ws / "secret.txt"
    secret.write_bytes(SAFE)

    orig_read_bytes = pathlib.Path.read_bytes
    orig_pread = os.pread

    def swap() -> None:
        _swap_file_for_symlink(secret, big_outside)

    def patched_read_bytes(self):
        if self == secret:
            swap()
        return orig_read_bytes(self)

    def patched_pread(fd, length, offset=0):
        swap()
        return orig_pread(fd, length, offset)

    monkeypatch.setattr(pathlib.Path, "read_bytes", patched_read_bytes)
    monkeypatch.setattr(os, "pread", patched_pread)
    try:
        resp = _download(["secret.txt"], "secret.txt")
    finally:
        _restore_file(secret, SAFE)
    assert resp.status == 200
    assert bytes(resp.body) == SAFE


def test_deterministic_parent_swap_before_read_returns_original(
    download_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same rig, but the swap replaces the parent dir with a symlink.

    Pre-fix code resolved the pathname first and read through the
    swapped parent (red); fixed code holds the parent dir fd (green).
    """
    ws = download_env["ws"]
    sub = ws / "sub"
    sub.mkdir()
    secret = sub / "secret.txt"
    secret.write_bytes(SAFE)
    outside_dir = download_env["outside_dir"]

    orig_read_bytes = pathlib.Path.read_bytes
    orig_pread = os.pread

    def swap() -> None:
        shutil.rmtree(sub, ignore_errors=True)
        try:
            sub.unlink()
        except FileNotFoundError:
            pass
        sub.symlink_to(outside_dir, target_is_directory=True)

    def patched_read_bytes(self):
        if self == secret:
            swap()
        return orig_read_bytes(self)

    def patched_pread(fd, length, offset=0):
        swap()
        return orig_pread(fd, length, offset)

    monkeypatch.setattr(pathlib.Path, "read_bytes", patched_read_bytes)
    monkeypatch.setattr(os, "pread", patched_pread)
    try:
        resp = _download(["sub/secret.txt"], "sub/secret.txt")
    finally:
        if sub.is_symlink():
            sub.unlink()
        sub.mkdir(exist_ok=True)
        secret.write_bytes(SAFE)
    assert resp.status == 200
    assert bytes(resp.body) == SAFE


def test_size_race_never_exceeds_cap(download_env) -> None:
    """1500 handler calls while a thread swaps in a 6 MB outside file.

    Probabilistic: the leak needs the swap to land in the tiny
    stat->read window, so a single pre-fix run may pass it — it caught
    the leak in some pre-fix runs (PoC: attempt 224). The deterministic
    growth test above is the reliable proof for the size cap.
    """
    ws = download_env["ws"]
    big_outside = download_env["big_outside"]
    secret = ws / "secret.txt"
    secret.write_bytes(SAFE)

    stop = threading.Event()

    def swapper() -> None:
        while not stop.is_set():
            _swap_file_for_symlink(secret, big_outside)
            try:
                secret.unlink()
            except FileNotFoundError:
                pass
            _write_atomically(secret, SAFE)

    thread = threading.Thread(target=swapper, daemon=True)
    thread.start()
    try:
        for _ in range(1500):
            resp = _download(["secret.txt"], "secret.txt")
            if resp.status == 200:
                body = bytes(resp.body)
                assert body == SAFE
                assert len(body) <= MAX_DOWNLOAD_BYTES
            else:
                assert resp.status in (400, 404)
    finally:
        stop.set()
        thread.join(timeout=5)
        _restore_file(secret, SAFE)


def test_direct_symlink_refused(download_env) -> None:
    """Guard: a directly requested symlink is refused (was already so)."""
    ws = download_env["ws"]
    link = ws / "link.txt"
    link.symlink_to(download_env["outside"])
    resp = _download(["link.txt"], "link.txt")
    assert resp.status == 400


def test_parent_dir_swap_no_leak(download_env) -> None:
    """600 handler calls while a thread swaps the parent dir for a symlink.

    Probabilistic like the size race (the swap must land inside the
    handler's resolve+read); it caught the leak in some pre-fix runs.
    The deterministic parent-swap test above is the reliable proof.
    """
    ws = download_env["ws"]
    sub = ws / "sub"
    sub.mkdir()
    secret = sub / "secret.txt"
    secret.write_bytes(SAFE)
    outside_dir = download_env["outside_dir"]

    stop = threading.Event()

    def swapper() -> None:
        while not stop.is_set():
            shutil.rmtree(sub, ignore_errors=True)
            try:
                sub.symlink_to(outside_dir, target_is_directory=True)
            except FileExistsError:
                pass
            try:
                sub.unlink()
            except FileNotFoundError:
                pass
            sub.mkdir(exist_ok=True)
            _write_atomically(secret, SAFE)

    thread = threading.Thread(target=swapper, daemon=True)
    thread.start()
    try:
        for _ in range(600):
            resp = _download(["sub/secret.txt"], "sub/secret.txt")
            if resp.status == 200:
                assert bytes(resp.body) == SAFE
            else:
                assert resp.status in (400, 404)
    finally:
        stop.set()
        thread.join(timeout=5)
        if sub.is_symlink():
            sub.unlink()
        sub.mkdir(exist_ok=True)
        secret.write_bytes(SAFE)


def test_dotdot_and_absolute_paths_rejected(download_env) -> None:
    """Helper contract: parent escapes and absolute paths are refused."""
    from server.http.common.workspace_runtime import _read_workspace_file_bytes

    ws = download_env["ws"]
    (ws / "secret.txt").write_bytes(SAFE)
    for bad in ("../outside.txt", "/etc/passwd", "sub/../../outside.txt"):
        with pytest.raises(ValueError, match="path outside workspace"):
            _read_workspace_file_bytes(bad, workspace_root=ws, max_download_bytes=100)
    assert (
        _read_workspace_file_bytes("secret.txt", workspace_root=ws, max_download_bytes=100) == SAFE
    )
