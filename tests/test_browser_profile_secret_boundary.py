"""E02-T07 — profile secret boundary.

Session cookies in a browser profile are credentials-equivalent state: no
workspace tool and no workspace executor may read the browser profile path,
a workspace root may not be pointed at it, and a symlink from a workspace
into the profile root is blocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import shared.sandbox as sandbox_module
from core.managed_workspace import ManagedWorkspace, WorkspaceLimits
from shared.models import ToolRequest
from shared.sandbox import is_private_runtime_path
from shared.workspace_binding import WorkspaceUnavailable
from tools.workspace_tools import (
    ListFilesTool,
    ReadFileTool,
    set_workspace_root,
    workspace_root_context,
)


@pytest.fixture()
def profile_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    profiles = tmp_path / "browser-profiles"
    leases = tmp_path / "browser-profiles.leases"
    profile_dir = profiles / "local" / "default"
    profile_dir.mkdir(parents=True)
    (profile_dir / "Cookies").write_text("synthetic-cookie-canary", encoding="utf-8")
    leases.mkdir(parents=True)
    (leases / "local").mkdir()
    (leases / "local" / "default.lease").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sandbox_module, "BROWSER_PROFILE_STORAGE", profiles)
    monkeypatch.setattr(sandbox_module, "BROWSER_PROFILE_LEASE_STORAGE", leases)
    return profiles


def _request(name: str, args: dict[str, object]) -> ToolRequest:
    return ToolRequest(name=name, args=args)


def test_profile_and_lease_paths_are_private(profile_storage: Path) -> None:
    assert is_private_runtime_path(profile_storage)
    assert is_private_runtime_path(profile_storage / "local" / "default" / "Cookies")
    assert is_private_runtime_path(
        profile_storage.parent / "browser-profiles.leases" / "local" / "default.lease"
    )
    assert not is_private_runtime_path(profile_storage.parent / "ordinary.txt")


def test_workspace_root_cannot_be_profile_dir(profile_storage: Path) -> None:
    with pytest.raises(ValueError, match="закрытому состоянию"):
        set_workspace_root(profile_storage / "local" / "default")


def test_workspace_root_cannot_be_profile_root(profile_storage: Path) -> None:
    with pytest.raises(ValueError, match="закрытому состоянию"):
        set_workspace_root(profile_storage)


def test_read_profile_file_under_parent_root_is_refused(
    profile_storage: Path,
) -> None:
    parent = profile_storage.parent
    with workspace_root_context(parent):
        result = ReadFileTool().handle(
            _request(
                "workspace_read",
                {"path": "browser-profiles/local/default/Cookies"},
            )
        )
    assert not result.ok
    assert "synthetic-cookie-canary" not in str(result.data)


def test_list_under_parent_root_hides_profile_dir(profile_storage: Path) -> None:
    parent = profile_storage.parent
    (parent / "visible.txt").write_text("ok", encoding="utf-8")
    with workspace_root_context(parent):
        result = ListFilesTool().handle(_request("workspace_list", {"recursive": True}))
    assert result.ok
    paths = str(result.data)
    assert "visible.txt" in paths
    assert "browser-profiles" not in paths
    assert "Cookies" not in paths


def test_symlink_from_workspace_to_profile_is_refused(
    profile_storage: Path, tmp_path: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "link").symlink_to(profile_storage / "local" / "default")
    with workspace_root_context(workspace):
        result = ReadFileTool().handle(_request("workspace_read", {"path": "link/Cookies"}))
        listing = ListFilesTool().handle(_request("workspace_list", {"recursive": True}))
    assert not result.ok
    assert "synthetic-cookie-canary" not in str(result.data)
    assert listing.ok
    assert "Cookies" not in str(listing.data)


def test_managed_capture_refuses_symlink_to_profile(profile_storage: Path, tmp_path: Path) -> None:
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "cookies-link").symlink_to(profile_storage / "local" / "default" / "Cookies")
    with pytest.raises(WorkspaceUnavailable, match="inode"):
        workspace.capture(
            workspace.bind(principal_id="alice", session_id="chat-a"),
            limits=WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8),
        )


def test_ordinary_workspace_access_still_works(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello", encoding="utf-8")
    with workspace_root_context(workspace):
        result = ReadFileTool().handle(_request("workspace_read", {"path": "note.txt"}))
    assert result.ok
    assert result.data["output"] == "hello"
