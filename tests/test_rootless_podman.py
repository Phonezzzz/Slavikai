from __future__ import annotations

import os

import pytest

from core.managed_workspace import ManagedWorkspace, WorkspaceLimits
from core.rootless_podman import RootlessPodmanProfile
from shared.workspace_binding import WorkspaceUnavailable


def test_rootful_identity_rejected_before_engine_command(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("os.geteuid", lambda: 0)

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("engine must not be contacted")

    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(WorkspaceUnavailable, match="rootless"):
        RootlessPodmanProfile.qualify(tmp_path)
    assert calls == []


def test_stale_environment_binding_cannot_create_container(tmp_path, monkeypatch):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "script.py").write_text("print('synthetic')")
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    snapshot = workspace.capture(
        binding, limits=WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8)
    )
    stage = workspace.stage(snapshot)
    # Ни identity, ни private stage не заменяют environment qualification binding.
    profile = RootlessPodmanProfile(
        tmp_path, os.geteuid(), os.getegid(), "profile-revision", (), ()
    )
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("stale binding must refuse before any engine call")

    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(WorkspaceUnavailable, match="environment"):
        profile.create_python_container(stage, "script.py")
    assert calls == []


def test_plain_host_stage_is_not_a_bounded_execution_backing_store(tmp_path, monkeypatch):
    from dataclasses import replace

    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    owner = ManagedWorkspace.create(parent, principal_id="alice")
    (owner.root / "script.py").write_text("print('synthetic')")
    binding = replace(
        owner.bind(principal_id="alice", session_id="chat"), environment_revision="profile"
    )
    snapshot = owner.capture(binding, limits=WorkspaceLimits(1024, 4096, 20, 8))
    stage = owner.stage(snapshot)
    profile = RootlessPodmanProfile(tmp_path, os.geteuid(), os.getegid(), "profile", (), ())

    def forbidden(*args, **kwargs):
        pytest.fail("plain host staging must be rejected before contacting engine")

    monkeypatch.setattr(RootlessPodmanProfile, "check_current", lambda self: None)
    monkeypatch.setattr("subprocess.run", forbidden)
    with pytest.raises(WorkspaceUnavailable, match="bounded"):
        profile.create_python_container(stage, "script.py")
