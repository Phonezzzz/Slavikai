from __future__ import annotations

import os

from core.managed_workspace import ManagedWorkspace, WorkspaceLimits
from core.rootless_podman import RootlessPodmanProfile
from core.workspace_python_execution import RootlessPythonExecutor


def test_stale_workspace_refuses_before_any_engine_command(tmp_path, monkeypatch):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    owner = ManagedWorkspace.create(parent, principal_id="alice")
    (owner.root / "script.py").write_text("print('original')")
    binding = owner.bind(principal_id="alice", session_id="chat-a")
    with owner.mutation(binding):
        (owner.root / "script.py").write_text("print('replacement')")
    profile = RootlessPodmanProfile(tmp_path, os.geteuid(), os.getegid(), "not-a-proof", (), ())
    executor = RootlessPythonExecutor(
        profile, owner, WorkspaceLimits(1024, 4096, 20, 8), max_output_bytes=1024
    )
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("stale workspace must not contact engine")

    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    result = executor.execute_python(binding, "script.py", timeout=3)
    assert result.reason_code == "workspace_conflict"
    assert result.execution_started is False
    assert result.exit_code is None
    assert result.stdout == result.stderr == b""
    assert calls == []


def test_unexpected_engine_inspection_fault_cleans_own_container_without_publication(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from core.rootless_podman import PreparedPythonContainer
    from core.rootless_staging import BoundedStage, RootlessStagePool, StagingLimits

    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    owner = ManagedWorkspace.create(parent, principal_id="alice")
    (owner.root / "script.py").write_text("print('original')")
    binding = replace(
        owner.bind(principal_id="alice", session_id="chat-a"),
        environment_revision="synthetic-bounded-environment",
    )
    profile = RootlessPodmanProfile(tmp_path, os.geteuid(), os.getegid(), "revision", (), ())
    pool = RootlessStagePool(profile, StagingLimits(4096, 32, 2))
    # Только fault propagation/cleanup: OS staging и engine здесь заменены boundary stubs.
    monkeypatch.setattr(
        RootlessStagePool,
        "execution_revision",
        property(lambda self: "synthetic-bounded-environment"),
    )

    def prepare(self, workspace, snapshot):
        stage = workspace.stage(snapshot)
        info = stage.root.stat()
        return BoundedStage(stage, self, stage.root, stage.root, (info.st_dev, info.st_ino))

    monkeypatch.setattr(RootlessStagePool, "prepare", prepare)
    executor = RootlessPythonExecutor(
        profile, owner, WorkspaceLimits(1024, 4096, 20, 8), max_output_bytes=1024, staging_pool=pool
    )
    prepared_containers = []
    removed = []

    def create(self, stage, script):
        prepared = PreparedPythonContainer(
            "synthetic-cid", "synthetic-own", stage, script, "revision"
        )
        prepared_containers.append(prepared)
        return prepared

    def inspect_fault(self, prepared):
        raise RuntimeError("synthetic control decoding fault")

    def remove(self, prepared):
        removed.append(prepared.container_id)
        return True

    monkeypatch.setattr(RootlessPodmanProfile, "create_python_container", create)
    monkeypatch.setattr(RootlessPodmanProfile, "verify_created", inspect_fault)
    monkeypatch.setattr(RootlessPodmanProfile, "remove_owned", remove)
    result = executor.execute_python(binding, "script.py", timeout=3)
    assert result.state == "unavailable"
    assert result.execution_started is False
    assert result.exit_code is None
    assert result.publication == "retained"
    assert result.reason_code == "workspace_transport_unknown"
    assert removed == ["synthetic-cid"]
    assert (owner.root / "script.py").read_text() == "print('original')"
    import pytest

    from shared.workspace_binding import WorkspaceUnavailable

    with pytest.raises(WorkspaceUnavailable):
        with prepared_containers[0].stage.publication():
            pytest.fail("unknown stage must not publish")
