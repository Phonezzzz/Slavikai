from __future__ import annotations

import pytest


def test_staging_limits_have_finite_explicit_individual_and_aggregate_bounds():
    from core.rootless_staging import StagingLimits

    limits = StagingLimits(slot_bytes=2097152, slot_inodes=64, slots=2)
    assert limits.total_bytes == 4194304
    assert limits.total_inodes == 128
    for kwargs in (
        {"slot_bytes": 0, "slot_inodes": 64, "slots": 2},
        {"slot_bytes": 4097, "slot_inodes": 64, "slots": 2},
        {"slot_bytes": 4096, "slot_inodes": True, "slots": 2},
        {"slot_bytes": 4096, "slot_inodes": 64, "slots": 0},
    ):
        with pytest.raises(ValueError):
            StagingLimits(**kwargs)


def test_mount_helper_refuses_initial_host_identity_before_mount_syscalls(tmp_path, monkeypatch):
    from core import rootless_stage_helper

    monkeypatch.setattr("os.geteuid", lambda: 1000)

    def forbidden(*args, **kwargs):
        pytest.fail("host namespace must be rejected before any mount syscall")

    monkeypatch.setattr(rootless_stage_helper.ctypes, "CDLL", forbidden)
    with pytest.raises(ValueError, match="rootless namespaces"):
        rootless_stage_helper.run(
            tmp_path, 4096, 32, 2, 1000, 1000, "mnt:[host]", "user:[host]", "n"
        )


@pytest.mark.parametrize("delivery", ["closed", "broken"])
def test_pool_close_requires_confirmed_cleanup_command(tmp_path, delivery):
    import io
    from types import SimpleNamespace
    from unittest.mock import Mock

    from core.rootless_staging import RootlessStagePool, StagingLimits
    from shared.workspace_binding import WorkspaceUnavailable

    pool = RootlessStagePool(SimpleNamespace(state_root=tmp_path), StagingLimits(4096, 32, 1))
    stream = io.BytesIO()
    if delivery == "closed":
        stream.close()
    else:
        stream = Mock(closed=False, write=Mock(side_effect=BrokenPipeError))
    process = Mock(stdin=stream, stdout=io.BytesIO(), stderr=io.BytesIO(), returncode=0)
    pool._process = process
    with pytest.raises(WorkspaceUnavailable, match="cleanup unavailable"):
        pool.close()


def test_pool_close_confirmed_cleanup_is_idempotent(tmp_path):
    import io
    from types import SimpleNamespace
    from unittest.mock import Mock

    from core.rootless_staging import RootlessStagePool, StagingLimits

    pool = RootlessStagePool(SimpleNamespace(state_root=tmp_path), StagingLimits(4096, 32, 1))
    command = Mock(closed=False)
    process = Mock(stdin=command, stdout=io.BytesIO(), stderr=io.BytesIO(), returncode=0)
    pool._process = process
    pool.close()
    pool.close()
    command.write.assert_called_once_with(b"close\n")
    process.wait.assert_called_once_with(timeout=15)


def test_pool_close_retains_live_allocation_without_requesting_unmount(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from core.rootless_staging import RootlessStagePool, StagingLimits

    pool = RootlessStagePool(SimpleNamespace(state_root=tmp_path), StagingLimits(4096, 32, 1))
    process = Mock()
    pool._process = process
    pool._allocations[tmp_path / "retained-operation"] = None
    pool.close()
    process.stdin.write.assert_not_called()
    process.wait.assert_not_called()
