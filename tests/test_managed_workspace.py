from __future__ import annotations

import os

import pytest

from core.managed_workspace import (
    ManagedWorkspace,
    ManagedWorkspacePool,
    WorkspaceAccess,
    WorkspaceLimits,
)
from shared.workspace_binding import WorkspaceConflict, WorkspaceUnavailable


def test_capture_rejects_private_inode_alias(tmp_path):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    private = tmp_path / "synthetic-private.txt"
    private.write_bytes(b"synthetic-private-canary")
    os.link(private, workspace.root / "alias.txt")

    with pytest.raises(WorkspaceUnavailable, match="inode"):
        workspace.capture(
            workspace.bind(principal_id="alice", session_id="chat-a"),
            limits=WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8),
        )
    assert private.read_bytes() == b"synthetic-private-canary"


def test_other_session_gateway_write_invalidates_process_snapshot(tmp_path):
    from core.approval_policy import ApprovalContext
    from core.tool_gateway import ToolGateway
    from shared.models import ToolRequest
    from tools.tool_logger import ToolCallLogger
    from tools.tool_registry import ToolRegistry
    from tools.workspace_tools import WriteFileTool, workspace_root_context

    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    binding_a = workspace.bind(principal_id="alice", session_id="chat-a")
    limits = WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8)
    captured = workspace.capture(binding_a, limits=limits)
    stage = workspace.stage(captured)
    (stage.root / "value.txt").write_text("A-output")
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "tools.log"))
    registry.register("workspace_write", WriteFileTool(), capability="write")
    registry.set_execution_policy(mode="auto")
    gateway_b = ToolGateway(
        registry=registry,
        approval_context=ApprovalContext(False, "chat-b", set()),
        workspace_access=WorkspaceAccess(workspace, "alice", "chat-b"),
    )
    with workspace_root_context(workspace.root):
        result = gateway_b.call(
            ToolRequest("workspace_write", {"path": "value.txt", "content": "B"})
        )
    assert result.ok
    assert (workspace.root / "value.txt").read_text() == "B"
    with pytest.raises(WorkspaceConflict, match="revision"):
        workspace.capture(captured.binding, limits=limits)
    with pytest.raises(WorkspaceConflict, match="revision"):
        workspace.publish(captured, stage, limits=limits)
    assert (workspace.root / "value.txt").read_text() == "B"
    assert (stage.root / "value.txt").read_text() == "A-output"


def test_once_approval_cannot_follow_workspace_revision_change(tmp_path):
    from core.approval_policy import ApprovalContext, ApprovalRequired
    from core.tool_gateway import ToolGateway
    from shared.models import ToolRequest
    from tools.tool_logger import ToolCallLogger
    from tools.tool_registry import ToolRegistry
    from tools.workspace_tools import RunCodeTool, workspace_root_context

    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "script.py").write_text("print('synthetic')")
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "tools.log"))
    registry.register("workspace_run", RunCodeTool(), capability="exec", risk_classes=["execute"])
    registry.set_execution_policy(mode="auto")
    context = ApprovalContext(True, "chat-a", set())
    gateway = ToolGateway(
        registry=registry,
        approval_context=context,
        workspace_access=WorkspaceAccess(workspace, "alice", "chat-a"),
    )
    request = ToolRequest("workspace_run", {"path": "script.py"})
    with workspace_root_context(workspace.root):
        with pytest.raises(ApprovalRequired) as pending:
            gateway.call(request)
        with workspace.mutation(workspace.bind(principal_id="alice", session_id="chat-b")):
            (workspace.root / "script.py").write_text("print('replacement')")
        result = gateway.call_approved_once(request, pending.value.request)
    assert not result.ok
    assert result.meta["reason_code"] == "workspace_conflict"
    assert result.meta["execution_started"] is False
    assert context.approved_categories == set()


def test_private_stage_publishes_bytes_and_rejects_stale_second_publication(tmp_path):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "source.py").write_bytes(b"original\x00\xff")
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    limits = WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8)
    snapshot = workspace.capture(binding, limits=limits)
    stage = workspace.stage(snapshot)
    assert stage.root != workspace.root
    assert (stage.root / "source.py").stat().st_ino != (workspace.root / "source.py").stat().st_ino
    (stage.root / "source.py").write_bytes(b"updated\x00\xfe")
    (stage.root / "created.txt").write_bytes(b"new bytes")
    assert (workspace.root / "source.py").read_bytes() == b"original\x00\xff"
    published = workspace.publish(snapshot, stage, limits=limits)
    assert published.revision != binding.revision
    assert (workspace.root / "source.py").read_bytes() == b"updated\x00\xfe"
    assert (workspace.root / "created.txt").read_bytes() == b"new bytes"
    with pytest.raises(WorkspaceConflict, match="revision"):
        workspace.publish(snapshot, stage, limits=limits)
    workspace.discard_retired(snapshot, stage)
    assert not stage.root.exists()
    assert (workspace.root / "source.py").read_bytes() == b"updated\x00\xfe"


def test_read_only_registry_denial_does_not_mutate_workspace_revision(tmp_path):
    from core.approval_policy import ApprovalContext
    from core.tool_gateway import ToolGateway
    from shared.models import ToolRequest
    from tools.tool_logger import ToolCallLogger
    from tools.tool_registry import ToolRegistry
    from tools.workspace_tools import WriteFileTool, workspace_root_context

    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "tools.log"))
    registry.register("workspace_write", WriteFileTool(), capability="write")
    registry.set_execution_policy(mode="plan")
    gateway = ToolGateway(
        registry=registry,
        approval_context=ApprovalContext(False, "chat-a", set()),
        workspace_access=WorkspaceAccess(workspace, "alice", "chat-a"),
    )
    with workspace_root_context(workspace.root):
        result = gateway.call(ToolRequest("workspace_write", {"path": "value.txt", "content": "X"}))
    assert not result.ok
    assert "PLAN_READ_ONLY_BLOCK" in result.error
    assert workspace.bind(principal_id="alice", session_id="chat-a") == binding
    assert not (workspace.root / "value.txt").exists()


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        (
            "workspace_write",
            {"path": "source.txt", "content": "after\n"},
            {"source.txt": b"after\n"},
        ),
        (
            "workspace_create",
            {"path": "new.txt", "content": "new\n"},
            {"source.txt": b"before\n", "new.txt": b"new\n"},
        ),
        (
            "workspace_rename",
            {"old_path": "source.txt", "new_path": "renamed.txt"},
            {"renamed.txt": b"before\n"},
        ),
        (
            "workspace_move",
            {"from_path": "source.txt", "to_path": "sub/moved.txt"},
            {"sub/moved.txt": b"before\n"},
        ),
        ("workspace_delete", {"path": "source.txt"}, {}),
        (
            "workspace_patch",
            {"path": "source.txt", "patch": "@@ -1,1 +1,1 @@\n-before\n+after\n"},
            {"source.txt": b"after\n"},
        ),
    ],
)
def test_all_native_writers_share_workspace_revision(tmp_path, name, args, expected):
    from core.approval_policy import ApprovalContext
    from core.tool_gateway import ToolGateway
    from shared.models import ToolRequest
    from tools.tool_logger import ToolCallLogger
    from tools.tool_registry import ToolRegistry
    from tools.workspace_tools import (
        ApplyPatchTool,
        CreateFileTool,
        DeleteFileTool,
        MoveFileTool,
        RenameFileTool,
        WriteFileTool,
        workspace_root_context,
    )

    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "source.txt").write_bytes(b"before\n")
    original = workspace.bind(principal_id="alice", session_id="chat-a")
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "tools.log"))
    handlers = {
        "workspace_write": WriteFileTool(),
        "workspace_create": CreateFileTool(),
        "workspace_rename": RenameFileTool(),
        "workspace_move": MoveFileTool(),
        "workspace_delete": DeleteFileTool(),
        "workspace_patch": ApplyPatchTool(),
    }
    registry.register(name, handlers[name], capability="write")
    registry.set_execution_policy(mode="auto")
    gateway = ToolGateway(
        registry=registry,
        approval_context=ApprovalContext(False, "chat-b", set()),
        workspace_access=WorkspaceAccess(workspace, "alice", "chat-b"),
    )
    with workspace_root_context(workspace.root):
        result = gateway.call(ToolRequest(name, args))
    assert result.ok, result.error
    limits = WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8)
    with pytest.raises(WorkspaceConflict):
        workspace.capture(original, limits=limits)
    current = workspace.capture(
        workspace.bind(principal_id="alice", session_id="chat-b"), limits=limits
    )
    assert {file.path: file.content for file in current.files} == expected


@pytest.mark.parametrize("alias", ["symlink", "hardlink", "fifo"])
def test_unsafe_stage_output_never_replaces_source(tmp_path, alias):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "source.txt").write_bytes(b"original")
    private = tmp_path / "private.txt"
    private.write_bytes(b"synthetic-private-canary")
    limits = WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8)
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    snapshot = workspace.capture(binding, limits=limits)
    stage = workspace.stage(snapshot)
    target = stage.root / "bad.txt"
    if alias == "symlink":
        target.symlink_to(private)
    elif alias == "hardlink":
        os.link(private, target)
    else:
        os.mkfifo(target)
    with pytest.raises(WorkspaceUnavailable, match="inode"):
        workspace.publish(snapshot, stage, limits=limits)
    assert (workspace.root / "source.txt").read_bytes() == b"original"
    assert not (workspace.root / "bad.txt").exists()
    assert private.read_bytes() == b"synthetic-private-canary"
    assert workspace.bind(principal_id="alice", session_id="chat-a") == binding


def test_foreign_principal_cannot_bind_or_capture_owned_workspace(tmp_path):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "source.txt").write_bytes(b"alice-only")
    with pytest.raises(WorkspaceUnavailable, match="owner"):
        workspace.bind(principal_id="bob", session_id="chat-b")


def test_changed_source_outside_owner_revision_is_not_overwritten(tmp_path):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "source.txt").write_bytes(b"original")
    limits = WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8)
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    snapshot = workspace.capture(binding, limits=limits)
    stage = workspace.stage(snapshot)
    (stage.root / "source.txt").write_bytes(b"worker-result")
    (workspace.root / "source.txt").write_bytes(b"unexpected-host-edit")
    with pytest.raises(WorkspaceConflict, match="source changed"):
        workspace.publish(snapshot, stage, limits=limits)
    assert (workspace.root / "source.txt").read_bytes() == b"unexpected-host-edit"
    assert (stage.root / "source.txt").read_bytes() == b"worker-result"


def test_exchange_failure_has_no_copy_fallback(tmp_path, monkeypatch):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "source.txt").write_bytes(b"original")
    limits = WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8)
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    snapshot = workspace.capture(binding, limits=limits)
    stage = workspace.stage(snapshot)
    (stage.root / "source.txt").write_bytes(b"worker-result")

    def unavailable(*args):
        raise OSError("synthetic unsupported filesystem")

    monkeypatch.setattr("core.managed_workspace._exchange", unavailable)
    with pytest.raises(OSError, match="unsupported filesystem"):
        workspace.publish(snapshot, stage, limits=limits)
    assert (workspace.root / "source.txt").read_bytes() == b"original"
    assert (stage.root / "source.txt").read_bytes() == b"worker-result"
    assert workspace.bind(principal_id="alice", session_id="chat-a") == binding


def test_running_or_unknown_process_stage_cannot_be_published(tmp_path):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "source.txt").write_bytes(b"original")
    limits = WorkspaceLimits(file_bytes=1024, total_bytes=4096, entries=20, depth=8)
    snapshot = workspace.capture(
        workspace.bind(principal_id="alice", session_id="chat-a"), limits=limits
    )
    stage = workspace.stage(snapshot)
    stage.begin_process()
    (stage.root / "source.txt").write_bytes(b"possibly-still-changing")
    with pytest.raises(WorkspaceUnavailable, match="settled"):
        workspace.publish(snapshot, stage, limits=limits)
    stage.mark_unknown()
    with pytest.raises(WorkspaceUnavailable, match="settled"):
        workspace.publish(snapshot, stage, limits=limits)
    with pytest.raises(WorkspaceUnavailable):
        stage.mark_settled()
    assert (workspace.root / "source.txt").read_bytes() == b"original"


def test_pool_preserves_principal_workspace_and_fences_foreign_or_overlapping_roots(tmp_path):
    root = tmp_path / "managed"
    pool = ManagedWorkspacePool(root)
    alice = pool.for_principal("alice")
    (alice.root / "source.txt").write_bytes(b"alice-existing-work")
    assert pool.for_principal("alice") is alice
    bob = pool.for_principal("bob")
    assert alice.root != bob.root
    assert pool.resolve_root("alice", alice.root) == alice.root
    with pytest.raises(WorkspaceUnavailable):
        pool.resolve_root("bob", alice.root)
    with pytest.raises(WorkspaceUnavailable):
        pool.resolve_root("alice", tmp_path)
    unrelated = tmp_path / "unmanaged"
    unrelated.mkdir()
    assert pool.resolve_root("alice", unrelated) == unrelated
    fresh_pool = ManagedWorkspacePool(root)
    with pytest.raises(WorkspaceUnavailable, match="another runtime owner"):
        fresh_pool.for_principal("alice")
    old_identity = alice.bind(principal_id="alice", session_id="old").workspace_id
    old_root = alice.root
    pool.close()
    with pytest.raises(WorkspaceUnavailable, match="another runtime owner"):
        fresh_pool.for_principal("alice")
    del alice
    rebound = fresh_pool.for_principal("alice")
    assert rebound.root == old_root
    assert (rebound.root / "source.txt").read_bytes() == b"alice-existing-work"
    assert rebound.bind(principal_id="alice", session_id="new").workspace_id != old_identity


def test_session_root_policy_keeps_principal_boundary_on_set_and_get(tmp_path):
    import asyncio

    async def scenario():
        from pathlib import Path

        from server.ui_hub import UIHub

        pool = ManagedWorkspacePool(tmp_path / "managed")

        def policy(principal, root):
            return str(pool.resolve_root(principal, None if root is None else Path(root)))

        hub = UIHub(workspace_root_policy=policy)
        alice = await hub.create_session("alice")
        bob = await hub.create_session("bob")
        alice_root = await hub.get_workspace_root(alice)
        bob_root = await hub.get_workspace_root(bob)
        assert alice_root != bob_root
        with pytest.raises(WorkspaceUnavailable):
            await hub.set_workspace_root(bob, alice_root)
        assert await hub.get_workspace_root(bob) == bob_root

    asyncio.run(scenario())


@pytest.mark.parametrize("relative", ["storage", "config", "home", "tmp", "workspaces"])
def test_gateway_blocks_all_private_control_roots_even_without_executor(tmp_path, relative):
    from core.approval_policy import ApprovalContext
    from core.tool_gateway import ToolGateway
    from shared.models import ToolRequest
    from tools.tool_logger import ToolCallLogger
    from tools.tool_registry import ToolRegistry
    from tools.workspace_tools import WriteFileTool, workspace_root_context

    state = tmp_path / "private-control"
    state.mkdir(mode=0o700)
    selected = state / relative
    selected.mkdir(mode=0o700)
    pool = ManagedWorkspacePool(state / "workspaces", protected_root=state)
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "tools.log"))
    registry.register("workspace_write", WriteFileTool(), capability="write")
    registry.set_execution_policy(mode="auto")
    gateway = ToolGateway(
        registry=registry,
        approval_context=ApprovalContext(False, "chat", {"FS_DELETE_OVERWRITE"}),
        workspace_pool=pool,
        workspace_principal_id="alice",
    )
    with workspace_root_context(selected):
        result = gateway.call(
            ToolRequest("workspace_write", {"path": "canary.txt", "content": "must not write"})
        )
    assert not result.ok
    assert result.meta["execution_started"] is False
    assert result.meta["reason_code"] == "workspace_unavailable"
    assert not (selected / "canary.txt").exists()
    assert list((state / "workspaces").glob("*")) == []
    pool.close()


def test_agent_foreign_workspace_binding_rejected_before_database_initialization(
    tmp_path, monkeypatch
):
    import sqlite3

    from core.agent import Agent

    pool = ManagedWorkspacePool(tmp_path / "managed")
    alice = pool.for_principal("alice")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid principal must be rejected before any database connection")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with pytest.raises(ValueError, match="principal mismatch"):
        Agent(
            user_id="bob",
            workspace_pool=pool,
            workspace_access=WorkspaceAccess(alice, "alice", "same-session"),
        )
    pool.close()


def test_revision_conflict_at_actual_writer_guard_requires_explicit_review(tmp_path):
    from core.approval_policy import ApprovalContext
    from core.tool_gateway import ToolGateway
    from shared.models import ToolRequest
    from tools.tool_logger import ToolCallLogger
    from tools.tool_registry import ToolRegistry
    from tools.workspace_tools import WriteFileTool, workspace_root_context

    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    owner = ManagedWorkspace.create(parent, principal_id="alice")
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "tools.log"))
    registry.register("workspace_write", WriteFileTool(), capability="write")
    registry.set_execution_policy(mode="auto")

    def concurrent_write(request):
        binding = owner.bind(principal_id="alice", session_id="other-chat")
        with owner.mutation(binding):
            (owner.root / "value.txt").write_text("other writer")

    gateway = ToolGateway(
        registry=registry,
        approval_context=ApprovalContext(False, "chat", set()),
        workspace_access=WorkspaceAccess(owner, "alice", "chat"),
        pre_call=concurrent_write,
    )
    with workspace_root_context(owner.root):
        result = gateway.call(ToolRequest("workspace_write", {"path": "value.txt", "content": "A"}))
    assert not result.ok
    assert result.meta["reason_code"] == "workspace_conflict"
    assert result.meta["execution_review_required"] is True
    assert result.meta["execution_started"] is False
    assert (owner.root / "value.txt").read_text() == "other writer"


def test_concurrent_writer_and_publisher_never_lose_accepted_write(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    owner = ManagedWorkspace.create(parent, principal_id="alice")
    (owner.root / "value.txt").write_text("original")
    limits = WorkspaceLimits(1024, 4096, 20, 8)
    binding = owner.bind(principal_id="alice", session_id="process-chat")
    snapshot = owner.capture(binding, limits=limits)
    stage = owner.stage(snapshot)
    (stage.root / "value.txt").write_text("process-output")
    writer_inside = threading.Event()
    publisher_ready = threading.Event()

    def writer():
        other = owner.bind(principal_id="alice", session_id="writer-chat")
        with owner.mutation(other):
            (owner.root / "value.txt").write_text("accepted-write")
            writer_inside.set()
            assert publisher_ready.wait(5)

    def publisher():
        assert writer_inside.wait(5)
        publisher_ready.set()
        with pytest.raises(WorkspaceConflict):
            owner.publish(snapshot, stage, limits=limits)

    with ThreadPoolExecutor(max_workers=2) as workers:
        writing = workers.submit(writer)
        publishing = workers.submit(publisher)
        writing.result(timeout=10)
        publishing.result(timeout=10)
    assert (owner.root / "value.txt").read_text() == "accepted-write"
    assert (stage.root / "value.txt").read_text() == "process-output"
