from __future__ import annotations


def test_run_tool_publishes_only_committed_reference_and_bounded_utf8_view(tmp_path):
    from types import SimpleNamespace

    from core.managed_workspace import ManagedWorkspace
    from core.task_run_storage import TaskRunStore
    from core.workspace_python_execution import WorkspacePythonOutcome
    from shared.models import ToolRequest
    from shared.task_run import RunScope, RunState, TransitionAuthority
    from shared.tool_capture import (
        MAX_VIEW_BYTES,
        CaptureBinding,
        ProcessObservation,
        ToolCaptureContext,
        ToolCaptureReceipt,
    )
    from tools.workspace_tools import RunCodeTool, workspace_root_context

    owner = ManagedWorkspace.create(tmp_path, principal_id="alice")
    (owner.root / "emit.py").write_text("print('synthetic')")
    binding = owner.bind(principal_id="alice", session_id="chat")
    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("alice", "chat")
    admitted = store.admit(
        scope, request_key="key", goal="synthetic", mode="auto", request_fingerprint="f" * 64
    )
    run = store.transition(
        scope,
        admitted.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="started",
    )
    context = ToolCaptureContext(CaptureBinding(store, run), "label")
    raw = b"\xff" * MAX_VIEW_BYTES

    def execute(bound, script, *, timeout, capture=None):
        assert bound == binding and script == "emit.py" and capture is context
        operation = store.begin_tool_operation(run, model_call_id="label", request_path=script)
        observation = ProcessObservation(raw, b"", len(raw), 0, True, True, 0, False)
        result_id = store.record_tool_observation(operation, observation)
        return WorkspacePythonOutcome(
            "finished",
            raw,
            b"",
            0,
            True,
            None,
            "published",
            capture_receipt=ToolCaptureReceipt(operation, result_id, observation),
        )

    tool = RunCodeTool(backend=SimpleNamespace(execute_python=execute))
    with workspace_root_context(owner.root):
        result = tool.handle(
            ToolRequest(
                "workspace_run", {"path": "emit.py"}, workspace_binding=binding, capture=context
            )
        )
    assert result.ok
    assert len(result.data["stdout"].encode("utf-8")) <= MAX_VIEW_BYTES
    assert result.data["capture"]["stdout_view_truncated"] is True
    ref = result.data["capture"]
    saved = TaskRunStore(store.path).get_tool_observation(
        scope, run.task_run_id, ref["operation_id"], ref["tool_attempt_id"]
    )
    assert saved.result_id == ref["result_id"] and saved.observation.stdout == raw
    import json

    assert "\\udcff" not in json.dumps(result.data)


def test_capture_fault_propagates_through_registry_and_gateway_without_payload_log(tmp_path):
    import pytest

    from core.tool_gateway import ToolGateway
    from shared.models import ToolRequest
    from shared.tool_capture import ToolCaptureFault
    from tools.tool_logger import ToolCallLogger
    from tools.tool_registry import ToolRegistry

    def fail(request):
        raise ToolCaptureFault("workspace_capture_store_failed")

    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "calls.log"))
    registry.register("workspace_run", fail, capability="exec")
    with pytest.raises(ToolCaptureFault, match="store_failed"):
        ToolGateway(registry).call(ToolRequest("workspace_run", {"path": "emit.py"}))
