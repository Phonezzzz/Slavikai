from __future__ import annotations

import pytest


@pytest.mark.parametrize("approval", [False, True])
def test_trusted_capture_context_reaches_gateway_and_fault_stops_next_batch_call(
    tmp_path, approval
):
    from core.approval_policy import ApprovalContext, ApprovalRequired
    from core.task_run_storage import TaskRunStore
    from core.tool_gateway import ToolGateway
    from core.tool_loop import AgentToolLoop, ToolLoopExecutionError
    from llm.types import LLMResult, ToolCall
    from shared.models import LLMMessage, ToolResult
    from shared.task_run import RunScope, RunState, TransitionAuthority
    from shared.tool_capture import CaptureBinding, ToolCaptureFault
    from tools.tool_logger import ToolCallLogger
    from tools.tool_registry import ToolRegistry

    store = TaskRunStore(tmp_path / "runs.db")
    scope = RunScope("alice", "chat")
    admitted = store.admit(
        scope, request_key="k", goal="synthetic", mode="auto", request_fingerprint="a" * 64
    )
    run = store.transition(
        scope,
        admitted.task_run_id,
        expected_version=0,
        state=RunState.RUNNING,
        authority=TransitionAuthority.RUNTIME,
        reason_code="started",
    )
    binding = CaptureBinding(store, run)
    calls = []

    class Brain:
        calls = 0

        def generate(self, messages, config=None, tools=None):
            self.calls += 1
            return LLMResult(
                text="",
                tool_calls=[
                    ToolCall(
                        "label",
                        "workspace_run",
                        {
                            "path": "emit.py",
                            "capture": {"principal_id": "mallory", "task_run_id": "forged"},
                        },
                    ),
                    ToolCall("later", "write", {}),
                ],
            )

    def fail(request):
        calls.append(request.name)
        assert request.capture.binding is binding
        assert request.capture.model_call_id == "label"
        raise ToolCaptureFault("workspace_capture_store_failed")

    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "calls.log"))
    registry.register("workspace_run", fail, capability="exec", risk_classes=["execute"])
    registry.register(
        "write", lambda request: calls.append("write") or ToolResult.success({}), capability="write"
    )
    registry.set_execution_policy(mode="auto")
    brain = Brain()
    gateway = ToolGateway(
        registry, approval_context=ApprovalContext(True, "chat", set()) if approval else None
    )
    loop = AgentToolLoop()
    arguments = dict(
        brain=brain,
        gateway=gateway,
        messages=[LLMMessage("user", "synthetic")],
        tools=registry.list_tool_specs(),
        capture_binding=binding,
    )
    if approval:
        with pytest.raises(ApprovalRequired) as pending:
            loop.run(**arguments)
        continuation = pending.value.continuation
        assert continuation.capture_binding is binding and not calls
        other = store.admit(
            scope, request_key="other", goal="other", mode="auto", request_fingerprint="b" * 64
        )
        other = store.transition(
            scope,
            other.task_run_id,
            expected_version=0,
            state=RunState.RUNNING,
            authority=TransitionAuthority.RUNTIME,
            reason_code="started",
        )
        with pytest.raises(ToolCaptureFault, match="continuation_binding_changed"):
            loop.run(
                **(arguments | {"capture_binding": CaptureBinding(store, other)}),
                continuation=continuation,
            )
        assert not calls
        arguments["continuation"] = continuation
    with pytest.raises(ToolLoopExecutionError) as error:
        loop.run(**arguments)
    assert isinstance(error.value.cause, ToolCaptureFault)
    assert error.value.result.error == "tool_capture_failed"
    assert calls == ["workspace_run"] and brain.calls == 1
