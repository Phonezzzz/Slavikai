from __future__ import annotations

import pytest

import core.auto_runtime as auto_runtime
from core.approval_policy import ApprovalPrompt, ApprovalRequest, ApprovalRequired
from core.mwv.models import VerificationResult, VerificationStatus
from shared.auto_models import AutoPlan, AutoRunStatus, AutoShard


class _FakeBrain:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, messages, config=None):  # noqa: ANN001
        del messages, config

        class _Result:
            def __init__(self, text: str) -> None:
                self.text = text

        return _Result(self._text)


class _FakeTracer:
    def log(self, event_type: str, message: str, meta=None) -> None:  # noqa: ANN001
        del event_type, message, meta


class _FakeAgent:
    def __init__(self, *, brain_text: str) -> None:
        self._brain = _FakeBrain(brain_text)
        self.main_config = None
        self.session_id = "session-test"
        self.tools_enabled = {"safe_mode": True}
        self.approved_categories: set[str] = set()
        self.tracer = _FakeTracer()
        self.last_auto_state = None

    def _get_main_brain(self):
        return self._brain

    def _build_tool_gateway(self):
        raise RuntimeError("not-used")

    def _append_report_block(self, text: str, **kwargs):  # noqa: ANN003
        del kwargs
        return text

    def _format_stop_response(self, **kwargs):  # noqa: ANN003
        return f"stop:{kwargs.get('what', '')}"


class _PassingVerifierRuntime:
    def __init__(self, project_root):  # noqa: ANN001
        del project_root

    def run(self, context):  # noqa: ANN001
        del context
        return VerificationResult(
            status=VerificationStatus.PASSED,
            command=["check"],
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
            error=None,
        )


def _completed_result(shard_id: str, coder_id: str) -> auto_runtime.CoderResult:
    return auto_runtime.CoderResult(
        coder_id=coder_id,
        shard_id=shard_id,
        status="completed",
        bundle=auto_runtime.PatchBundle(status="ok", changed_paths=[]),
        error=None,
    )


def test_auto_runtime_conflict_detection() -> None:
    plan = AutoPlan(
        plan_id="plan-1",
        goal="goal",
        shards=[
            AutoShard(shard_id="a", goal="a", path_scope=["src/a.py"]),
            AutoShard(shard_id="b", goal="b", path_scope=["src/a.py"]),
        ],
    )
    left = auto_runtime.CoderResult(
        coder_id="coder-1",
        shard_id="a",
        status="completed",
        bundle=auto_runtime.PatchBundle(status="ok", changed_paths=["src/a.py"]),
        error=None,
    )
    right = auto_runtime.CoderResult(
        coder_id="coder-2",
        shard_id="b",
        status="completed",
        bundle=auto_runtime.PatchBundle(status="ok", changed_paths=["src/a.py"]),
        error=None,
    )

    conflict = auto_runtime._detect_conflict([left, right], plan)
    assert conflict is not None
    assert conflict[2] == ["src/a.py"]


def test_auto_runtime_planner_fallback_and_complete(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    agent = _FakeAgent(brain_text="not-json")
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)

    monkeypatch.setattr(
        orchestrator,
        "_run_coder_pool",
        lambda **kwargs: [_completed_result("shard-1", "coder-1")],
    )
    monkeypatch.setattr(orchestrator, "_apply_patch_bundles", lambda results: [])
    monkeypatch.setattr(auto_runtime, "VerifierRuntime", _PassingVerifierRuntime)

    outcome = orchestrator.run("Сделать задачу")
    assert outcome.status == AutoRunStatus.COMPLETED
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("status") == AutoRunStatus.COMPLETED.value
    plan_raw = agent.last_auto_state.get("plan")
    assert isinstance(plan_raw, dict)
    shards_raw = plan_raw.get("shards")
    assert isinstance(shards_raw, list)
    assert len(shards_raw) == 1


def test_auto_runtime_waiting_approval_and_resume(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    planner_payload = '{"plan_id":"p","goal":"g","shards":[{"shard_id":"s1","goal":"g1"}]}'
    agent = _FakeAgent(brain_text=planner_payload)
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)

    request = ApprovalRequest(
        category="EXEC_ARBITRARY",
        required_categories=["EXEC_ARBITRARY"],
        prompt=ApprovalPrompt(
            what="need approval",
            why="test",
            risk="risk",
            changes=["files"],
        ),
        tool="workspace_write",
        details={"path": "a.txt"},
        session_id="session-test",
    )

    def _raise_approval(**kwargs):  # noqa: ANN003
        del kwargs
        raise ApprovalRequired(request)

    monkeypatch.setattr(orchestrator, "_run_coder_pool", _raise_approval)

    with pytest.raises(ApprovalRequired):
        orchestrator.run("goal")

    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("status") == AutoRunStatus.WAITING_APPROVAL.value
    run_id_raw = agent.last_auto_state.get("run_id")
    assert isinstance(run_id_raw, str)

    monkeypatch.setattr(
        orchestrator,
        "_run_coder_pool",
        lambda **kwargs: [_completed_result("s1", "coder-1")],
    )
    monkeypatch.setattr(orchestrator, "_apply_patch_bundles", lambda results: [])
    monkeypatch.setattr(auto_runtime, "VerifierRuntime", _PassingVerifierRuntime)

    resumed = orchestrator.resume(run_id_raw)
    assert resumed is not None
    assert resumed.status == AutoRunStatus.COMPLETED
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("status") == AutoRunStatus.COMPLETED.value
