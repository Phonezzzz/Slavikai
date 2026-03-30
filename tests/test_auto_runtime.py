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


class _CapturingVerifierRuntime:
    captured_tasks = []

    def __init__(self, project_root):  # noqa: ANN001
        del project_root

    def run(self, task, context):  # noqa: ANN001
        del context
        self.__class__.captured_tasks.append(task)
        return VerificationResult(
            status=VerificationStatus.PASSED,
            command=["make", "check"],
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


def test_auto_runtime_extracts_missing_paths() -> None:
    failed = auto_runtime.CoderResult(
        coder_id="coder-1",
        shard_id="s1",
        status="failed",
        bundle=auto_runtime.PatchBundle(
            status="failed",
            diagnostics=[
                "Файл не найден: /tmp/project/AGENTS.md",
                "File not found: /tmp/project/docs/README.md",
                "other error",
            ],
        ),
        error="failed",
    )

    missing = auto_runtime._extract_missing_paths([failed])
    assert missing == ["/tmp/project/AGENTS.md", "/tmp/project/docs/README.md"]


def test_auto_runtime_planner_fallback_and_complete(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    tmp_path.mkdir(exist_ok=True)
    agent = _FakeAgent(brain_text="not-json")
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)

    monkeypatch.setattr(
        orchestrator,
        "_run_coder_pool",
        lambda **kwargs: [_completed_result("shard-1", "coder-1")],
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_patch_bundles",
        lambda results, **kwargs: [],
    )
    monkeypatch.setattr(auto_runtime, "VerifierRuntime", _PassingVerifierRuntime)

    outcome = orchestrator.run("Сделать задачу")
    assert outcome.status == AutoRunStatus.COMPLETED
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("status") == AutoRunStatus.COMPLETED.value
    assert agent.last_auto_state.get("root_path") == str(tmp_path.resolve())
    plan_raw = agent.last_auto_state.get("plan")
    assert isinstance(plan_raw, dict)
    shards_raw = plan_raw.get("shards")
    assert isinstance(shards_raw, list)
    assert len(shards_raw) == 1


def test_auto_runtime_uses_canonical_make_check_verifier(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    tmp_path.mkdir(exist_ok=True)
    agent = _FakeAgent(brain_text="not-json")
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)

    monkeypatch.setattr(
        orchestrator,
        "_run_coder_pool",
        lambda **kwargs: [_completed_result("shard-1", "coder-1")],
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_patch_bundles",
        lambda results, **kwargs: [],
    )
    _CapturingVerifierRuntime.captured_tasks = []
    monkeypatch.setattr(auto_runtime, "VerifierRuntime", _CapturingVerifierRuntime)

    outcome = orchestrator.run("Сделать задачу")

    assert outcome.status == AutoRunStatus.COMPLETED
    assert len(_CapturingVerifierRuntime.captured_tasks) == 1
    verifier_task = _CapturingVerifierRuntime.captured_tasks[0]
    assert verifier_task.verifier == {"command": ["make", "check"], "cwd": str(tmp_path)}


@pytest.mark.behavior
def test_auto_runtime_waiting_approval_and_resume(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    planner_payload = '{"plan_id":"p","goal":"g","shards":[{"shard_id":"s1","goal":"g1"}]}'
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir(parents=True, exist_ok=True)
    other_root = tmp_path / "other"
    other_root.mkdir(parents=True, exist_ok=True)

    agent = _FakeAgent(brain_text=planner_payload)
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=fallback_root)

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

    with auto_runtime.workspace_root_context(runtime_root):
        with pytest.raises(ApprovalRequired):
            orchestrator.run("goal")

    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("status") == AutoRunStatus.WAITING_APPROVAL.value
    assert agent.last_auto_state.get("root_path") == str(runtime_root.resolve())
    run_id_raw = agent.last_auto_state.get("run_id")
    assert isinstance(run_id_raw, str)

    monkeypatch.setattr(
        orchestrator,
        "_run_coder_pool",
        lambda **kwargs: [_completed_result("s1", "coder-1")],
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_patch_bundles",
        lambda results, **kwargs: [],
    )
    monkeypatch.setattr(auto_runtime, "VerifierRuntime", _PassingVerifierRuntime)

    with auto_runtime.workspace_root_context(other_root):
        resumed = orchestrator.resume(run_id_raw)
    assert resumed is not None
    assert resumed.status == AutoRunStatus.COMPLETED
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("status") == AutoRunStatus.COMPLETED.value
    assert agent.last_auto_state.get("root_path") == str(runtime_root.resolve())


def test_auto_runtime_sets_error_code_for_missing_target_path(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    agent = _FakeAgent(brain_text="not-json")
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)
    failed_result = auto_runtime.CoderResult(
        coder_id="coder-1",
        shard_id="shard-1",
        status="failed",
        bundle=auto_runtime.PatchBundle(
            status="failed",
            diagnostics=["Не указан путь к файлу workspace для записи."],
        ),
        error="Не указан путь к файлу workspace для записи.",
    )

    monkeypatch.setattr(orchestrator, "_run_coder_pool", lambda **kwargs: [failed_result])

    outcome = orchestrator.run("Сделай изменения")
    assert outcome.status == AutoRunStatus.FAILED_WORKER
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("error_code") == "missing_target_path"
    missing_paths = agent.last_auto_state.get("missing_paths")
    assert missing_paths == []


def test_auto_runtime_stops_on_files_budget(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    agent = _FakeAgent(brain_text="not-json")
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)

    monkeypatch.setenv("AUTO_MAX_FILES_TOUCHED", "1")
    monkeypatch.setattr(
        orchestrator,
        "_run_coder_pool",
        lambda **kwargs: [_completed_result("shard-1", "coder-1")],
    )
    monkeypatch.setattr(
        orchestrator,
        "_apply_patch_bundles",
        lambda results, **kwargs: ["a.py", "b.py"],
    )

    outcome = orchestrator.run("Сделай изменения")
    assert outcome.status == AutoRunStatus.FAILED_INTERNAL
    assert outcome.stop_reason_code == auto_runtime.StopReasonCode.BUDGET_EXHAUSTED
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("error_code") == "budget_files_touched"
