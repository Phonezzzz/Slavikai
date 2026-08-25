from __future__ import annotations

import pytest

import core.auto_runtime as auto_runtime
from core.approval_policy import ApprovalPrompt, ApprovalRequest, ApprovalRequired
from core.auto_agent import AutoAgent
from core.mwv.models import VerificationResult, VerificationStatus
from core.skills.index import SkillResolution
from core.skills.models import SkillEntry
from core.tool_gateway import ToolGateway
from core.tool_loop import ExecutedToolCall
from llm.types import LLMResult, ToolCall, ToolSpec
from shared.auto_models import AutoPlan, AutoRunStatus, AutoShard
from shared.models import LLMMessage, ToolResult
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import workspace_root_context


class _FakeBrain:
    supports_native_tools = False

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
        self.last_report_kwargs = None
        self.report_kwargs_history = []
        self.last_stop_kwargs = None

    def _get_main_brain(self):
        return self._brain

    def _build_tool_gateway(self):
        raise RuntimeError("not-used")

    def _append_report_block(self, text: str, **kwargs):  # noqa: ANN003
        self.last_report_kwargs = kwargs
        self.report_kwargs_history.append(kwargs)
        return text

    def _format_stop_response(self, **kwargs):  # noqa: ANN003
        self.last_stop_kwargs = kwargs
        return f"stop:{kwargs.get('what', '')}"


class _ToolLoopBrain:
    supports_native_tools = True

    def __init__(self) -> None:
        self.calls = 0
        self.seen_tools: list[ToolSpec] = []
        self.messages_seen: list[list[LLMMessage]] = []

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del config
        self.calls += 1
        self.seen_tools = list(tools or [])
        self.messages_seen.append(list(messages))
        if self.calls == 1:
            return LLMResult(
                text="call tool",
                tool_calls=[
                    ToolCall(
                        id="auto-call-1",
                        name="echo",
                        arguments={"value": messages[-1].content},
                    )
                ],
            )
        assert messages[-1].role == "tool"
        return LLMResult(text="auto v1 final")


class _ResponseOnlyBrain:
    supports_native_tools = True

    def __init__(self) -> None:
        self.messages_seen: list[list[LLMMessage]] = []

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del config, tools
        self.messages_seen.append(list(messages))
        return LLMResult(text="Привет! Чем помочь?")


class _AutoV1Agent(_FakeAgent):
    def __init__(self) -> None:
        super().__init__(brain_text="")
        self._brain = _ToolLoopBrain()
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(
            "echo",
            lambda request: ToolResult.success({"output": request.args["value"]}),
            description="Echo auto input",
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )

    def _build_tool_gateway(self):
        return ToolGateway(self.tool_registry)


class _RecoveryToolLoopBrain:
    """First call triggers a failing tool; second call recovers with a final answer."""

    supports_native_tools = True

    def __init__(self) -> None:
        self.calls = 0
        self.messages_seen: list[list[LLMMessage]] = []

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del config, tools
        self.calls += 1
        self.messages_seen.append(list(messages))
        if self.calls == 1:
            return LLMResult(
                text="call tool",
                tool_calls=[
                    ToolCall(id="auto-call-fail", name="fail_tool", arguments={}),
                ],
            )
        assert messages[-1].role == "tool"
        return LLMResult(text="auto v1 final after recovery")


class _RecoveryAgent(_FakeAgent):
    def __init__(self) -> None:
        super().__init__(brain_text="")
        self._brain = _RecoveryToolLoopBrain()
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(
            "fail_tool",
            lambda request: ToolResult.failure("boom"),
            description="Always fails",
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )

    def _build_tool_gateway(self):
        return ToolGateway(self.tool_registry)


class _ResponseOnlyAgent(_FakeAgent):
    def __init__(self) -> None:
        super().__init__(brain_text="")
        self._brain = _ResponseOnlyBrain()
        self.tool_registry = ToolRegistry()

    def _build_tool_gateway(self):
        return ToolGateway(self.tool_registry)


class _PassingVerifierRuntime:
    def __init__(self, project_root):  # noqa: ANN001
        del project_root

    def run(self, task, context):  # noqa: ANN001
        del task, context
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


def _skill_resolution() -> SkillResolution:
    supporting = SkillEntry(
        id="codebase-design",
        version="1.0.0",
        title="Codebase design",
        entrypoints=[],
        patterns=[],
        requires=[],
        risk="low",
        tests=[],
        path="skills/engineering/codebase-design/skill.md",
        content_hash="supporting-hash",
        instructions="Use domain vocabulary from the repository.",
        dependencies=[],
        supporting=True,
    )
    primary = SkillEntry(
        id="implement",
        version="1.0.0",
        title="Implement",
        entrypoints=["auto"],
        patterns=["implement"],
        requires=[],
        risk="medium",
        tests=[],
        path="skills/engineering/implement/skill.md",
        content_hash="primary-hash",
        instructions="Implement the requested coherent change.",
        dependencies=["codebase-design"],
        supporting=False,
    )
    return SkillResolution(primary=primary, supporting=(supporting,))


def test_auto_agent_run_outcome_uses_auto_v1_tool_loop(monkeypatch) -> None:  # noqa: ANN001
    agent = _AutoV1Agent()
    auto = AutoAgent(agent)  # type: ignore[arg-type]
    monkeypatch.setattr(auto_runtime, "VerifierRuntime", _PassingVerifierRuntime)

    outcome = auto.run_outcome("inspect workspace", skill_resolution=_skill_resolution())

    assert outcome.status == AutoRunStatus.COMPLETED
    assert isinstance(agent._brain, _ToolLoopBrain)
    assert agent._brain.calls == 2
    assert agent._brain.seen_tools == [
        ToolSpec(
            name="echo",
            description="Echo auto input",
            parameters_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )
    ]
    assert "implement@1.0.0" in agent._brain.messages_seen[0][0].content
    assert isinstance(agent.last_auto_state, dict)
    planner = agent.last_auto_state.get("planner")
    assert isinstance(planner, dict)
    assert planner.get("runtime") == "auto_v1_tool_loop"
    coders = agent.last_auto_state.get("coders")
    assert isinstance(coders, list)
    assert coders[0]["tool"] == "echo"
    assert coders[0]["status"] == "completed"
    verifier = agent.last_auto_state.get("verifier")
    assert isinstance(verifier, dict)
    assert verifier.get("verifier_profile") == "tool_outcomes"


def test_auto_success_text_excludes_service_report(monkeypatch) -> None:  # noqa: ANN001
    agent = _AutoV1Agent()
    auto = AutoAgent(agent)  # type: ignore[arg-type]
    monkeypatch.setattr(auto_runtime, "VerifierRuntime", _PassingVerifierRuntime)

    outcome = auto.run_outcome("inspect workspace", skill_resolution=_skill_resolution())

    assert outcome.status == AutoRunStatus.COMPLETED
    assert "Auto-run v1 завершён успешно" not in outcome.text
    assert "Tool calls:" not in outcome.text
    assert "auto v1 final" in outcome.text


def test_auto_v1_recovery_from_failed_tool_call_is_not_worker_failed(
    monkeypatch,
    tmp_path,
) -> None:  # noqa: ANN001
    agent = _RecoveryAgent()
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)  # type: ignore[arg-type]
    run_root = tmp_path / "runtime"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "Makefile").write_text("check:\n\t@true\n", encoding="utf-8")
    monkeypatch.setattr(auto_runtime, "VerifierRuntime", _PassingVerifierRuntime)

    with workspace_root_context(run_root):
        outcome = orchestrator.run_v1(
            "inspect workspace",
            skill_resolution=_skill_resolution(),
            run_root_override=run_root,
        )

    assert outcome.status == AutoRunStatus.COMPLETED
    assert isinstance(agent._brain, _RecoveryToolLoopBrain)
    assert agent._brain.calls == 2
    coders = agent.last_auto_state.get("coders")
    assert isinstance(coders, list)
    assert coders[0]["tool"] == "fail_tool"
    assert coders[0]["status"] == "failed"
    verifier = agent.last_auto_state.get("verifier")
    assert isinstance(verifier, dict)
    assert verifier.get("status") == VerificationStatus.PASSED.value


def test_auto_runtime_returns_conversation_without_forcing_tool_action(tmp_path) -> None:
    agent = _ResponseOnlyAgent()
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)

    outcome = orchestrator.run_v1("privet", run_root_override=tmp_path)

    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.text == "Привет! Чем помочь?"
    assert outcome.stop_reason_code is None
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("coders") == []
    verifier = agent.last_auto_state.get("verifier")
    assert isinstance(verifier, dict)
    assert verifier.get("status") == VerificationStatus.PASSED.value
    assert verifier.get("verifier_profile") == "response_only"


def test_auto_runtime_injects_skill_per_run_without_changing_tools_or_brain(tmp_path) -> None:
    agent = _ResponseOnlyAgent()
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)
    resolution = _skill_resolution()

    outcome = orchestrator.run_v1(
        "implement the change",
        skill_resolution=resolution,
        run_root_override=tmp_path,
    )
    second = orchestrator.run_v1("ordinary chat", run_root_override=tmp_path)

    assert outcome.status == AutoRunStatus.COMPLETED
    assert second.status == AutoRunStatus.COMPLETED
    assert isinstance(agent._brain, _ResponseOnlyBrain)
    first_system = agent._brain.messages_seen[0][0].content
    second_system = agent._brain.messages_seen[1][0].content
    assert "codebase-design@1.0.0" in first_system
    assert "implement@1.0.0" in first_system
    assert "do not grant tools" in first_system
    assert "codebase-design@1.0.0" not in second_system
    assert not hasattr(agent._brain, "skill_resolution")
    assert agent._brain.messages_seen[0][0].role == "system"
    assert agent.report_kwargs_history[0]["skill"] == {
        "status": "completed",
        "skill_id": "implement",
        "version": "1.0.0",
        "supporting_skills": [{"skill_id": "codebase-design", "version": "1.0.0"}],
    }
    assert agent.report_kwargs_history[1]["skill"] == {
        "status": "skipped",
        "skill_id": None,
        "version": None,
        "supporting_skills": [],
        "reason": "no_match",
    }


def test_auto_runtime_marks_selected_skill_failed_on_terminal_stop(tmp_path) -> None:
    agent = _FakeAgent(brain_text="unused")
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)

    outcome = orchestrator.run_v1(
        "implement the change",
        skill_resolution=_skill_resolution(),
        run_root_override=tmp_path,
    )

    assert outcome.status == AutoRunStatus.FAILED_WORKER
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state["skill"]["status"] == "failed"
    assert agent.last_auto_state["skill"]["skill_id"] == "implement"
    assert agent.last_stop_kwargs is not None
    assert agent.last_stop_kwargs["skill"]["status"] == "failed"


def test_auto_runtime_rejects_provider_without_native_tools(tmp_path) -> None:
    agent = _FakeAgent(brain_text="unused")
    orchestrator = auto_runtime.AutoOrchestrator(agent, workspace_root=tmp_path)

    outcome = orchestrator.run_v1("write a file", run_root_override=tmp_path)

    assert outcome.status == AutoRunStatus.FAILED_WORKER
    assert outcome.stop_reason_code is not None
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("error_code") == "native_tools_required"
    assert "DeepSeek" in outcome.next_steps[0]


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


def test_auto_orchestrator_has_no_legacy_run_entrypoint() -> None:
    assert not hasattr(auto_runtime.AutoOrchestrator, "run")


@pytest.mark.behavior
def test_auto_runtime_v1_waiting_approval_and_resume(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    other_root = tmp_path / "other"
    other_root.mkdir(parents=True, exist_ok=True)
    agent = _AutoV1Agent()
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

    class _ApprovalThenSuccessLoop:
        calls = 0

        def __init__(self, max_iterations: int) -> None:
            del max_iterations

        def run(self, **kwargs):  # noqa: ANN003
            del kwargs
            self.__class__.calls += 1
            if self.__class__.calls == 1:
                raise ApprovalRequired(request)

            class _LoopResult:
                tool_calls = [
                    ExecutedToolCall(
                        call=ToolCall(id="resume-call", name="echo", arguments={}),
                        result=ToolResult.success({"output": "ok"}),
                    )
                ]
                iterations = 1
                text = "resumed"

            return _LoopResult()

    _ApprovalThenSuccessLoop.calls = 0
    monkeypatch.setattr(auto_runtime, "AgentToolLoop", _ApprovalThenSuccessLoop)
    monkeypatch.setattr(auto_runtime, "VerifierRuntime", _PassingVerifierRuntime)

    with workspace_root_context(runtime_root):
        with pytest.raises(ApprovalRequired):
            orchestrator.run_v1("goal")

    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("status") == AutoRunStatus.WAITING_APPROVAL.value
    assert agent.last_auto_state.get("root_path") == str(runtime_root.resolve())
    run_id_raw = agent.last_auto_state.get("run_id")
    assert isinstance(run_id_raw, str)

    with workspace_root_context(other_root):
        resumed = orchestrator.resume(run_id_raw)

    assert resumed is not None
    assert resumed.status == AutoRunStatus.COMPLETED
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("status") == AutoRunStatus.COMPLETED.value
    assert agent.last_auto_state.get("root_path") == str(runtime_root.resolve())
