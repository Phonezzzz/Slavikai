"""Auto as a policy preset over the canonical loop (E01-T07).

Ports tests/test_auto_runtime.py: every behaviour the old engine really
had (canonical loop, budgets, approval pause/resume, fail-closed tool
errors, skill injection/reporting, usage accounting, file-effects cap)
is asserted against ``AutoPresetRunner``. Assertions that only checked
the old orchestration theatre (fabricated plan/shards, planner/coders/
merge labels, fake ``PASSED`` verifier profiles) are replaced by the
honest equivalents: no fabricated plan in state, ``verifier is None``
where no real verifier ran.
"""

from __future__ import annotations

import json

import pytest

import core.auto_preset as auto_preset
from core.agent_tools import AgentToolsMixin
from core.approval_policy import ApprovalPrompt, ApprovalRequest, ApprovalRequired
from core.auto_preset import AutoPresetRunner
from core.mwv.models import StopReasonCode, VerificationResult, VerificationStatus
from core.response_preferences import ResponsePreferences
from core.skills.index import SkillResolution
from core.skills.models import SkillEntry
from core.tool_gateway import ToolGateway
from core.tool_loop import ExecutedToolCall
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ToolCall, ToolSpec
from shared.auto_models import AutoRunStatus
from shared.models import LLMMessage, ToolResult
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import RunCodeTool, WriteFileTool, workspace_root_context


@pytest.mark.parametrize("exit_code", [0, 9])
def test_auto_process_outcome_through_gateway_and_next_model_request(tmp_path, exit_code) -> None:
    (tmp_path / "result.py").write_text(
        "import sys\nprint('process output')\nprint('process diagnostic', file=sys.stderr)\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )

    class ScriptBrain(Brain):
        supports_native_tools = True

        def __init__(self):
            self.projection = None

        def generate(self, messages, config=None, tools=None):
            del config, tools
            if messages[-1].role != "tool":
                return LLMResult(
                    usage=LLMUsage(0, 0, 0),
                    text="run script",
                    tool_calls=[
                        ToolCall(
                            id="script-call", name="workspace_run", arguments={"path": "result.py"}
                        )
                    ],
                )
            self.projection = json.loads(messages[-1].content)
            assert messages[-1].tool_call_id == "script-call"
            return LLMResult(usage=LLMUsage(0, 0, 0), text="script completed")

    class FacadeAgent(_PresetAgent, AgentToolsMixin):
        pass

    agent = FacadeAgent()
    brain = ScriptBrain()
    agent._brain = brain
    agent.tool_registry.register(
        "workspace_run",
        RunCodeTool().handle,
        description="Run a workspace script",
        parameters_schema={"type": "object"},
    )
    agent.auto_preset = AutoPresetRunner(agent, workspace_root=tmp_path)
    with workspace_root_context(tmp_path):
        outcome = agent.handle_auto_command("run script")

    assert isinstance(outcome, auto_preset.AutoRunOutcome)
    assert outcome.status == AutoRunStatus.FAILED_WORKER
    assert outcome.stop_reason_code == StopReasonCode.WORKER_FAILED
    assert outcome.verifier is None
    assert brain.projection["trust"] == "untrusted_observation"
    assert brain.projection["ok"] is False
    assert "exit_code" not in brain.projection["data"]
    assert "изолированная среда" in brain.projection["error"]


class _FakeBrain(Brain):
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
    response_preferences = ResponsePreferences()

    def __init__(self, *, brain_text: str) -> None:
        self._brain = _FakeBrain(brain_text)
        self.main_config = None
        self.session_id = "session-test"
        self.user_id = "test-principal"
        self.runtime_mode = "auto"
        self.runtime_workspace_root = None
        self.tools_enabled = {"safe_mode": True}
        self.approved_categories: set[str] = set()
        self.tracer = _FakeTracer()
        self.last_auto_state = None
        self.last_report_kwargs = None
        self.report_kwargs_history = []
        self.last_stop_kwargs = None

    def _get_main_brain(self):
        return self._brain

    def _build_tool_gateway(self, *, budget=None):
        del budget
        raise RuntimeError("not-used")

    def _append_report_block(self, text: str, **kwargs):  # noqa: ANN003
        self.last_report_kwargs = kwargs
        self.report_kwargs_history.append(kwargs)
        return text

    def _format_stop_response(self, **kwargs):  # noqa: ANN003
        self.last_stop_kwargs = kwargs
        return f"stop:{kwargs.get('what', '')}"


class _ToolLoopBrain(Brain):
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
                usage=LLMUsage(0, 0, 0),
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
        return LLMResult(usage=LLMUsage(0, 0, 0), text="auto v1 final")


class _ResponseOnlyBrain(Brain):
    supports_native_tools = True

    def __init__(self) -> None:
        self.messages_seen: list[list[LLMMessage]] = []

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del config, tools
        self.messages_seen.append(list(messages))
        return LLMResult(usage=LLMUsage(0, 0, 0), text="Привет! Чем помочь?")


class _PresetAgent(_FakeAgent):
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

    def _build_tool_gateway(self, *, budget=None):
        del budget
        return ToolGateway(self.tool_registry)


@pytest.mark.parametrize("batch_size", [2, 3])
def test_auto_native_call_cap_before_dispatch(tmp_path, monkeypatch, batch_size):
    class BatchBrain(Brain):
        supports_native_tools = True

        def generate(self, messages, config=None, tools=None):
            if messages[-1].role == "tool":
                return LLMResult(usage=LLMUsage(0, 0, 0), text="finished")
            return LLMResult(
                usage=LLMUsage(0, 0, 0),
                text="batch",
                tool_calls=[
                    ToolCall(id=str(i), name="counted", arguments={"value": i})
                    for i in range(batch_size)
                ],
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(auto_preset.AUTO_MAX_TOOL_CALLS_ENV, "2")
    agent = _PresetAgent()
    agent._brain = BatchBrain()
    dispatched = []
    agent.tool_registry.register(
        "counted",
        lambda request: dispatched.append(request.args["value"])
        or ToolResult.success({"output": "ok"}),
        capability="read",
    )
    outcome = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path).run("count")
    assert dispatched == [0, 1]
    assert agent.last_auto_state["execution_metrics"]["tool_calls_used"] == 2
    if batch_size == 3:
        assert outcome.stop_reason_code == StopReasonCode.BUDGET_EXHAUSTED
        assert outcome.verifier is None
    else:
        assert outcome.status == AutoRunStatus.COMPLETED
        # No tool call certifies the goal: the preset claims no verifier.
        assert outcome.verifier is None


def test_auto_call_cap_is_run_scoped_and_retained_across_approval(tmp_path, monkeypatch):
    from core.approval_policy import ApprovalContext
    from tests.test_tool_call_cap import BatchBrain, calls

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(auto_preset.AUTO_MAX_TOOL_CALLS_ENV, "2")
    runs = []
    for principal in ["one", "two"]:
        agent = _PresetAgent()
        agent.user_id = principal
        agent.session_id = principal
        agent._brain = BatchBrain([calls("echo", "network", "network")])
        dispatched = []
        # Каждый run наблюдает собственные handler calls.
        agent.tool_registry.register(
            "echo",
            lambda request, seen=dispatched: seen.append(request.name) or ToolResult.success({}),
            capability="read",
        )
        agent.tool_registry.register(
            "network",
            lambda request, seen=dispatched: seen.append(request.name) or ToolResult.success({}),
            capability="read",
            risk_classes=["network"],
        )
        gateway = ToolGateway(
            agent.tool_registry, approval_context=ApprovalContext(True, principal, set())
        )
        agent._build_tool_gateway = lambda current=gateway, budget=None: current
        preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)
        with pytest.raises(ApprovalRequired):
            preset.run("count")
        assert dispatched == ["echo"]
        assert agent.last_auto_state["execution_metrics"]["files_touched"] == 0
        runs.append((preset, agent, dispatched, agent.last_auto_state["run_id"]))
    monkeypatch.setenv(auto_preset.AUTO_MAX_TOOL_CALLS_ENV, "100")
    for preset, agent, dispatched, run_id in runs:
        outcome = preset.resume(run_id)
        assert outcome.stop_reason_code == StopReasonCode.BUDGET_EXHAUSTED
        assert dispatched == ["echo", "network"]
        assert agent.last_auto_state["execution_metrics"]["tool_calls_used"] == 2
        with pytest.raises(auto_preset.AutoContinuationUnavailable):
            preset.resume(run_id)


def test_auto_resume_publishes_running_before_continuation(tmp_path, monkeypatch):
    from core.approval_policy import ApprovalContext
    from tests.test_tool_call_cap import BatchBrain, calls

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(auto_preset.AUTO_MAX_TOOL_CALLS_ENV, "2")
    agent = _PresetAgent()
    agent._brain = BatchBrain([calls("echo", "network", "network")])
    dispatched = []
    agent.tool_registry.register(
        "echo",
        lambda request, seen=dispatched: seen.append(request.name) or ToolResult.success({}),
        capability="read",
    )
    agent.tool_registry.register(
        "network",
        lambda request, seen=dispatched: seen.append(request.name) or ToolResult.success({}),
        capability="read",
        risk_classes=["network"],
    )
    gateway = ToolGateway(
        agent.tool_registry, approval_context=ApprovalContext(True, agent.user_id, set())
    )
    agent._build_tool_gateway = lambda current=gateway, budget=None: current
    seen_statuses: list[str] = []
    preset = auto_preset.AutoPresetRunner(
        agent,
        workspace_root=tmp_path,
        progress_callback=lambda state: seen_statuses.append(str(state.get("status"))),
    )
    with pytest.raises(ApprovalRequired):
        preset.run("count")
    assert agent.last_auto_state["status"] == AutoRunStatus.WAITING_APPROVAL.value
    monkeypatch.setenv(auto_preset.AUTO_MAX_TOOL_CALLS_ENV, "100")
    outcome = preset.resume(agent.last_auto_state["run_id"])
    assert outcome.stop_reason_code == StopReasonCode.BUDGET_EXHAUSTED
    # The resumed frame must publish RUNNING before the continuation
    # runs, so the UI stops showing a pending approval mid-execution.
    waiting_index = seen_statuses.index(AutoRunStatus.WAITING_APPROVAL.value)
    assert AutoRunStatus.RUNNING.value in seen_statuses[waiting_index:]


def test_auto_resume_keeps_repo_verification_obligation(tmp_path, monkeypatch):
    # P1 round 4: обязанность repository verification зафиксирована в
    # _RunFrame до loop и переживает resume. Репозиторий без критерия
    # на старте; пока run на паузе, агент подкладывает Makefile —
    # после resume run всё равно завершается FAILED_VERIFIER с
    # canonical_repo_verifier_unavailable, а не COMPLETED.
    from core.approval_policy import ApprovalContext
    from tests.test_tool_call_cap import BatchBrain, calls

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".git").mkdir()
    agent = _PresetAgent()
    agent._brain = BatchBrain([calls("echo"), calls("network")])
    dispatched = []
    agent.tool_registry.register(
        "echo",
        lambda request, seen=dispatched: seen.append(request.name) or ToolResult.success({}),
        capability="read",
    )
    agent.tool_registry.register(
        "network",
        lambda request, seen=dispatched: seen.append(request.name) or ToolResult.success({}),
        capability="read",
        risk_classes=["network"],
    )
    gateway = ToolGateway(
        agent.tool_registry, approval_context=ApprovalContext(True, agent.user_id, set())
    )
    agent._build_tool_gateway = lambda current=gateway, budget=None: current
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)
    with pytest.raises(ApprovalRequired):
        preset.run("count")
    assert agent.last_auto_state["status"] == AutoRunStatus.WAITING_APPROVAL.value
    (tmp_path / "Makefile").write_text("check:\n\t@true\n")
    outcome = preset.resume(agent.last_auto_state["run_id"])
    assert dispatched == ["echo", "network"]
    assert outcome.status == AutoRunStatus.FAILED_VERIFIER
    assert outcome.verifier is not None
    assert outcome.verifier.error == "canonical_repo_verifier_unavailable"


def test_auto_keeps_available_prefix_after_dispatch_observation_fault(tmp_path, monkeypatch):
    from tests.test_tool_call_cap import BatchBrain, calls

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(auto_preset.AUTO_MAX_TOOL_CALLS_ENV, "4")
    agent = _PresetAgent()
    agent._brain = BatchBrain([calls("echo", "second", "echo")])
    dispatched = []
    for name in ["echo", "second"]:
        agent.tool_registry.register(
            name,
            lambda request: dispatched.append(request.name) or ToolResult.success({}),
            capability="read",
        )

    def observe(request, result, context):
        if request.name == "second":
            raise RuntimeError("observation failed after handler")

    agent._build_tool_gateway = lambda budget=None: ToolGateway(
        agent.tool_registry, post_call=observe
    )
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)
    outcome = preset.run("observe")
    assert outcome.status == AutoRunStatus.FAILED_INTERNAL
    assert dispatched == ["echo", "second"]
    assert agent.last_auto_state["execution_metrics"]["tool_calls_used"] == 1
    assert agent.last_auto_state["error"] == "tool_dispatch_failed"
    # An already executed prefix, followed by a dispatch fault, has unknown effects.
    assert agent.last_auto_state["execution_metrics"]["files_touched"] is None
    with pytest.raises(auto_preset.AutoContinuationUnavailable):
        preset.resume(agent.last_auto_state["run_id"])


class _RecoveryToolLoopBrain(Brain):
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
                usage=LLMUsage(0, 0, 0),
                text="call tool",
                tool_calls=[
                    ToolCall(id="auto-call-fail", name="fail_tool", arguments={}),
                ],
            )
        assert messages[-1].role == "tool"
        return LLMResult(usage=LLMUsage(0, 0, 0), text="auto v1 final after recovery")


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

    def _build_tool_gateway(self, *, budget=None):
        del budget
        return ToolGateway(self.tool_registry)


class _ResponseOnlyAgent(_FakeAgent):
    def __init__(self) -> None:
        super().__init__(brain_text="")
        self._brain = _ResponseOnlyBrain()
        self.tool_registry = ToolRegistry()

    def _build_tool_gateway(self, *, budget=None):
        del budget
        return ToolGateway(self.tool_registry)


class _EmptyResponseOnlyBrain(Brain):
    supports_native_tools = True

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del messages, config, tools
        return LLMResult(usage=LLMUsage(0, 0, 0), text="   \n")


class _EmptyResponseOnlyAgent(_FakeAgent):
    def __init__(self) -> None:
        super().__init__(brain_text="")
        self._brain = _EmptyResponseOnlyBrain()
        self.tool_registry = ToolRegistry()

    def _build_tool_gateway(self, *, budget=None):
        del budget
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


def test_auto_preset_run_outcome_uses_canonical_loop(tmp_path) -> None:  # noqa: ANN001
    agent = _PresetAgent()
    preset = AutoPresetRunner(agent, workspace_root=tmp_path)

    outcome = preset.run(
        "inspect workspace", skill_resolution=_skill_resolution(), run_root_override=tmp_path
    )

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
    # No fabricated plan: the preset state carries no planner/coders/merge.
    assert agent.last_auto_state.get("planner") == {}
    assert agent.last_auto_state.get("plan") is None
    assert agent.last_auto_state.get("coders") == []
    assert agent.last_auto_state.get("merge") == {}
    # A bare successful tool call is not a verification claim.
    assert outcome.verifier is None
    assert agent.last_auto_state.get("verifier") is None
    # The fake echo ToolResult does not certify file effects; Auto cannot
    # infer an authoritative count merely from a completed tool call.
    assert agent.last_auto_state["execution_metrics"]["files_touched"] is None


def test_auto_success_text_excludes_service_report(tmp_path) -> None:  # noqa: ANN001
    agent = _PresetAgent()
    preset = AutoPresetRunner(agent, workspace_root=tmp_path)

    outcome = preset.run(
        "inspect workspace", skill_resolution=_skill_resolution(), run_root_override=tmp_path
    )

    assert outcome.status == AutoRunStatus.COMPLETED
    assert "Auto-run v1 завершён успешно" not in outcome.text
    assert "Tool calls:" not in outcome.text
    # Echo tool call без verifier evidence: канонический outcome —
    # UNVERIFIED, финальный текст модели не публикуется как успех.
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.ui_label == "UNVERIFIED"
    assert outcome.text == "Действие выполнено, результат не подтверждён"
    assert "auto v1 final" not in outcome.text


def test_auto_preset_parity_write_scenario(tmp_path, monkeypatch) -> None:
    """Parity: the useful Auto scenario — write a file via the canonical loop.

    The old engine ran this through the same AgentToolLoop + ToolGateway
    and then claimed the goal verified (fake ``tool_outcomes`` PASSED).
    The preset must produce the same file, under the same budgets and
    approval rules, but claim no verification it does not have: a
    non-repository workspace has no canonical verifier, so
    ``outcome.verifier is None`` and the state carries no fabricated
    plan/coders/merge.
    """
    from core.approval_policy import ApprovalContext
    from core.managed_workspace import ManagedWorkspace, WorkspaceAccess

    class WriteBrain(Brain):
        supports_native_tools = True

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, messages, config=None, tools=None):  # noqa: ANN001
            del config, tools
            self.calls += 1
            if self.calls == 1:
                return LLMResult(
                    usage=LLMUsage(0, 0, 0),
                    text="",
                    tool_calls=[
                        ToolCall(
                            id="write-1",
                            name="workspace_write",
                            arguments={"path": "notes.txt", "content": "hello preset"},
                        )
                    ],
                )
            return LLMResult(usage=LLMUsage(0, 0, 0), text="Готово")

    monkeypatch.chdir(tmp_path)
    agent = _PresetAgent()
    agent._brain = WriteBrain()
    agent.tool_registry.register(
        "workspace_write",
        WriteFileTool().handle,
        description="Write a workspace file",
        parameters_schema={"type": "object"},
        capability="write",
    )
    # A workspace write through the preset gateway requires a managed
    # workspace binding, exactly as the real Agent provides it. The
    # approval context carries the session (binding match) with safe mode
    # off, so the write itself needs no approval in this scenario;
    # approval pause/resume is covered by the ported tests.
    workspace = ManagedWorkspace.create(tmp_path, principal_id=agent.user_id)
    agent._build_tool_gateway = lambda budget=None: ToolGateway(
        agent.tool_registry,
        approval_context=ApprovalContext(False, agent.session_id, set()),
        workspace_access=WorkspaceAccess(workspace, agent.user_id, agent.session_id),
    )
    preset = AutoPresetRunner(agent, workspace_root=workspace.root)
    with workspace_root_context(workspace.root):
        outcome = preset.run("create notes.txt with hello preset")

    assert (workspace.root / "notes.txt").read_text(encoding="utf-8") == "hello preset"
    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.stop_reason_code is None
    assert outcome.verifier is None
    state = agent.last_auto_state
    assert isinstance(state, dict)
    assert state["status"] == AutoRunStatus.COMPLETED.value
    assert state["budgets"] == {
        "max_runtime_seconds": 900,
        "max_tool_calls": 80,
        "max_files_touched": 120,
        "max_llm_tokens": 120000,
        "max_retries": 0,
    }
    assert state["execution_metrics"]["files_touched"] == 1
    assert state["execution_metrics"]["tool_calls_used"] == 1
    assert state.get("planner") == {}
    assert state.get("plan") is None
    assert state.get("coders") == []
    assert state.get("merge") == {}


def test_auto_preset_failed_tool_call_without_recovery_is_worker_failed(
    monkeypatch,
    tmp_path,
) -> None:  # noqa: ANN001
    agent = _RecoveryAgent()
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)  # type: ignore[arg-type]
    run_root = tmp_path / "runtime"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "Makefile").write_text("check:\n\t@true\n", encoding="utf-8")
    monkeypatch.setattr(auto_preset, "VerifierRuntime", _PassingVerifierRuntime)

    with workspace_root_context(run_root):
        outcome = preset.run(
            "inspect workspace",
            skill_resolution=_skill_resolution(),
            run_root_override=run_root,
        )

    assert outcome.status == AutoRunStatus.FAILED_WORKER
    assert outcome.stop_reason_code == StopReasonCode.WORKER_FAILED
    assert isinstance(agent._brain, _RecoveryToolLoopBrain)
    assert agent._brain.calls == 2
    assert agent.last_auto_state["execution_metrics"]["tool_calls_used"] == 1


class _CorrectiveRecoveryBrain(Brain):
    supports_native_tools = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del config, tools
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                usage=LLMUsage(0, 0, 0),
                text="call fail_tool",
                tool_calls=[ToolCall(id="c1", name="fail_tool", arguments={})],
            )
        if self.calls == 2:
            assert messages[-1].role == "tool"
            return LLMResult(
                usage=LLMUsage(0, 0, 0),
                text="call ok_tool",
                tool_calls=[ToolCall(id="c2", name="ok_tool", arguments={})],
            )
        return LLMResult(usage=LLMUsage(0, 0, 0), text="done after corrective action")


class _CorrectiveRecoveryAgent(_FakeAgent):
    def __init__(self) -> None:
        super().__init__(brain_text="")
        self._brain = _CorrectiveRecoveryBrain()
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(
            "fail_tool",
            lambda request: ToolResult.failure("boom"),
            description="Always fails",
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )
        self.tool_registry.register(
            "ok_tool",
            lambda request: ToolResult.success({"ok": True}),
            description="Corrective success",
            parameters_schema={"type": "object", "properties": {}, "required": []},
        )

    def _build_tool_gateway(self, *, budget=None):
        del budget
        return ToolGateway(self.tool_registry)


def test_auto_preset_unrelated_success_after_failure_is_worker_failed(
    monkeypatch,
    tmp_path,
) -> None:  # noqa: ANN001
    agent = _CorrectiveRecoveryAgent()
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)  # type: ignore[arg-type]
    run_root = tmp_path / "runtime"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "Makefile").write_text("check:\n\t@true\n", encoding="utf-8")
    monkeypatch.setattr(auto_preset, "VerifierRuntime", _PassingVerifierRuntime)

    with workspace_root_context(run_root):
        outcome = preset.run(
            "inspect workspace",
            skill_resolution=_skill_resolution(),
            run_root_override=run_root,
        )

    assert outcome.status == AutoRunStatus.FAILED_WORKER
    assert outcome.stop_reason_code == StopReasonCode.WORKER_FAILED
    assert agent._brain.calls == 3
    assert agent.last_auto_state["execution_metrics"]["tool_calls_used"] == 2


def test_auto_preset_returns_conversation_without_forcing_tool_action(tmp_path) -> None:
    agent = _ResponseOnlyAgent()
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)

    outcome = preset.run("privet", run_root_override=tmp_path)

    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.text == "Привет! Чем помочь?"
    assert outcome.stop_reason_code is None
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("coders") == []
    # A bare answer is not a verification claim: no fake response_only PASSED.
    assert outcome.verifier is None
    assert agent.last_auto_state.get("verifier") is None
    # No native dispatch means the run itself cannot have changed workspace files.
    assert agent.last_auto_state["execution_metrics"]["files_touched"] == 0


def test_auto_preset_rejects_empty_response_only_result(tmp_path) -> None:
    agent = _EmptyResponseOnlyAgent()
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)

    outcome = preset.run("privet", run_root_override=tmp_path)

    # A blank answer with no tool calls is not a completion: the user
    # got neither an answer nor an executed action.
    assert outcome.status == AutoRunStatus.FAILED_VERIFIER
    assert outcome.stop_reason_code == StopReasonCode.VERIFIER_FAILED
    assert outcome.verifier is not None
    assert outcome.verifier.error == "auto_no_observable_result"
    assert agent.last_auto_state["status"] == AutoRunStatus.FAILED_VERIFIER.value


def test_auto_preset_injects_skill_per_run_without_changing_tools_or_brain(tmp_path) -> None:
    agent = _ResponseOnlyAgent()
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)
    resolution = _skill_resolution()

    outcome = preset.run(
        "implement the change",
        skill_resolution=resolution,
        run_root_override=tmp_path,
    )
    second = preset.run("ordinary chat", run_root_override=tmp_path)

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


def test_auto_preset_marks_selected_skill_failed_on_terminal_stop(tmp_path) -> None:
    agent = _FakeAgent(brain_text="unused")
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)

    outcome = preset.run(
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


def test_auto_preset_rejects_provider_without_native_tools(tmp_path) -> None:
    agent = _FakeAgent(brain_text="unused")
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)

    outcome = preset.run("write a file", run_root_override=tmp_path)

    assert outcome.status == AutoRunStatus.FAILED_WORKER
    assert outcome.stop_reason_code is not None
    assert isinstance(agent.last_auto_state, dict)
    assert agent.last_auto_state.get("error_code") == "native_tools_required"
    assert "native tools" in outcome.next_steps[0]
    assert "не поддерживает" in agent.last_auto_state["error"]


def test_auto_preset_has_no_legacy_run_entrypoint() -> None:
    # The old engine's entrypoint was run_v1; the preset's canonical
    # entrypoint is run. Nothing else may start a run.
    assert not hasattr(auto_preset.AutoPresetRunner, "run_v1")
    assert callable(auto_preset.AutoPresetRunner.run)


@pytest.mark.behavior
def test_auto_preset_rejects_approval_without_native_checkpoint(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    other_root = tmp_path / "other"
    other_root.mkdir(parents=True, exist_ok=True)
    agent = _PresetAgent()
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=tmp_path)
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

        def __init__(
            self,
            max_iterations: int,
            *,
            max_tool_calls: int | None = None,
            expires_at_monotonic: float | None = None,
            max_model_tokens: int | None = None,
        ) -> None:
            del max_iterations, max_tool_calls, expires_at_monotonic, max_model_tokens

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
    monkeypatch.setattr(auto_preset, "AgentToolLoop", _ApprovalThenSuccessLoop)
    monkeypatch.setattr(auto_preset, "VerifierRuntime", _PassingVerifierRuntime)

    with workspace_root_context(runtime_root):
        with pytest.raises(auto_preset.AutoContinuationUnavailable):
            preset.run("goal")
    assert not preset._paused_runs


@pytest.mark.parametrize("override", ["GIT_DIR", "GIT_WORK_TREE", "GIT_CONFIG_GLOBAL"])
def test_auto_does_not_claim_process_free_verification_when_git_context_is_ambiguous(
    tmp_path,
    monkeypatch,
    override,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(override, str(tmp_path / "private-git-context"))
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    agent = _PresetAgent()
    preset = AutoPresetRunner(agent, workspace_root=tmp_path)
    with workspace_root_context(tmp_path):
        result = preset.run("inspect workspace")
    assert calls == []
    assert result.status == AutoRunStatus.FAILED_VERIFIER
    assert result.verifier.error == "canonical_repo_verifier_unavailable"


@pytest.mark.parametrize("case", ["retry", "cleanup_failure", "missing_evidence"])
def test_auto_file_budget_survives_approval_and_blocks_publication(tmp_path, monkeypatch, case):
    from types import SimpleNamespace

    from core.approval_policy import ApprovalContext
    from core.managed_workspace import ManagedWorkspace, WorkspaceAccess, WorkspaceLimits
    from core.workspace_python_execution import WorkspacePythonOutcome
    from shared.workspace_binding import WorkspaceFileBudgetExceeded
    from tests.test_tool_call_cap import BatchBrain

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(auto_preset.AUTO_MAX_FILES_TOUCHED_ENV, "2")
    agent = _PresetAgent()
    workspace = ManagedWorkspace.create(tmp_path, principal_id=agent.user_id)
    storage = tmp_path / "staging"
    storage.mkdir(mode=0o700)
    for script in ("initial.py", "repeat.py", "extra.py"):
        (workspace.root / script).write_text("# test input")
        (workspace.root / script).chmod(0o600)
    limits = WorkspaceLimits(1024, 4096, 20, 8)
    executed = []

    def execute(
        binding, script, *, timeout, capture, max_files_touched=None, already_touched=frozenset()
    ):
        executed.append(script)
        snapshot = workspace.capture(binding, limits=limits)
        stage = workspace.stage(snapshot, storage=storage)
        stage.begin_process()
        if script == "initial.py":
            (stage.root / "folder").mkdir(exist_ok=True)
            (stage.root / "data").write_text("first")
        elif script == "repeat.py":
            (stage.root / "data").write_text("second")
        else:
            (stage.root / "excess").mkdir()
        stage.mark_settled()
        try:
            publication = workspace.publish_bounded(
                snapshot,
                stage,
                limits=limits,
                max_files_touched=max_files_touched,
                already_touched=already_touched,
            )
        except WorkspaceFileBudgetExceeded as exc:
            return WorkspacePythonOutcome(
                "unavailable",
                b"",
                b"",
                0,
                True,
                exc.reason_code,
                "retained" if case == "cleanup_failure" else "none",
                cleanup_warning=case == "cleanup_failure",
            )
        return WorkspacePythonOutcome(
            "finished",
            b"",
            b"",
            0,
            True,
            None,
            "published",
            changed_paths=None
            if case == "missing_evidence" and script == "repeat.py"
            else publication.changed_paths,
        )

    agent.tool_registry.register(
        "workspace_run",
        RunCodeTool(backend=SimpleNamespace(execute_python=execute)).handle,
        capability="exec",
    )
    agent._brain = BatchBrain(
        [
            [
                ToolCall(id="missing", name="workspace_run", arguments={"path": "missing.py"}),
                ToolCall(id="1", name="workspace_run", arguments={"path": "initial.py"}),
                ToolCall(
                    id="3",
                    name="workspace_run",
                    arguments={"path": "repeat.py", "config_path": "repeat.py"},
                ),
                ToolCall(
                    id="4",
                    name="workspace_run",
                    arguments={"path": "extra.py", "max_files_touched": 99},
                ),
                ToolCall(id="5", name="workspace_run", arguments={"path": "initial.py"}),
            ]
        ]
    )
    agent._build_tool_gateway = lambda budget=None: ToolGateway(
        agent.tool_registry,
        approval_context=ApprovalContext(True, agent.session_id, {"EXEC_ARBITRARY"}),
        workspace_access=WorkspaceAccess(workspace, agent.user_id, agent.session_id),
    )
    preset = auto_preset.AutoPresetRunner(agent, workspace_root=workspace.root)
    with workspace_root_context(workspace.root):
        with pytest.raises(ApprovalRequired) as paused:
            preset.run("modify")
        assert paused.value.request.tool == "workspace_run"
        assert (workspace.root / "folder").is_dir()
        paused_state = agent.last_auto_state
        monkeypatch.setenv(auto_preset.AUTO_MAX_FILES_TOUCHED_ENV, "100")
        preset.resume(paused_state["run_id"])

    expected_scripts = ["initial.py", "repeat.py"]
    if case != "missing_evidence":
        expected_scripts.append("extra.py")
    if case == "retry":
        expected_scripts.append("initial.py")
    assert executed == expected_scripts
    assert not (workspace.root / "excess").exists()
    assert (workspace.root / "data").read_text() == ("first" if case == "retry" else "second")
    assert paused_state["execution_metrics"]["files_touched"] == 2
    assert agent.last_auto_state["execution_metrics"]["files_touched"] == (
        2 if case == "retry" else None
    )
