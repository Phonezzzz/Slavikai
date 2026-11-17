from __future__ import annotations

import asyncio

import pytest

from core.approval_policy import ApprovalContext
from core.plan_execute import (
    PlanExecutionBindingError,
    PlanLoopApprovalPending,
    PlanLoopCheckpoints,
    PlanLoopContinuationUnavailable,
    execute_approved_plan,
    resume_approved_plan,
    validate_approved_plan,
)
from core.run_contract import build_run_spec
from core.run_outcome import TerminalReason, VerificationOutcome
from core.tool_gateway import ToolGateway
from llm.types import LLMResult, ToolCall
from server.http.common.workflow_state import plan_hash_payload
from shared.models import JSONValue, ToolResult
from tools.tool_registry import ToolRegistry


class ScriptedBrain:
    """Brain, выдающий заранее заданные вызовы, затем финальный текст."""

    def __init__(self, calls: list[tuple[str, dict[str, JSONValue]]]) -> None:
        self._calls = list(calls)
        self.generate_calls = 0

    def generate(self, messages, config=None, tools=None):
        del messages, config, tools
        self.generate_calls += 1
        if self._calls:
            name, args = self._calls.pop(0)
            return LLMResult(
                text="",
                tool_calls=[
                    ToolCall(
                        id=f"call-{self.generate_calls}",
                        name=name,
                        arguments=dict(args),
                    )
                ],
            )
        return LLMResult(text="Готово")


class ToggleBrain:
    """Brain для resume-теста: сначала утверждённый вызов, потом ошибка."""

    def __init__(self, call: tuple[str, dict[str, JSONValue]]) -> None:
        self._call = call
        self.fail_generate = False
        self.generate_calls = 0

    def generate(self, messages, config=None, tools=None):
        del messages, config, tools
        self.generate_calls += 1
        if self.fail_generate:
            raise RuntimeError("provider boom")
        name, args = self._call
        return LLMResult(
            text="",
            tool_calls=[
                ToolCall(
                    id=f"call-{self.generate_calls}",
                    name=name,
                    arguments=dict(args),
                )
            ],
        )


class FakePlanAgent:
    """Минимальный агент для executor: тот же ToolGateway-путь, что у Agent."""

    def __init__(
        self,
        *,
        brain: ScriptedBrain,
        tools: list[tuple[str, object, str, list[str]]] | None = None,
    ) -> None:
        self.tool_registry = ToolRegistry()
        self.main_config = None
        self._brain = brain
        self.runtime_mode = "ask"
        self.runtime_active_plan: dict[str, JSONValue] | None = None
        self.runtime_active_task: dict[str, JSONValue] | None = None
        self.runtime_auto_state: dict[str, JSONValue] | None = None
        self.runtime_plan_guard_enabled = False
        self.runtime_workspace_root: str | None = None
        # Как в UI runner при профиле sandbox: safe_mode включён.
        self.tools_enabled: dict[str, bool] = {"safe_mode": True}
        self._approved_categories: set[str] = set()
        self._session_id: str | None = None
        self.recorded_specs = []
        for name, handler, capability, risk_classes in tools or []:
            self.tool_registry.register(
                name,
                handler,  # type: ignore[arg-type]
                capability=capability,
                risk_classes=risk_classes,
                description=f"test tool {name}",
                parameters_schema={"type": "object"},
            )

    def set_session_context(self, session_id, approved_categories) -> None:
        self._session_id = session_id
        self._approved_categories = set(approved_categories)

    def set_runtime_state(
        self,
        *,
        mode,
        active_plan=None,
        active_task=None,
        auto_state=None,
        enforce_plan_guard=False,
    ) -> None:
        self.runtime_mode = mode
        self.runtime_active_plan = active_plan
        self.runtime_active_task = active_task
        self.runtime_auto_state = auto_state
        self.runtime_plan_guard_enabled = enforce_plan_guard

    def _build_tool_gateway(self, *, budget=None):
        spec = build_run_spec(
            mode=self.runtime_mode,
            active_plan=self.runtime_active_plan,
            active_task=self.runtime_active_task,
            enforce_plan_guard=self.runtime_plan_guard_enabled,
            budget=budget,
        )
        self.recorded_specs.append(spec)
        return ToolGateway(
            self.tool_registry,
            approval_context=ApprovalContext(
                safe_mode=bool(self.tools_enabled.get("safe_mode", False)),
                session_id=self._session_id,
                approved_categories=set(self._approved_categories),
                execution_target="sandbox",
            ),
            run_policy=spec.policy,
        )

    def _get_main_brain(self):
        return self._brain


def _step(
    step_id: str,
    tool: str,
    args: dict[str, JSONValue],
    *,
    description: str = "Проверить",
) -> dict[str, JSONValue]:
    return {
        "step_id": step_id,
        "title": f"Шаг {step_id}",
        "description": description,
        "allowed_tool_kinds": [tool],
        "inputs": {"operation": tool, "tool_args": dict(args)},
        "expected_outputs": [],
        "acceptance_checks": [],
        "status": "todo",
        "evidence": None,
    }


def _approved_plan(
    steps: list[dict[str, JSONValue]],
    *,
    plan_id: str = "plan-1",
    revision: int = 1,
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    plan: dict[str, JSONValue] = {
        "plan_id": plan_id,
        "goal": "Тестовый план",
        "status": "approved",
        "plan_revision": revision,
        "steps": steps,
    }
    plan["plan_hash"] = plan_hash_payload(plan)
    task: dict[str, JSONValue] = {
        "plan_id": plan_id,
        "plan_hash": plan["plan_hash"],
        "task_id": "task-1",
        "status": "running",
        # Ревизия в реальном task не на верхнем уровне, а в контексте
        # task_packet, скомпилированного из плана при execute.
        "task_packet": {
            "context": {
                "plan_id": plan_id,
                "plan_revision": revision,
                "source": "plan_execute",
            },
        },
    }
    return plan, task


def _read_agent(*, brain: ScriptedBrain) -> tuple[FakePlanAgent, list]:
    calls: list = []

    def workspace_list(request):
        calls.append(request)
        return ToolResult.success({"files": ["a.py"]})

    agent = FakePlanAgent(
        brain=brain,
        tools=[("workspace_list", workspace_list, "read", [])],
    )
    return agent, calls


def test_execute_approved_plan_runs_read_step_through_canonical_loop() -> None:
    brain = ScriptedBrain([("workspace_list", {"path": "."})])
    agent, calls = _read_agent(brain=brain)
    plan, task = _approved_plan([_step("step-1", "workspace_list", {"path": "."})])

    result = execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
    )

    assert result.outcome.is_full_done is True
    assert result.outcome.terminal_reason is TerminalReason.COMPLETED
    assert result.outcome.verification.outcome is VerificationOutcome.NOT_APPLICABLE
    assert len(result.steps) == 1
    assert result.steps[0].status == "done"
    assert result.tool_calls_used == 1
    assert len(calls) == 1
    assert calls[0].name == "workspace_list"
    assert calls[0].args == {"path": "."}
    # Тот же gateway-путь, что прямой execute: intent=execute, mode=act, guard on.
    assert agent.recorded_specs
    spec = agent.recorded_specs[-1]
    assert spec.intent == "execute"
    assert spec.target == "sandbox"
    assert spec.policy.mode == "act"
    assert spec.policy.enforce_plan_guard is True
    assert agent.runtime_active_task is not None
    assert agent.runtime_active_task.get("current_step_id") == "step-1"


def test_execute_approved_plan_gateway_matches_direct_execute() -> None:
    brain = ScriptedBrain([("workspace_list", {"path": "."})])
    agent, _ = _read_agent(brain=brain)
    plan, task = _approved_plan([_step("step-1", "workspace_list", {"path": "."})])

    execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
    )

    plan_spec = agent.recorded_specs[-1]
    # Прямой execute с тем же runtime state строит ту же политику.
    agent.set_runtime_state(
        mode="act",
        active_plan=plan,
        active_task={
            "plan_id": plan.get("plan_id"),
            "plan_hash": plan.get("plan_hash"),
            "current_step_id": "step-1",
        },
        enforce_plan_guard=True,
    )
    agent._build_tool_gateway()
    direct_spec = agent.recorded_specs[-1]
    assert direct_spec.intent == plan_spec.intent == "execute"
    assert direct_spec.policy == plan_spec.policy


def test_tampered_plan_hash_rejected() -> None:
    brain = ScriptedBrain([("workspace_list", {"path": "."})])
    agent, calls = _read_agent(brain=brain)
    plan, task = _approved_plan([_step("step-1", "workspace_list", {"path": "."})])
    plan["steps"][0]["description"] = "Подменённое описание"

    with pytest.raises(PlanExecutionBindingError, match="plan_hash"):
        execute_approved_plan(
            agent=agent,
            plan=plan,
            task=task,
            session_id="s1",
            plan_hash_fn=plan_hash_payload,
            checkpoints=PlanLoopCheckpoints(),
        )
    assert calls == []


def test_plan_revision_growth_after_compile_is_allowed() -> None:
    # Ревизия — CAS-счётчик: она растёт при каждом обновлении статуса
    # шага, поэтому пакет, скомпилированный из более старой ревизии, —
    # норма, а не подмена. Подмена ловится хешем и пакетом из будущего.
    brain = ScriptedBrain([("workspace_list", {"path": "."})])
    agent, calls = _read_agent(brain=brain)
    plan, task = _approved_plan([_step("step-1", "workspace_list", {"path": "."})])
    plan["plan_revision"] = 2

    result = execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
    )
    assert result.outcome.is_full_done is True
    assert len(calls) == 1


def test_tampered_packet_revision_rejected() -> None:
    brain = ScriptedBrain([("workspace_list", {"path": "."})])
    agent, calls = _read_agent(brain=brain)
    plan, task = _approved_plan([_step("step-1", "workspace_list", {"path": "."})])
    packet = task["task_packet"]
    assert isinstance(packet, dict)
    context = packet["context"]
    assert isinstance(context, dict)
    context["plan_revision"] = 2

    with pytest.raises(PlanExecutionBindingError, match="plan_binding_mismatch"):
        execute_approved_plan(
            agent=agent,
            plan=plan,
            task=task,
            session_id="s1",
            plan_hash_fn=plan_hash_payload,
            checkpoints=PlanLoopCheckpoints(),
        )
    assert calls == []


def test_tampered_task_hash_rejected_by_validate() -> None:
    plan, task = _approved_plan([_step("step-1", "workspace_list", {"path": "."})])
    task["plan_hash"] = "0" * 64

    with pytest.raises(PlanExecutionBindingError, match="plan_binding_mismatch"):
        validate_approved_plan(plan=plan, task=task, plan_hash_fn=plan_hash_payload)


def test_tool_outside_step_is_blocked_by_plan_guard() -> None:
    write_calls: list = []

    def workspace_write(request):
        write_calls.append(request)
        return ToolResult.success({})

    brain = ScriptedBrain([("workspace_write", {"path": "x.txt"})])
    agent = FakePlanAgent(
        brain=brain,
        tools=[
            ("workspace_list", lambda request: ToolResult.success({}), "read", []),
            ("workspace_write", workspace_write, "write", []),
        ],
    )
    plan, task = _approved_plan([_step("step-1", "workspace_list", {"path": "."})])

    result = execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
    )

    assert result.outcome.is_full_done is False
    # tool_policy_denied мапится outcome-модулем в ABORTED, а не FAILED.
    assert result.outcome.terminal_reason is TerminalReason.ABORTED
    assert result.steps[0].status == "failed"
    assert write_calls == []


def test_approval_pause_and_resume() -> None:
    net_calls: list = []

    def net_read(request):
        net_calls.append(request)
        return ToolResult.success({"output": "observed"})

    brain = ScriptedBrain([("net_read", {"index": 1})])
    agent = FakePlanAgent(
        brain=brain,
        tools=[("net_read", net_read, "read", ["network"])],
    )
    plan, task = _approved_plan([_step("step-1", "net_read", {"index": 1})])
    checkpoints = PlanLoopCheckpoints()
    # Runner выставляет session context до запуска (как run_plan_runner).
    agent.set_session_context("s1", set())

    with pytest.raises(PlanLoopApprovalPending) as excinfo:
        execute_approved_plan(
            agent=agent,
            plan=plan,
            task=task,
            session_id="s1",
            plan_hash_fn=plan_hash_payload,
            checkpoints=checkpoints,
        )
    assert net_calls == []
    assert len(checkpoints) == 1
    checkpoint = excinfo.value.checkpoint
    assert checkpoints.peek(checkpoint.checkpoint_id) is checkpoint
    assert checkpoint.step_id == "step-1"

    # Runner одобряет категорию (как approve_session) и продолжает с паузы.
    agent.set_session_context("s1", {"NETWORK_RISK"})
    taken = checkpoints.take(checkpoint.checkpoint_id)
    assert taken is checkpoint
    result = resume_approved_plan(
        agent=agent,
        checkpoint=checkpoint,
        plan=plan,
        task=task,
        plan_hash_fn=plan_hash_payload,
        checkpoints=checkpoints,
    )

    assert result.outcome.is_full_done is True
    assert len(net_calls) == 1
    assert net_calls[0].args == {"index": 1}
    assert len(checkpoints) == 0


def test_resume_rejects_edited_plan_content() -> None:
    net_calls: list = []

    def net_read(request):
        net_calls.append(request)
        return ToolResult.success({"output": "observed"})

    brain = ScriptedBrain([("net_read", {"index": 1})])
    agent = FakePlanAgent(
        brain=brain,
        tools=[("net_read", net_read, "read", ["network"])],
    )
    plan, task = _approved_plan([_step("step-1", "net_read", {"index": 1})])
    checkpoints = PlanLoopCheckpoints()

    with pytest.raises(PlanLoopApprovalPending) as excinfo:
        execute_approved_plan(
            agent=agent,
            plan=plan,
            task=task,
            session_id="s1",
            plan_hash_fn=plan_hash_payload,
            checkpoints=checkpoints,
        )
    checkpoint = excinfo.value.checkpoint

    # План отредактирован после паузы: контент шага изменён, хеш
    # пересчитан, пакет перекомпилирован из новой ревизии.
    steps = plan["steps"]
    assert isinstance(steps, list)
    step = steps[0]
    assert isinstance(step, dict)
    step["description"] = "Проверить совсем другое"
    plan["plan_revision"] = 2
    plan["plan_hash"] = plan_hash_payload(plan)
    task["plan_hash"] = plan["plan_hash"]
    packet = task["task_packet"]
    assert isinstance(packet, dict)
    context = packet["context"]
    assert isinstance(context, dict)
    context["plan_revision"] = 2

    with pytest.raises(PlanLoopContinuationUnavailable):
        resume_approved_plan(
            agent=agent,
            checkpoint=checkpoint,
            plan=plan,
            task=task,
            plan_hash_fn=plan_hash_payload,
            checkpoints=checkpoints,
        )
    assert net_calls == []
    assert checkpoints.peek(checkpoint.checkpoint_id) is checkpoint


def test_write_step_without_postcondition_is_not_full_done() -> None:
    # E05: до postcondition-модели write-шаг не может дать DONE.
    write_calls: list = []

    def workspace_write(request):
        write_calls.append(request)
        return ToolResult.success({})

    brain = ScriptedBrain([("workspace_write", {"path": "x.txt"})])
    agent = FakePlanAgent(
        brain=brain,
        tools=[("workspace_write", workspace_write, "write", [])],
    )
    agent.set_session_context("s1", {"FS_DELETE_OVERWRITE"})
    plan, task = _approved_plan([_step("step-1", "workspace_write", {"path": "x.txt"})])

    result = execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
    )

    assert len(write_calls) == 1
    assert result.steps[0].status == "done"
    assert result.outcome.is_full_done is False
    assert result.outcome.terminal_reason is TerminalReason.COMPLETED
    assert result.outcome.verification.outcome is VerificationOutcome.UNKNOWN


def test_cancelled_plan_returns_cancelled_outcome() -> None:
    brain = ScriptedBrain([("workspace_list", {"path": "."})])
    agent, calls = _read_agent(brain=brain)
    plan, task = _approved_plan([_step("step-1", "workspace_list", {"path": "."})])
    token = asyncio.Event()
    token.set()

    result = execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
        cancellation_token=token,
    )

    assert result.cancelled is True
    assert result.outcome.terminal_reason is TerminalReason.CANCELLED
    assert result.steps == ()
    assert calls == []


def test_plan_step_args_mismatch_blocked_by_plan_guard() -> None:
    # Finding 3 (P1): модель подменяет аргументы при том же имени
    # инструмента — guard обязан сверить args с утверждёнными в шаге.
    write_calls: list = []

    def workspace_write(request):
        write_calls.append(request)
        return ToolResult.success({})

    brain = ScriptedBrain([("workspace_write", {"path": "unapproved.txt", "content": "wrong"})])
    agent = FakePlanAgent(
        brain=brain,
        tools=[("workspace_write", workspace_write, "write", [])],
    )
    agent.set_session_context("s1", {"FS_DELETE_OVERWRITE"})
    plan, task = _approved_plan(
        [_step("step-1", "workspace_write", {"path": "approved.txt", "content": "right"})]
    )

    result = execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
    )

    assert write_calls == []
    assert len(result.steps) == 1
    assert result.steps[0].status == "failed"
    assert result.outcome.terminal_reason is TerminalReason.FAILED
    assert "BLOCKED_OUTSIDE_PLAN" in result.steps[0].result


def test_resume_loop_error_after_executed_call_does_not_replay() -> None:
    # Finding 1 (P1): resume после approval выполняет утверждённый вызов,
    # а следующий запрос к модели падает. Ошибка loop не должна
    # выбрасываться наружу с живым чекпоинтом: выполненный prefix
    # сохраняется как failed-шаг, план FAILED, повтор не воспроизводит
    # уже выполненный вызов.
    net_calls: list = []

    def net_read(request):
        net_calls.append(request)
        return ToolResult.success({"output": "observed"})

    brain = ToggleBrain(("net_read", {"index": 1}))
    agent = FakePlanAgent(
        brain=brain,
        tools=[("net_read", net_read, "read", ["network"])],
    )
    plan, task = _approved_plan([_step("step-1", "net_read", {"index": 1})])
    checkpoints = PlanLoopCheckpoints()
    agent.set_session_context("s1", set())

    with pytest.raises(PlanLoopApprovalPending) as excinfo:
        execute_approved_plan(
            agent=agent,
            plan=plan,
            task=task,
            session_id="s1",
            plan_hash_fn=plan_hash_payload,
            checkpoints=checkpoints,
        )
    checkpoint = excinfo.value.checkpoint
    assert net_calls == []

    agent.set_session_context("s1", {"NETWORK_RISK"})
    brain.fail_generate = True
    taken = checkpoints.take(checkpoint.checkpoint_id)
    assert taken is checkpoint
    result = resume_approved_plan(
        agent=agent,
        checkpoint=checkpoint,
        plan=plan,
        task=task,
        plan_hash_fn=plan_hash_payload,
        checkpoints=checkpoints,
    )

    assert result.outcome.terminal_reason is TerminalReason.FAILED
    assert result.cancelled is False
    assert len(net_calls) == 1
    assert len(result.steps) == 1
    assert result.steps[0].status == "failed"
    assert result.steps[0].tool_calls_used == 1
    assert result.tool_calls_used == 1


def test_plan_loop_runs_inside_verified_workspace_context(tmp_path) -> None:
    # Finding 2 (P1): loop шага выполняется в контексте проверенного
    # workspace, а не глобального sandbox/project.
    seen_roots: list = []

    def workspace_list(request):
        from tools.workspace_tools import get_workspace_root

        seen_roots.append(get_workspace_root())
        return ToolResult.success({"files": []})

    brain = ScriptedBrain([("workspace_list", {"path": "."})])
    agent = FakePlanAgent(
        brain=brain,
        tools=[("workspace_list", workspace_list, "read", [])],
    )
    agent.runtime_workspace_root = str(tmp_path)
    plan, task = _approved_plan([_step("step-1", "workspace_list", {"path": "."})])

    result = execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
    )

    assert result.outcome.is_full_done is True
    assert seen_roots == [tmp_path.resolve()]


def test_cancelled_step_keeps_executed_calls() -> None:
    # Finding 4 (P2): отмена после выполненной записи не должна терять
    # вызов: частичный шаг сохраняется как failed с фактическим числом.
    token = asyncio.Event()
    write_calls: list = []

    def workspace_write(request):
        write_calls.append(request)
        token.set()
        return ToolResult.success({})

    brain = ScriptedBrain([("workspace_write", {"path": "x.txt"})])
    agent = FakePlanAgent(
        brain=brain,
        tools=[("workspace_write", workspace_write, "write", [])],
    )
    agent.set_session_context("s1", {"FS_DELETE_OVERWRITE"})
    plan, task = _approved_plan([_step("step-1", "workspace_write", {"path": "x.txt"})])

    result = execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
        cancellation_token=token,
    )

    assert result.cancelled is True
    assert result.outcome.terminal_reason is TerminalReason.CANCELLED
    assert len(write_calls) == 1
    assert len(result.steps) == 1
    assert result.steps[0].status == "failed"
    assert result.steps[0].tool_calls_used == 1
    assert result.tool_calls_used == 1


def test_binding_error_after_executed_step_keeps_prefix() -> None:
    # Finding (P2, re-review): инструмент второго шага отключён после
    # утверждения плана. Выполненный первый шаг не должен теряться:
    # терминальный результат с done-prefix и failed текущим шагом.
    write_calls: list = []

    def workspace_write(request):
        write_calls.append(request)
        return ToolResult.success({})

    brain = ScriptedBrain([("workspace_write", {"path": "x.txt"})])
    agent = FakePlanAgent(
        brain=brain,
        tools=[("workspace_write", workspace_write, "write", [])],
    )
    agent.set_session_context("s1", {"FS_DELETE_OVERWRITE"})
    plan, task = _approved_plan(
        [
            _step("step-1", "workspace_write", {"path": "x.txt"}),
            _step("step-2", "workspace_read", {"path": "x.txt"}),
        ]
    )

    result = execute_approved_plan(
        agent=agent,
        plan=plan,
        task=task,
        session_id="s1",
        plan_hash_fn=plan_hash_payload,
        checkpoints=PlanLoopCheckpoints(),
    )

    assert len(write_calls) == 1
    assert result.outcome.terminal_reason is TerminalReason.FAILED
    assert result.cancelled is False
    assert len(result.steps) == 2
    assert result.steps[0].status == "done"
    assert result.steps[0].tool_calls_used == 1
    assert result.steps[1].status == "failed"
    assert result.steps[1].operation == "workspace_read"
    assert result.steps[1].tool_calls_used == 0
    assert "plan_step_tool_unavailable" in result.steps[1].result
    assert result.tool_calls_used == 1


def test_binding_error_before_any_effect_propagates() -> None:
    # Обратная сторона: отказ привязки до любого эффекта не превращается
    # в терминальный результат — пробрасывается наружу, чтобы resume
    # сохранил чекпоинт (round-1 P1), а свежий запуск упал общей ошибкой.
    brain = ScriptedBrain([])
    agent = FakePlanAgent(brain=brain, tools=[])
    plan, task = _approved_plan([_step("step-1", "workspace_read", {"path": "x.txt"})])

    with pytest.raises(PlanExecutionBindingError, match="plan_step_tool_unavailable"):
        execute_approved_plan(
            agent=agent,
            plan=plan,
            task=task,
            session_id="s1",
            plan_hash_fn=plan_hash_payload,
            checkpoints=PlanLoopCheckpoints(),
        )
