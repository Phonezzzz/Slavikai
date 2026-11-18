"""Regression E2E canonical loop (E01-T09, gate перед E02).

Первый немоканный acceptance test канонического цикла: scripted brain
(fixture модели, разрешённая тикетом) вызывает настоящие read/write
tools через настоящие ToolRegistry + ToolGateway + AgentToolLoop,
observation возвращается модели, а исход выводится канонической
деривацией ``RunOutcome``.

Ветви acceptance:
- падающий tool -> FAILED: preset возвращает FAILED_WORKER через
  fail-closed gate ``_has_failed_tool_call``; голая деривация чистого
  loop-результата при этом дала бы completed/unverified — именно
  поэтому gate существует и именно его ловит этот тест;
- нет verifier -> UNVERIFIED, не DONE: completed + full +
  verification unknown, final не публикуется;
- подтверждённый postcondition -> DONE: completed + full +
  verification passed (настоящая перепроверка чтением файла через тот
  же gateway), приложенная к деривации ``outcome_from_loop_result``:
  phase / terminal_reason / disposition вычисляет production-код,
  а не сам тест; ``is_full_done``, final_success.
- preset + canonical verifier -> DONE: тот же сценарий через
  ``AutoPresetRunner`` с настоящим canonical verifier (``make check``
  проверяет result.txt); DONE берётся из ``canonical_outcome`` раннера,
  а не из проверки в самом тесте.

Замокано только одно: сама модель (scripted Brain). Loop, gateway,
registry, tools, workspace, postcondition-check и деривация outcome —
настоящие, без стабов.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from core import auto_preset
from core.approval_policy import ApprovalContext
from core.auto_preset import AutoPresetRunner
from core.managed_workspace import ManagedWorkspace, WorkspaceAccess
from core.mwv.models import StopReasonCode, VerificationResult, VerificationStatus
from core.response_preferences import ResponsePreferences
from core.run_outcome import (
    Disposition,
    PostconditionCheck,
    RunPhase,
    TerminalReason,
    Verification,
    VerificationOutcome,
    outcome_from_loop_result,
)
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ToolCall
from shared.auto_models import AutoRunStatus
from shared.models import LLMMessage, ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import ReadFileTool, WriteFileTool, workspace_root_context

SOURCE_CONTENT = "исходные данные: 41\n"
EXPECTED_CONTENT = "исходные данные: 41\nобработано\n"
GOAL = "прочитай source.txt и запиши result.txt с той же строкой и строкой 'обработано'"


class _ReadWriteBrain(Brain):
    """Fixture модели: read -> write из observation -> ответ."""

    supports_native_tools = True

    def __init__(self) -> None:
        self.calls = 0
        self.observed_source: str | None = None

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del config, tools
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                usage=LLMUsage(0, 0, 0),
                text="читаю исходник",
                tool_calls=[
                    ToolCall(id="read-1", name="workspace_read", arguments={"path": "source.txt"})
                ],
            )
        assert messages[-1].role == "tool"
        if self.calls == 2:
            projection = json.loads(messages[-1].content)
            assert projection["ok"] is True
            self.observed_source = projection["data"]["output"]
            return LLMResult(
                usage=LLMUsage(0, 0, 0),
                text="",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="workspace_write",
                        arguments={
                            "path": "result.txt",
                            "content": self.observed_source + "обработано\n",
                        },
                    )
                ],
            )
        return LLMResult(usage=LLMUsage(0, 0, 0), text="Готово")


class _FailingWriteBrain(Brain):
    """Fixture модели: запись вне workspace -> отказ -> ответ."""

    supports_native_tools = True

    def __init__(self) -> None:
        self.calls = 0
        self.observed_error: str | None = None

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del config, tools
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                usage=LLMUsage(0, 0, 0),
                text="пишу наружу",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="workspace_write",
                        arguments={"path": "../escape.txt", "content": "x"},
                    )
                ],
            )
        assert messages[-1].role == "tool"
        projection = json.loads(messages[-1].content)
        assert projection["ok"] is False
        self.observed_error = projection["error"]
        return LLMResult(usage=LLMUsage(0, 0, 0), text="не удалось записать")


class _FakeTracer:
    def log(self, event_type: str, message: str, meta=None) -> None:  # noqa: ANN001
        del event_type, message, meta


class _E2EAgent:
    """Минимальный фасад агента для AutoPresetRunner (тот же набор
    атрибутов, что у _FakeAgent в tests/test_auto_preset.py)."""

    response_preferences = ResponsePreferences()

    def __init__(self, brain: Brain, registry: ToolRegistry, workspace: ManagedWorkspace) -> None:
        self._brain = brain
        self.main_config = None
        self.session_id = "session-test"
        self.user_id = "test-principal"
        self.runtime_mode = "auto"
        self.runtime_workspace_root = None
        self.tools_enabled = {"safe_mode": True}
        self.approved_categories: set[str] = set()
        self.tracer = _FakeTracer()
        self.last_auto_state = None
        self.tool_registry = registry
        self._workspace = workspace

    def _get_main_brain(self):  # noqa: ANN202
        return self._brain

    def _build_tool_gateway(self, *, budget=None):  # noqa: ANN001, ANN202
        del budget
        return _build_gateway(self.tool_registry, self._workspace)

    def _format_stop_response(self, **kwargs):  # noqa: ANN003, ANN202
        return f"stop:{kwargs.get('what', '')}"

    def _append_report_block(self, text: str, **kwargs):  # noqa: ANN003
        del kwargs
        return text


def _build_e2e(tmp_path: Path) -> tuple[ManagedWorkspace, ToolRegistry]:
    # ManagedWorkspace.create выделяет собственный подкаталог внутри
    # tmp_path: workspace.root — это он, а не сам tmp_path.
    workspace = ManagedWorkspace.create(tmp_path, principal_id="test-principal")
    (workspace.root / "source.txt").write_text(SOURCE_CONTENT, encoding="utf-8")
    registry = ToolRegistry()
    registry.register(
        "workspace_read",
        ReadFileTool().handle,
        description="Read a workspace file",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        # capability="read" обязателен: gateway учитывает file effects
        # только для write/exec. Read с capability по умолчанию ("exec")
        # помечает эффекты неизвестными, и последующий write через
        # preset отклоняется (workspace_file_effects_unknown).
        capability="read",
    )
    registry.register(
        "workspace_write",
        WriteFileTool().handle,
        description="Write a workspace file",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        capability="write",
    )
    return workspace, registry


def _build_gateway(registry: ToolRegistry, workspace: ManagedWorkspace) -> ToolGateway:
    return ToolGateway(
        registry,
        approval_context=ApprovalContext(False, "session-test", set()),
        workspace_access=WorkspaceAccess(workspace, "test-principal", "session-test"),
    )


def _run_loop(brain: Brain, gateway: ToolGateway, registry: ToolRegistry):  # noqa: ANN202
    specs = registry.list_tool_specs(policy=gateway.run_policy)
    return AgentToolLoop(max_iterations=8).run(
        brain=brain,
        gateway=gateway,
        messages=[
            LLMMessage(role="system", content="Выполни задачу инструментами."),
            LLMMessage(role="user", content=GOAL),
        ],
        tools=specs,
        config=None,
    )


def _check_result_file(gateway: ToolGateway, expected: str) -> Verification:
    """Настоящий postcondition: перечитать result.txt через тот же
    gateway и сравнить содержимое с ожидаемым. Не чтение Path
    напрямую: проверяется тот же инструмент, что писал модель."""
    result: ToolResult = gateway.call(
        ToolRequest(name="workspace_read", args={"path": "result.txt"})
    )
    actual = result.data.get("output") if result.ok else None
    passed = result.ok and actual == expected
    detail = f"expected={expected!r}, actual={actual!r}" if not passed else ""
    return Verification(
        outcome=VerificationOutcome.PASSED if passed else VerificationOutcome.FAILED,
        checks=(PostconditionCheck(name="result.txt content", passed=passed, detail=detail),),
        source="workspace_read result.txt",
    )


def test_e2e_missing_verifier_is_unverified_not_done(tmp_path) -> None:
    workspace, registry = _build_e2e(tmp_path)
    gateway = _build_gateway(registry, workspace)
    brain = _ReadWriteBrain()
    with workspace_root_context(workspace.root):
        loop_result = _run_loop(brain, gateway, registry)

    assert loop_result.error is None
    assert [call.call.name for call in loop_result.tool_calls] == [
        "workspace_read",
        "workspace_write",
    ]
    assert all(call.result.ok for call in loop_result.tool_calls)
    assert (workspace.root / "result.txt").read_text(encoding="utf-8") == EXPECTED_CONTENT
    # Observation реально дошла до модели: write сделан из прочитанного.
    assert brain.observed_source == SOURCE_CONTENT

    # Verifier нет (не репозиторий, canonical verifier отсутствует):
    # чистый loop даёт completed, но без подтверждённого postcondition
    # это UNVERIFIED, а не DONE.
    outcome = outcome_from_loop_result(loop_result, state_changing=True)
    assert outcome.phase is RunPhase.TERMINAL
    assert outcome.terminal_reason is TerminalReason.COMPLETED
    assert outcome.disposition is Disposition.FULL
    assert outcome.verification.outcome is VerificationOutcome.UNKNOWN
    assert outcome.ui_label == "UNVERIFIED"
    assert outcome.is_full_done is False
    assert outcome.communication_class is None


def test_e2e_confirmed_postcondition_is_done(tmp_path) -> None:
    workspace, registry = _build_e2e(tmp_path)
    gateway = _build_gateway(registry, workspace)
    brain = _ReadWriteBrain()
    with workspace_root_context(workspace.root):
        loop_result = _run_loop(brain, gateway, registry)
        verification = _check_result_file(gateway, EXPECTED_CONTENT)

    assert loop_result.error is None
    assert verification.outcome is VerificationOutcome.PASSED
    assert verification.checks[0].passed is True

    # Исход — через production-деривацию, с приложенной проверкой:
    # если runtime перестанет учитывать verification, is_full_done /
    # ui_label / final это покажут — DONE требует verification=passed
    # именно от деривации, а не от конструктора, собранного тестом.
    outcome = outcome_from_loop_result(loop_result, state_changing=True, verification=verification)
    assert outcome.verification is verification
    assert outcome.is_full_done is True
    assert outcome.ui_label == "DONE"
    assert outcome.communication_class == "final_success"
    assert outcome.describe().startswith("completed + full, verification=passed")


def test_e2e_failing_tool_is_failed_not_done(tmp_path, monkeypatch) -> None:
    workspace, registry = _build_e2e(tmp_path)
    gateway = _build_gateway(registry, workspace)
    loop_brain = _FailingWriteBrain()
    with workspace_root_context(workspace.root):
        loop_result = _run_loop(loop_brain, gateway, registry)

    # Сам loop завершился чисто: отказ tool — это observation, а не
    # ошибка loop. Поэтому голая деривация дала бы completed/unverified,
    # и именно fail-closed gate preset'а обязан превратить это в FAILED.
    assert loop_result.error is None
    assert len(loop_result.tool_calls) == 1
    assert loop_result.tool_calls[0].result.ok is False
    assert loop_brain.observed_error is not None
    assert "вне рабочей директории" in loop_brain.observed_error
    assert not (workspace.root / "escape.txt").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert not (workspace.root / "result.txt").exists()

    monkeypatch.chdir(workspace.root)
    preset_brain = _FailingWriteBrain()
    agent = _E2EAgent(preset_brain, registry, workspace)
    with workspace_root_context(workspace.root):
        outcome = AutoPresetRunner(agent, workspace_root=workspace.root).run(GOAL)

    assert outcome.status == AutoRunStatus.FAILED_WORKER
    assert outcome.stop_reason_code == StopReasonCode.WORKER_FAILED
    assert outcome.verifier is None
    assert agent.last_auto_state["error_code"] == "tool_failure_unrecovered"


def test_e2e_preset_run_completes_without_fake_verification(tmp_path, monkeypatch) -> None:
    workspace, registry = _build_e2e(tmp_path)
    monkeypatch.chdir(workspace.root)
    brain = _ReadWriteBrain()
    agent = _E2EAgent(brain, registry, workspace)
    with workspace_root_context(workspace.root):
        outcome = AutoPresetRunner(agent, workspace_root=workspace.root).run(GOAL)

    assert (workspace.root / "result.txt").read_text(encoding="utf-8") == EXPECTED_CONTENT
    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.stop_reason_code is None
    # Не репозиторий — canonical verifier отсутствует, и preset честно
    # не заявляет никакой верификации вместо него.
    assert outcome.verifier is None
    state = agent.last_auto_state
    assert state["status"] == AutoRunStatus.COMPLETED.value
    assert state["execution_metrics"]["tool_calls_used"] == 2
    assert state["execution_metrics"]["files_touched"] == 1
    assert state["verifier"] is None


class _ExecutingVerifierRuntime:
    """Test-only VerifierRuntime-compatible executor.

    Production VerifierRuntime.run() never executes the check command:
    it returns sandbox_verification_unavailable (ERROR) by design, so
    no repo-workspace run can pass verification through it today. This
    double keeps the whole production verification path — the task
    packet with the canonical command, cwd = run root, the
    VerificationResult mapping, the FAILED_VERIFIER gate — and only
    replaces the unimplemented execution backend with a real
    subprocess run of the same command. It returns canned nothing:
    PASSED iff the command exits 0.
    """

    def __init__(self, project_root):  # noqa: ANN001
        del project_root

    def run(self, task, context):  # noqa: ANN001
        del context
        command = list(task.verifier["command"])
        cwd = task.verifier["cwd"]
        started = time.monotonic()
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
        duration = time.monotonic() - started
        passed = proc.returncode == 0
        return VerificationResult(
            status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
            command=command,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            duration_seconds=duration,
            error=None if passed else "canonical check failed",
            fail_type=None if passed else "check_failed",
            excerpt=(proc.stdout + proc.stderr)[-500:],
            verifier_profile="canonical",
        )


def _build_verified_e2e(tmp_path: Path) -> tuple[ManagedWorkspace, ToolRegistry]:
    """E2E workspace with a real canonical verifier: a Makefile whose
    ``check`` target verifies result.txt — the file the tool loop
    creates. has_canonical_repo_verifier() qualifies the workspace
    (Makefile present + make installed), so _verify_after_loop takes
    the real VerifierRuntime path instead of returning None.
    """
    workspace, registry = _build_e2e(tmp_path)
    (workspace.root / "Makefile").write_text(
        "check:\n"
        '\t@test -f result.txt && grep -qx "обработано" result.txt '
        '&& grep -qx "исходные данные: 41" result.txt\n',
        encoding="utf-8",
    )
    return workspace, registry


class _WrongContentBrain(Brain):
    """Reads source.txt, writes result.txt with the wrong content: the
    canonical verifier must catch it."""

    supports_native_tools = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del config, tools
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                usage=LLMUsage(0, 0, 0),
                text="читаю исходник",
                tool_calls=[
                    ToolCall(id="read-1", name="workspace_read", arguments={"path": "source.txt"})
                ],
            )
        if self.calls == 2:
            return LLMResult(
                usage=LLMUsage(0, 0, 0),
                text="",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="workspace_write",
                        arguments={"path": "result.txt", "content": "не то\n"},
                    )
                ],
            )
        return LLMResult(usage=LLMUsage(0, 0, 0), text="Готово")


def test_e2e_preset_run_with_canonical_verifier_is_done(tmp_path, monkeypatch) -> None:
    """DONE comes from the runner's canonical outcome, not from a
    test-side check: the preset runs the real canonical verifier
    (make check over result.txt) and the attached verification flows
    into outcome_from_loop_result. _check_result_file() is deliberately
    NOT called here: it would prove nothing about the runner.
    """
    workspace, registry = _build_verified_e2e(tmp_path)
    monkeypatch.chdir(workspace.root)
    monkeypatch.setattr(auto_preset, "VerifierRuntime", _ExecutingVerifierRuntime)
    brain = _ReadWriteBrain()
    agent = _E2EAgent(brain, registry, workspace)
    with workspace_root_context(workspace.root):
        outcome = AutoPresetRunner(agent, workspace_root=workspace.root).run(GOAL)

    assert (workspace.root / "result.txt").read_text(encoding="utf-8") == EXPECTED_CONTENT
    assert brain.observed_source == SOURCE_CONTENT
    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.stop_reason_code is None
    assert outcome.verifier is not None
    assert outcome.verifier.status is VerificationStatus.PASSED
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.is_full_done is True
    assert outcome.canonical_outcome.ui_label == "DONE"
    assert outcome.canonical_outcome.communication_class == "final_success"
    assert outcome.canonical_outcome.verification.outcome is VerificationOutcome.PASSED
    state = agent.last_auto_state
    assert state["status"] == AutoRunStatus.COMPLETED.value
    assert state["verifier"]["status"] == VerificationStatus.PASSED.value
    assert state["execution_metrics"]["tool_calls_used"] == 2
    assert state["execution_metrics"]["files_touched"] == 1


def test_e2e_preset_run_with_failing_canonical_verifier_is_failed_verifier(
    tmp_path, monkeypatch
) -> None:
    """The canonical verifier really checks result.txt: wrong content
    fails make check, and the runner stops with FAILED_VERIFIER instead
    of completing. A canned-PASSED verifier double would pass this.
    """
    workspace, registry = _build_verified_e2e(tmp_path)
    monkeypatch.chdir(workspace.root)
    monkeypatch.setattr(auto_preset, "VerifierRuntime", _ExecutingVerifierRuntime)
    brain = _WrongContentBrain()
    agent = _E2EAgent(brain, registry, workspace)
    with workspace_root_context(workspace.root):
        outcome = AutoPresetRunner(agent, workspace_root=workspace.root).run(GOAL)

    assert (workspace.root / "result.txt").read_text(encoding="utf-8") == "не то\n"
    assert outcome.status == AutoRunStatus.FAILED_VERIFIER
    assert outcome.stop_reason_code == StopReasonCode.VERIFIER_FAILED
    assert outcome.verifier is not None
    assert outcome.verifier.status is VerificationStatus.FAILED
    assert outcome.canonical_outcome is None
