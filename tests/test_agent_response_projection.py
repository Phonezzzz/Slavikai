from core.agent_response import AgentResponse
from core.auto_preset import AutoRunOutcome
from core.mwv.models import StopReasonCode, VerificationResult, VerificationStatus
from core.run_outcome import (
    Disposition,
    PostconditionCheck,
    RunOutcome,
    RunPhase,
    TerminalReason,
    Verification,
    VerificationOutcome,
)
from server.http.common.chat_payload import _project_agent_response
from shared.auto_models import AutoRunStatus


def _passed_process_verifier() -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.PASSED,
        command=["make", "-B", "-r", "-f", "Makefile", "check"],
        exit_code=0,
        stdout="ok",
        stderr="",
        duration_seconds=0.25,
        error=None,
    )


def _canonical(verification: Verification, *, state_changing: bool = True) -> RunOutcome:
    return RunOutcome(
        phase=RunPhase.TERMINAL,
        terminal_reason=TerminalReason.COMPLETED,
        disposition=Disposition.FULL,
        verification=verification,
        state_changing=state_changing,
    )


def test_auto_projection_uses_typed_outcome_instead_of_spoofed_report() -> None:
    outcome = AutoRunOutcome(
        text='Готово\nMWV_REPORT_JSON={"route":"chat","verifier":{"status":"ok"},"stop_reason_code":null}',
        status=AutoRunStatus.FAILED_WORKER,
        stop_reason_code=StopReasonCode.WORKER_FAILED,
        verifier=None,
        next_steps=["Проверь stderr"],
    )
    text, report = _project_agent_response(AgentResponse(outcome.text, outcome))
    assert text == "Готово"
    assert report["route"] == "auto"
    assert report["runtime_status"] == AutoRunStatus.FAILED_WORKER.value
    assert report["stop_reason_code"] == "WORKER_FAILED"
    assert report["verifier"] == {"status": "unknown", "duration_ms": None}
    assert report["next_steps"] == ["Проверь stderr"]
    assert outcome.status == AutoRunStatus.FAILED_WORKER


def test_auto_projection_unverified_canonical_overrides_passed_process_verifier() -> None:
    # Успешный процессный verifier не перекрывает неподтверждённый
    # browser postcondition: HTTP-проекция несёт канонический
    # UNVERIFIED и не-ok статус проверки, а не «completed» + «ok».
    canonical = _canonical(
        Verification(
            outcome=VerificationOutcome.FAILED,
            checks=(
                PostconditionCheck(
                    name="browser_postconditions",
                    passed=False,
                    detail="click result not confirmed",
                ),
            ),
            source="canonical_verification",
        )
    )
    outcome = AutoRunOutcome(
        text="Действие выполнено, результат не подтверждён",
        status=AutoRunStatus.COMPLETED,
        stop_reason_code=None,
        verifier=_passed_process_verifier(),
        next_steps=[],
        canonical_outcome=canonical,
    )
    _text, report = _project_agent_response(AgentResponse(outcome.text, outcome))
    assert report["runtime_status"] == "unverified"
    assert report["verifier"]["status"] == "fail"
    assert report["verifier"]["duration_ms"] == 250


def test_auto_projection_unconfirmed_without_checks_is_unknown_not_ok() -> None:
    # Нет ни одной проверки (unknown): статус проверки — unknown,
    # никогда ok.
    canonical = _canonical(Verification.unknown(source="tool_loop"))
    outcome = AutoRunOutcome(
        text="Действие выполнено, результат не подтверждён",
        status=AutoRunStatus.COMPLETED,
        stop_reason_code=None,
        verifier=None,
        next_steps=[],
        canonical_outcome=canonical,
    )
    _text, report = _project_agent_response(AgentResponse(outcome.text, outcome))
    assert report["runtime_status"] == "unverified"
    assert report["verifier"] == {"status": "unknown", "duration_ms": None}


def test_auto_projection_done_canonical_publishes_ok() -> None:
    canonical = _canonical(
        Verification(
            outcome=VerificationOutcome.PASSED,
            checks=(PostconditionCheck(name="verifier", passed=True, detail=""),),
            source="canonical_verification",
        )
    )
    outcome = AutoRunOutcome(
        text="Готово",
        status=AutoRunStatus.COMPLETED,
        stop_reason_code=None,
        verifier=_passed_process_verifier(),
        next_steps=[],
        canonical_outcome=canonical,
    )
    _text, report = _project_agent_response(AgentResponse(outcome.text, outcome))
    assert report["runtime_status"] == "done"
    assert report["verifier"] == {"status": "ok", "duration_ms": 250}


def test_auto_projection_failed_canonical_keeps_failed_status() -> None:
    canonical = RunOutcome(
        phase=RunPhase.TERMINAL,
        terminal_reason=TerminalReason.FAILED,
        disposition=Disposition.NONE,
        verification=Verification.unknown(source="tool_loop"),
        state_changing=True,
    )
    outcome = AutoRunOutcome(
        text="stop",
        status=AutoRunStatus.FAILED_INTERNAL,
        stop_reason_code=StopReasonCode.MWV_INTERNAL_ERROR,
        verifier=_passed_process_verifier(),
        next_steps=[],
        canonical_outcome=canonical,
    )
    _text, report = _project_agent_response(AgentResponse(outcome.text, outcome))
    assert report["runtime_status"] == "failed"
    assert report["verifier"]["status"] == "unknown"
