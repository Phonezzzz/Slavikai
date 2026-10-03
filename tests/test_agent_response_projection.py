from core.agent_response import AgentResponse
from core.auto_runtime import AutoRunOutcome
from core.mwv.models import StopReasonCode
from server.http.common.chat_payload import _project_agent_response
from shared.auto_models import AutoRunStatus


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
