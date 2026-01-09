from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from core.mwv.models import (
    MWVMessage,
    RetryDecision,
    RetryPolicy,
    RunContext,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkResult,
    WorkStatus,
)
from core.mwv.routing import MessageLike, RouteDecision, classify_request
from shared.models import JSONValue

RouteClassifier = Callable[[Sequence[MessageLike], str, dict[str, JSONValue] | None], RouteDecision]
TaskBuilder = Callable[[Sequence[MWVMessage], RunContext], TaskPacket]
WorkRunner = Callable[[TaskPacket, RunContext], WorkResult]
VerifierRunner = Callable[[RunContext], VerificationResult]


@dataclass(frozen=True)
class MWVRunResult:
    task: TaskPacket
    work_result: WorkResult
    verification_result: VerificationResult
    attempt: int
    max_attempts: int
    retry_decision: RetryDecision | None


@dataclass(frozen=True)
class ManagerRuntime:
    task_builder: TaskBuilder
    route_classifier: RouteClassifier = classify_request

    def decide_route(
        self,
        messages: Sequence[MessageLike],
        user_input: str,
        context: dict[str, JSONValue] | None = None,
    ) -> RouteDecision:
        return self.route_classifier(messages, user_input, context)

    def build_task_packet(self, messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
        return self.task_builder(messages, context)

    def run_flow(
        self,
        messages: Sequence[MWVMessage],
        context: RunContext,
        worker: WorkRunner,
        verifier: VerifierRunner,
    ) -> MWVRunResult:
        task = self.build_task_packet(messages, context)
        max_attempts = max(1, context.max_retries + 1)
        attempt = 1
        retry_decision: RetryDecision | None = None
        while True:
            attempt_context = replace(context, attempt=attempt)
            work_result = worker(task, attempt_context)
            verification_result = verifier(attempt_context)
            if _is_success(work_result, verification_result):
                return MWVRunResult(
                    task=task,
                    work_result=work_result,
                    verification_result=verification_result,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    retry_decision=retry_decision,
                )
            retry_decision = decide_retry(
                work_result=work_result,
                verification_result=verification_result,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            if not retry_decision.allow_retry:
                return MWVRunResult(
                    task=task,
                    work_result=work_result,
                    verification_result=verification_result,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    retry_decision=retry_decision,
                )
            task = build_retry_task(task, verification_result, retry_decision)
            attempt += 1


def decide_retry(
    *,
    work_result: WorkResult,
    verification_result: VerificationResult,
    attempt: int,
    max_attempts: int,
) -> RetryDecision:
    if attempt >= max_attempts:
        return RetryDecision(
            policy=RetryPolicy.LIMITED,
            allow_retry=False,
            reason="retry_limit_reached",
            attempt=attempt,
            max_retries=max_attempts - 1,
        )
    if verification_result.status == VerificationStatus.ERROR:
        return RetryDecision(
            policy=RetryPolicy.LIMITED,
            allow_retry=False,
            reason="verifier_error",
            attempt=attempt,
            max_retries=max_attempts - 1,
        )
    if work_result.status == WorkStatus.FAILURE:
        return RetryDecision(
            policy=RetryPolicy.LIMITED,
            allow_retry=False,
            reason="worker_failed",
            attempt=attempt,
            max_retries=max_attempts - 1,
        )
    return RetryDecision(
        policy=RetryPolicy.LIMITED,
        allow_retry=True,
        reason="verifier_failed",
        attempt=attempt,
        max_retries=max_attempts - 1,
    )


def build_retry_task(
    task: TaskPacket,
    verification_result: VerificationResult,
    decision: RetryDecision,
) -> TaskPacket:
    summary = summarize_verifier_failure(verification_result)
    constraint = f"Исправь только минимальные изменения. Причина: {summary}"
    constraints = list(task.constraints)
    if decision.attempt > 0:
        constraints.append(constraint)
    return TaskPacket(
        task_id=task.task_id,
        session_id=task.session_id,
        trace_id=task.trace_id,
        goal=task.goal,
        messages=task.messages,
        constraints=constraints,
        context=task.context,
    )


def summarize_verifier_failure(result: VerificationResult) -> str:
    if result.status == VerificationStatus.ERROR:
        return result.error or "Ошибка верификации"
    text = (result.stderr or result.stdout or "").strip()
    if not text:
        if result.exit_code is None:
            return "Неизвестная ошибка проверки"
        return f"Код возврата: {result.exit_code}"
    return text.splitlines()[0][:200]


def _is_success(work_result: WorkResult, verification_result: VerificationResult) -> bool:
    return (
        work_result.status == WorkStatus.SUCCESS
        and verification_result.status == VerificationStatus.PASSED
    )
