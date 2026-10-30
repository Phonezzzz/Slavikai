from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable, Generator, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

from core.approval_policy import ApprovalRequest, ApprovalRequired
from core.tool_gateway import ToolGateway
from llm.brain_base import Brain
from llm.cancellation import GenerationCancelled, cancellation_requested
from llm.stream_model import (
    Done,
    Error,
    StreamEvent,
    TextDelta,
    ToolCallCompleted,
)
from llm.types import LLMResult, LLMUsage, ModelConfig, ToolCall, ToolSpec
from shared.models import JSONValue, LLMMessage, ToolRequest, ToolResult
from shared.tool_capture import CaptureBinding, ToolCaptureContext, ToolCaptureFault


@dataclass(frozen=True)
class ExecutedToolCall:
    call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class ModelUsageSummary:
    """Reported usage внешних sync generate calls; не полный расход провайдера."""

    started_calls: int = 0
    reported_calls: int = 0
    unknown_calls: int = 0
    reported_sum: LLMUsage = field(default_factory=lambda: LLMUsage(0, 0, 0))

    @property
    def total_tokens(self) -> int | None:
        if self.started_calls == 0 or self.unknown_calls:
            return None
        return self.reported_sum.total_tokens

    def start_call(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            self.started_calls + 1,
            self.reported_calls,
            self.unknown_calls + 1,
            self.reported_sum,
        )

    def with_reported(self, usage: LLMUsage) -> ModelUsageSummary:
        return ModelUsageSummary(
            self.started_calls,
            self.reported_calls + 1,
            self.unknown_calls - 1,
            LLMUsage(
                self.reported_sum.prompt_tokens + usage.prompt_tokens,
                self.reported_sum.completion_tokens + usage.completion_tokens,
                self.reported_sum.total_tokens + usage.total_tokens,
            ),
        )


@dataclass(frozen=True)
class AgentToolLoopResult:
    text: str
    messages: list[LLMMessage]
    tool_calls: list[ExecutedToolCall] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    cancelled: bool = False
    model_usage: ModelUsageSummary | None = None


class ToolLoopExecutionError(RuntimeError):
    def __init__(self, result: AgentToolLoopResult, cause: Exception) -> None:
        super().__init__(result.error or "tool_loop_execution_failed")
        self.result = result
        self.cause = cause


@dataclass(frozen=True)
class ToolLoopContinuation:
    history: list[LLMMessage]
    executed: list[ExecutedToolCall]
    pending_calls: list[ToolCall]
    approval: ApprovalRequest
    text: str
    iteration: int
    max_iterations: int
    tools: list[ToolSpec]
    config: ModelConfig | None
    max_tool_calls: int | None = None
    expires_at_monotonic: float | None = None
    model_usage: ModelUsageSummary | None = None
    max_model_tokens: int | None = None
    capture_binding: CaptureBinding | None = None


class AgentToolLoop:
    def __init__(
        self,
        max_iterations: int = 8,
        *,
        max_tool_calls: int | None = None,
        expires_at_monotonic: float | None = None,
        max_model_tokens: int | None = None,
    ) -> None:
        self.max_iterations = max(1, max_iterations)
        if max_tool_calls is not None and (type(max_tool_calls) is not int or max_tool_calls < 1):
            raise ValueError("max_tool_calls must be a positive integer or None")
        self.max_tool_calls = max_tool_calls
        if max_model_tokens is not None and (
            type(max_model_tokens) is not int or max_model_tokens < 1
        ):
            raise ValueError("max_model_tokens must be a positive integer or None")
        self.max_model_tokens = max_model_tokens
        if expires_at_monotonic is not None and (
            type(expires_at_monotonic) not in (int, float)
            or not math.isfinite(expires_at_monotonic)
        ):
            raise ValueError("expires_at_monotonic must be finite or None")
        self.expires_at_monotonic = expires_at_monotonic

    def run(
        self,
        *,
        brain: Brain,
        gateway: ToolGateway,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        config: ModelConfig | None = None,
        cancellation_token: asyncio.Event | None = None,
        final_gate: Callable[[Sequence[ExecutedToolCall]], str | None] | None = None,
        continuation: ToolLoopContinuation | None = None,
        record_model_usage: bool = False,
        capture_binding: CaptureBinding | None = None,
    ) -> AgentToolLoopResult:
        if capture_binding is not None and not isinstance(capture_binding, CaptureBinding):
            raise ToolCaptureFault("workspace_capture_binding_invalid")
        if (
            continuation is not None
            and capture_binding is not None
            and continuation.capture_binding != capture_binding
        ):
            raise ToolCaptureFault("workspace_capture_continuation_binding_changed")
        model_usage = (
            ModelUsageSummary() if record_model_usage or self.max_model_tokens is not None else None
        )
        history = list(messages)
        executed: list[ExecutedToolCall] = []
        final_text = ""
        allowed_tool_names = {tool.name for tool in tools}

        if continuation is not None:
            continuation = deepcopy(continuation)
            capture_binding = continuation.capture_binding
            model_usage = continuation.model_usage
            history = continuation.history
            executed = continuation.executed
            final_text = continuation.text
            tools = continuation.tools
            config = continuation.config
            allowed_tool_names = {tool.name for tool in tools}
        first_iteration = continuation.iteration if continuation is not None else 1
        maximum = (
            min(self.max_iterations, continuation.max_iterations)
            if continuation is not None
            else self.max_iterations
        )
        token_cap = (
            continuation.max_model_tokens if continuation is not None else self.max_model_tokens
        )
        call_cap = continuation.max_tool_calls if continuation is not None else self.max_tool_calls
        expires_at = (
            continuation.expires_at_monotonic
            if continuation is not None
            else self.expires_at_monotonic
        )
        for iteration in range(first_iteration, maximum + 1):
            if cancellation_requested(cancellation_token):
                return AgentToolLoopResult(
                    model_usage=model_usage,
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration - 1,
                    cancelled=True,
                )
            expired = _expiry_stop(
                expires_at, final_text, history, executed, iteration - 1, model_usage
            )
            if expired is not None:
                return expired
            resuming = continuation is not None and iteration == first_iteration
            try:
                if resuming and continuation is not None:
                    result = LLMResult(text=final_text, tool_calls=continuation.pending_calls)
                else:
                    token_error = _token_admission_error(token_cap, model_usage)
                    if token_error is not None:
                        return AgentToolLoopResult(
                            text=final_text,
                            messages=history,
                            tool_calls=executed,
                            iterations=iteration - 1,
                            error=token_error,
                            model_usage=model_usage,
                        )
                    if model_usage is not None:
                        model_usage = model_usage.start_call()
                    result = brain.generate(history, config=config, tools=tools)
                    if model_usage is not None and result.usage is not None:
                        model_usage = model_usage.with_reported(result.usage)
            except GenerationCancelled:
                return AgentToolLoopResult(
                    model_usage=model_usage,
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                    cancelled=True,
                )
            except Exception as exc:
                raise ToolLoopExecutionError(
                    AgentToolLoopResult(
                        model_usage=model_usage,
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error="provider_generation_failed",
                    ),
                    exc,
                ) from exc
            if cancellation_requested(cancellation_token):
                return AgentToolLoopResult(
                    model_usage=model_usage,
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                    cancelled=True,
                )
            expired = _expiry_stop(
                expires_at, final_text, history, executed, iteration, model_usage
            )
            if expired is not None:
                return expired
            final_text = result.text
            if not resuming:
                history.append(
                    _assistant_message(
                        text=result.text,
                        tool_calls=result.tool_calls,
                        reasoning=result.reasoning,
                    )
                )

            if not result.tool_calls:
                gate_error = final_gate(executed) if final_gate is not None else None
                if gate_error is not None:
                    history.append(
                        LLMMessage(
                            role="system",
                            content=(
                                "Deterministic result verification rejected the current final "
                                f"answer: {gate_error}. Correct the execution and verify again."
                            ),
                        )
                    )
                    if iteration < maximum:
                        continue
                    return AgentToolLoopResult(
                        model_usage=model_usage,
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error=gate_error,
                    )
                return AgentToolLoopResult(
                    model_usage=model_usage,
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                )

            for call_index, tool_call in enumerate(result.tool_calls):
                if cancellation_requested(cancellation_token):
                    return AgentToolLoopResult(
                        model_usage=model_usage,
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )
                expired = _expiry_stop(
                    expires_at, final_text, history, executed, iteration, model_usage
                )
                if expired is not None:
                    return expired
                if call_cap is not None and len(executed) >= call_cap:
                    return AgentToolLoopResult(
                        model_usage=model_usage,
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error="tool_call_budget_exhausted",
                    )
                try:
                    if resuming and call_index == 0 and continuation is not None:
                        tool_result = gateway.call_approved_once(
                            _captured_tool_request(tool_call, capture_binding),
                            continuation.approval,
                        )
                    else:
                        tool_result = _dispatch_model_tool_call(
                            gateway=gateway,
                            tool_call=tool_call,
                            allowed_tool_names=allowed_tool_names,
                            capture_binding=capture_binding,
                        )
                except ApprovalRequired as exc:
                    exc.continuation = deepcopy(
                        ToolLoopContinuation(
                            history,
                            executed,
                            result.tool_calls[call_index:],
                            exc.request,
                            final_text,
                            iteration,
                            maximum,
                            tools,
                            config,
                            call_cap,
                            expires_at,
                            model_usage,
                            token_cap,
                            capture_binding,
                        )
                    )
                    raise
                except Exception as exc:
                    cause = exc
                    if (
                        capture_binding is not None
                        and tool_call.name == "workspace_run"
                        and not isinstance(exc, ToolCaptureFault)
                    ):
                        cause = ToolCaptureFault("workspace_capture_dispatch_or_publication_failed")
                        cause.__cause__ = exc
                    if (
                        call_cap is None
                        and expires_at is None
                        and not isinstance(cause, ToolCaptureFault)
                    ):
                        raise
                    raise ToolLoopExecutionError(
                        AgentToolLoopResult(
                            model_usage=model_usage,
                            text=final_text,
                            messages=history,
                            tool_calls=executed,
                            iterations=iteration,
                            error="tool_capture_failed"
                            if isinstance(cause, ToolCaptureFault)
                            else "tool_dispatch_failed",
                        ),
                        cause,
                    ) from cause
                executed.append(ExecutedToolCall(call=tool_call, result=tool_result))
                history.append(
                    LLMMessage(
                        role="tool",
                        content=_serialize_tool_result(tool_result),
                        tool_call_id=tool_call.id,
                    )
                )
                if (
                    tool_result.meta is not None
                    and tool_result.meta.get("execution_review_required") is True
                ):
                    return AgentToolLoopResult(
                        model_usage=model_usage,
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error="workspace_execution_review_required",
                    )
                if _policy_rejected(tool_result):
                    return AgentToolLoopResult(
                        model_usage=model_usage,
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error="tool_policy_denied",
                    )
                if cancellation_requested(cancellation_token):
                    return AgentToolLoopResult(
                        model_usage=model_usage,
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )

        message = f"Цикл инструментов превысил лимит: {maximum} итераций."
        return AgentToolLoopResult(
            model_usage=model_usage,
            text=final_text,
            messages=history,
            tool_calls=executed,
            iterations=maximum,
            error=message,
        )

    def run_stream_events(
        self,
        *,
        brain: Brain,
        gateway: ToolGateway,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        config: ModelConfig | None = None,
        cancellation_token: asyncio.Event | None = None,
        final_gate: Callable[[Sequence[ExecutedToolCall]], str | None] | None = None,
    ) -> Generator[StreamEvent, None, AgentToolLoopResult]:
        history = list(messages)
        executed: list[ExecutedToolCall] = []
        visible_text = ""
        allowed_tool_names = {tool.name for tool in tools}

        for iteration in range(1, self.max_iterations + 1):
            if cancellation_requested(cancellation_token):
                yield Done(finish_reason="cancelled")
                return AgentToolLoopResult(
                    text=visible_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration - 1,
                    cancelled=True,
                )
            expired = _expiry_stop(
                self.expires_at_monotonic, visible_text, history, executed, iteration - 1
            )
            if expired is not None:
                yield Error(
                    message="Auto run expired before the next operation.",
                    code="runtime_budget_exhausted",
                )
                yield Done(finish_reason="error")
                return expired
            iteration_text = ""
            pending_calls: list[ToolCall] = []
            stream_error: str | None = None
            stream_tools = tools if tools else None
            try:
                if cancellation_token is None:
                    provider_events = brain.generate_stream_events(
                        history,
                        config=config,
                        tools=stream_tools,
                    )
                else:
                    provider_events = brain.generate_stream_events(
                        history,
                        config=config,
                        tools=stream_tools,
                        cancellation_token=cancellation_token,
                    )
                for event in provider_events:
                    if cancellation_requested(cancellation_token):
                        yield Done(finish_reason="cancelled")
                        return AgentToolLoopResult(
                            text=visible_text,
                            messages=history,
                            tool_calls=executed,
                            iterations=iteration,
                            cancelled=True,
                        )
                    if isinstance(event, TextDelta):
                        if event.mode == "replace":
                            iteration_text = event.text
                            visible_text = event.text
                        else:
                            iteration_text = f"{iteration_text}{event.text}"
                            visible_text = f"{visible_text}{event.text}"
                        yield event
                        continue
                    if isinstance(event, ToolCallCompleted):
                        pending_calls.append(event.call)
                        continue
                    if isinstance(event, Error):
                        stream_error = event.message
                        yield event
                        continue
                    if isinstance(event, Done):
                        if event.finish_reason == "cancelled":
                            yield event
                            return AgentToolLoopResult(
                                text=visible_text,
                                messages=history,
                                tool_calls=executed,
                                iterations=iteration,
                                cancelled=True,
                            )
                        continue
                    yield event

            except GenerationCancelled:
                yield Done(finish_reason="cancelled")
                return AgentToolLoopResult(
                    text=visible_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                    cancelled=True,
                )
            except Exception as exc:
                raise ToolLoopExecutionError(
                    AgentToolLoopResult(
                        text=visible_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error="provider_generation_failed",
                    ),
                    exc,
                ) from exc

            if cancellation_requested(cancellation_token):
                yield Done(finish_reason="cancelled")
                return AgentToolLoopResult(
                    text=visible_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                    cancelled=True,
                )

            expired = _expiry_stop(
                self.expires_at_monotonic, visible_text, history, executed, iteration
            )
            if expired is not None:
                yield Error(
                    message="Auto run expired before the next operation.",
                    code="runtime_budget_exhausted",
                )
                yield Done(finish_reason="error")
                return expired
            if stream_error is not None:
                yield Done(finish_reason="error")
                return AgentToolLoopResult(
                    text=visible_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                    error=stream_error,
                )

            history.append(
                _assistant_message(
                    text=iteration_text,
                    tool_calls=pending_calls,
                    reasoning=None,
                )
            )
            if not pending_calls:
                gate_error = final_gate(executed) if final_gate is not None else None
                if gate_error is not None:
                    history.append(
                        LLMMessage(
                            role="system",
                            content=(
                                "Deterministic result verification rejected the current final "
                                f"answer: {gate_error}. Correct the execution and verify again."
                            ),
                        )
                    )
                    if iteration < self.max_iterations:
                        continue
                    yield Error(message=gate_error, code="tool_loop_verification_failed")
                    yield Done(finish_reason="error")
                    return AgentToolLoopResult(
                        text=visible_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error=gate_error,
                    )
                yield Done()
                return AgentToolLoopResult(
                    text=visible_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                )

            for call_index, tool_call in enumerate(pending_calls):
                if cancellation_requested(cancellation_token):
                    yield Done(finish_reason="cancelled")
                    return AgentToolLoopResult(
                        text=visible_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )
                expired = _expiry_stop(
                    self.expires_at_monotonic, visible_text, history, executed, iteration
                )
                if expired is not None:
                    yield Error(
                        message="Auto run expired before the next operation.",
                        code="runtime_budget_exhausted",
                    )
                    yield Done(finish_reason="error")
                    return expired
                if self.max_tool_calls is not None and len(executed) >= self.max_tool_calls:
                    yield Error(
                        message="Native tool call budget exhausted.",
                        code="tool_call_budget_exhausted",
                    )
                    yield Done(finish_reason="error")
                    return AgentToolLoopResult(
                        text=visible_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error="tool_call_budget_exhausted",
                    )
                try:
                    tool_result = _dispatch_model_tool_call(
                        gateway=gateway, tool_call=tool_call, allowed_tool_names=allowed_tool_names
                    )
                except ApprovalRequired as exc:
                    exc.continuation = deepcopy(
                        ToolLoopContinuation(
                            history,
                            executed,
                            pending_calls[call_index:],
                            exc.request,
                            visible_text,
                            iteration,
                            self.max_iterations,
                            tools,
                            config,
                            self.max_tool_calls,
                            self.expires_at_monotonic,
                        )
                    )
                    raise
                except Exception as exc:
                    if self.max_tool_calls is None and self.expires_at_monotonic is None:
                        raise
                    raise ToolLoopExecutionError(
                        AgentToolLoopResult(
                            text=visible_text,
                            messages=history,
                            tool_calls=executed,
                            iterations=iteration,
                            error="tool_dispatch_failed",
                        ),
                        exc,
                    ) from exc
                executed_call = ExecutedToolCall(call=tool_call, result=tool_result)
                executed.append(executed_call)
                history.append(
                    LLMMessage(
                        role="tool",
                        content=_serialize_tool_result(tool_result),
                        tool_call_id=tool_call.id,
                    )
                )
                yield ToolCallCompleted(call=tool_call, result=tool_result)
                if (
                    tool_result.meta is not None
                    and tool_result.meta.get("execution_review_required") is True
                ):
                    yield Error(
                        message=(
                            "Результат исполнения требует проверки; "
                            "автоматическое продолжение остановлено."
                        ),
                        code="workspace_execution_review_required",
                    )
                    yield Done(finish_reason="error")
                    return AgentToolLoopResult(
                        text=visible_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error="workspace_execution_review_required",
                    )
                if _policy_rejected(tool_result):
                    yield Error(message="tool_policy_denied", code="tool_policy_denied")
                    yield Done(finish_reason="error")
                    return AgentToolLoopResult(
                        text=visible_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error="tool_policy_denied",
                    )
                if cancellation_requested(cancellation_token):
                    yield Done(finish_reason="cancelled")
                    return AgentToolLoopResult(
                        text=visible_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )

        message = f"Цикл инструментов превысил лимит: {self.max_iterations} итераций."
        yield Error(message=message, code="tool_loop_iteration_limit")
        yield Done(finish_reason="error")
        return AgentToolLoopResult(
            text=visible_text,
            messages=history,
            tool_calls=executed,
            iterations=self.max_iterations,
            error=message,
        )


def _expiry_stop(
    expires_at: float | None,
    text: str,
    history: list[LLMMessage],
    executed: list[ExecutedToolCall],
    iterations: int,
    model_usage: ModelUsageSummary | None = None,
) -> AgentToolLoopResult | None:
    if expires_at is None or time.monotonic() < expires_at:
        return None
    return AgentToolLoopResult(
        text=text,
        messages=history,
        tool_calls=executed,
        iterations=iterations,
        error="runtime_budget_exhausted",
        model_usage=model_usage,
    )


def _dispatch_model_tool_call(
    *,
    gateway: ToolGateway,
    tool_call: ToolCall,
    allowed_tool_names: set[str],
    capture_binding: CaptureBinding | None = None,
) -> ToolResult:
    if tool_call.name not in allowed_tool_names:
        return ToolResult.failure(
            f"MODEL_TOOL_NOT_EXPOSED: tool '{tool_call.name}' was not provided to the model.",
            meta={"policy_reason": "model_tool_not_exposed"},
        )
    return gateway.call(_captured_tool_request(tool_call, capture_binding))


def _captured_tool_request(call: ToolCall, binding: CaptureBinding | None) -> ToolRequest:
    context = (
        ToolCaptureContext(binding, call.id)
        if binding is not None and call.name == "workspace_run"
        else None
    )
    return ToolRequest(call.name, dict(call.arguments), capture=context)


def _policy_rejected(result: ToolResult) -> bool:
    return (
        not result.ok
        and result.meta is not None
        and isinstance(result.meta.get("policy_reason"), str)
    )


def _assistant_message(
    *,
    text: str,
    tool_calls: list[ToolCall],
    reasoning: str | None,
) -> LLMMessage:
    assistant_tool_calls: list[dict[str, JSONValue]] | None = None
    if tool_calls:
        assistant_tool_calls = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                },
            }
            for tool_call in tool_calls
        ]
    return LLMMessage(
        role="assistant",
        content=text,
        tool_calls=assistant_tool_calls,
        reasoning_content=reasoning,
    )


def _serialize_tool_result(result: ToolResult) -> str:
    payload: dict[str, JSONValue] = {
        "trust": "untrusted_observation",
        "ok": result.ok,
        "data": result.data,
        "error": result.error,
        "meta": result.meta,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _token_admission_error(cap: int | None, usage: ModelUsageSummary | None) -> str | None:
    if cap is None:
        return None
    if usage is None or usage.unknown_calls:
        return "model_usage_unknown"
    if usage.started_calls > 0 and usage.reported_sum.total_tokens >= cap:
        return "model_token_budget_exhausted"
    return None
