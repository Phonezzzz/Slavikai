from __future__ import annotations

import asyncio
import json
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
from llm.types import LLMResult, ModelConfig, ToolCall, ToolSpec
from shared.models import JSONValue, LLMMessage, ToolRequest, ToolResult


@dataclass(frozen=True)
class ExecutedToolCall:
    call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class AgentToolLoopResult:
    text: str
    messages: list[LLMMessage]
    tool_calls: list[ExecutedToolCall] = field(default_factory=list)
    iterations: int = 0
    error: str | None = None
    cancelled: bool = False


class ToolLoopExecutionError(RuntimeError):
    def __init__(self, result: AgentToolLoopResult, cause: Exception) -> None:
        super().__init__("provider_generation_failed")
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


class AgentToolLoop:
    def __init__(self, max_iterations: int = 8) -> None:
        self.max_iterations = max(1, max_iterations)

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
    ) -> AgentToolLoopResult:
        history = list(messages)
        executed: list[ExecutedToolCall] = []
        final_text = ""
        allowed_tool_names = {tool.name for tool in tools}

        if continuation is not None:
            continuation = deepcopy(continuation)
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
        for iteration in range(first_iteration, maximum + 1):
            if cancellation_requested(cancellation_token):
                return AgentToolLoopResult(
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration - 1,
                    cancelled=True,
                )
            resuming = continuation is not None and iteration == first_iteration
            try:
                result = (
                    LLMResult(text=final_text, tool_calls=continuation.pending_calls)
                    if resuming and continuation is not None
                    else brain.generate(history, config=config, tools=tools)
                )
            except GenerationCancelled:
                return AgentToolLoopResult(
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                    cancelled=True,
                )
            except Exception as exc:
                raise ToolLoopExecutionError(
                    AgentToolLoopResult(
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
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                    cancelled=True,
                )
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
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error=gate_error,
                    )
                return AgentToolLoopResult(
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                )

            for call_index, tool_call in enumerate(result.tool_calls):
                if cancellation_requested(cancellation_token):
                    return AgentToolLoopResult(
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )
                try:
                    if resuming and call_index == 0 and continuation is not None:
                        tool_result = gateway.call_approved_once(
                            ToolRequest(name=tool_call.name, args=dict(tool_call.arguments)),
                            continuation.approval,
                        )
                    else:
                        tool_result = _dispatch_model_tool_call(
                            gateway=gateway,
                            tool_call=tool_call,
                            allowed_tool_names=allowed_tool_names,
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
                        )
                    )
                    raise
                executed.append(ExecutedToolCall(call=tool_call, result=tool_result))
                history.append(
                    LLMMessage(
                        role="tool",
                        content=_serialize_tool_result(tool_result),
                        tool_call_id=tool_call.id,
                    )
                )
                if _policy_rejected(tool_result):
                    return AgentToolLoopResult(
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        error="tool_policy_denied",
                    )
                if cancellation_requested(cancellation_token):
                    return AgentToolLoopResult(
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )

        message = f"Цикл инструментов превысил лимит: {maximum} итераций."
        return AgentToolLoopResult(
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
            iteration_text = ""
            pending_calls: list[ToolCall] = []
            stream_error: str | None = None
            stream_tools = tools if tools else None
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

            if cancellation_requested(cancellation_token):
                yield Done(finish_reason="cancelled")
                return AgentToolLoopResult(
                    text=visible_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                    cancelled=True,
                )

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
                        )
                    )
                    raise
                if cancellation_requested(cancellation_token):
                    yield Done(finish_reason="cancelled")
                    return AgentToolLoopResult(
                        text=visible_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )
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


def _dispatch_model_tool_call(
    *,
    gateway: ToolGateway,
    tool_call: ToolCall,
    allowed_tool_names: set[str],
) -> ToolResult:
    if tool_call.name not in allowed_tool_names:
        return ToolResult.failure(
            f"MODEL_TOOL_NOT_EXPOSED: tool '{tool_call.name}' was not provided to the model.",
            meta={"policy_reason": "model_tool_not_exposed"},
        )
    return gateway.call(ToolRequest(name=tool_call.name, args=dict(tool_call.arguments)))


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
