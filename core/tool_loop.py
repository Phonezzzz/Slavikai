from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field

from core.tool_gateway import ToolGateway
from llm.brain_base import Brain
from llm.cancellation import cancellation_requested
from llm.stream_model import (
    Done,
    Error,
    StreamEvent,
    TextDelta,
    ToolCallCompleted,
)
from llm.types import ModelConfig, ToolCall, ToolSpec
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
    ) -> AgentToolLoopResult:
        history = list(messages)
        executed: list[ExecutedToolCall] = []
        final_text = ""
        allowed_tool_names = {tool.name for tool in tools}

        for iteration in range(1, self.max_iterations + 1):
            if cancellation_requested(cancellation_token):
                return AgentToolLoopResult(
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration - 1,
                    cancelled=True,
                )
            result = brain.generate(history, config=config, tools=tools)
            if cancellation_requested(cancellation_token):
                return AgentToolLoopResult(
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                    cancelled=True,
                )
            final_text = result.text
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
                    if iteration < self.max_iterations:
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

            for tool_call in result.tool_calls:
                if cancellation_requested(cancellation_token):
                    return AgentToolLoopResult(
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )
                tool_result = _dispatch_model_tool_call(
                    gateway=gateway,
                    tool_call=tool_call,
                    allowed_tool_names=allowed_tool_names,
                )
                executed.append(ExecutedToolCall(call=tool_call, result=tool_result))
                history.append(
                    LLMMessage(
                        role="tool",
                        content=_serialize_tool_result(tool_result),
                        tool_call_id=tool_call.id,
                    )
                )
                if cancellation_requested(cancellation_token):
                    return AgentToolLoopResult(
                        text=final_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )

        message = f"Цикл инструментов превысил лимит: {self.max_iterations} итераций."
        return AgentToolLoopResult(
            text=final_text,
            messages=history,
            tool_calls=executed,
            iterations=self.max_iterations,
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

            for tool_call in pending_calls:
                if cancellation_requested(cancellation_token):
                    yield Done(finish_reason="cancelled")
                    return AgentToolLoopResult(
                        text=visible_text,
                        messages=history,
                        tool_calls=executed,
                        iterations=iteration,
                        cancelled=True,
                    )
                tool_result = _dispatch_model_tool_call(
                    gateway=gateway,
                    tool_call=tool_call,
                    allowed_tool_names=allowed_tool_names,
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
