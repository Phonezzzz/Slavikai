from __future__ import annotations

import json
from dataclasses import dataclass, field

from core.tool_gateway import ToolGateway
from llm.brain_base import Brain
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
    ) -> AgentToolLoopResult:
        history = list(messages)
        executed: list[ExecutedToolCall] = []
        final_text = ""

        for iteration in range(1, self.max_iterations + 1):
            result = brain.generate(history, config=config, tools=tools)
            final_text = result.text

            assistant_tool_calls: list[dict[str, JSONValue]] | None = None
            if result.tool_calls:
                assistant_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in result.tool_calls
                ]

            history.append(
                LLMMessage(
                    role="assistant",
                    content=result.text,
                    tool_calls=assistant_tool_calls,
                    reasoning_content=result.reasoning,
                )
            )

            if not result.tool_calls:
                return AgentToolLoopResult(
                    text=final_text,
                    messages=history,
                    tool_calls=executed,
                    iterations=iteration,
                )

            for tool_call in result.tool_calls:
                tool_result = gateway.call(
                    ToolRequest(name=tool_call.name, args=dict(tool_call.arguments))
                )
                executed.append(ExecutedToolCall(call=tool_call, result=tool_result))
                history.append(
                    LLMMessage(
                        role="tool",
                        content=_serialize_tool_result(tool_result),
                        tool_call_id=tool_call.id,
                    )
                )

        return AgentToolLoopResult(
            text=final_text,
            messages=history,
            tool_calls=executed,
            iterations=self.max_iterations,
        )


def _serialize_tool_result(result: ToolResult) -> str:
    payload: dict[str, JSONValue] = {
        "ok": result.ok,
        "data": result.data,
        "error": result.error,
        "meta": result.meta,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
