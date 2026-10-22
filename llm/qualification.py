from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace

from llm.cancellation import GenerationCancelled, cancellation_requested
from llm.local_http_brain import ChatResponseError, LocalHttpBrain
from llm.stream_model import Done, Error, TextDelta, ToolCallCompleted
from llm.types import LLMResult, ModelConfig, ToolSpec
from shared.models import LLMMessage


class ProtocolQualificationError(ValueError):
    """Ответ модели не подтвердил проверяемый inert tool protocol."""


def _sync_result(
    brain: LocalHttpBrain,
    messages: list[LLMMessage],
    config: ModelConfig,
    tools: list[ToolSpec],
    token: asyncio.Event,
) -> LLMResult:
    try:
        return brain.generate_cancellable(
            messages, config=config, tools=tools, cancellation_token=token, max_attempts=1
        )
    except (ChatResponseError, ValueError) as exc:
        raise ProtocolQualificationError("Некорректный ответ chat completion protocol.") from exc


def _stream_result(
    brain: LocalHttpBrain,
    messages: list[LLMMessage],
    config: ModelConfig,
    tools: list[ToolSpec],
    token: asyncio.Event,
) -> LLMResult:
    result = LLMResult(text="")
    terminal = False
    with closing(
        brain.generate_stream_events(messages, config=config, tools=tools, cancellation_token=token)
    ) as events:
        for index, event in enumerate(events):
            if cancellation_requested(token):
                raise GenerationCancelled("Qualification cancelled")
            if index >= 256:
                raise ProtocolQualificationError("Слишком длинный ответ проверки.")
            if isinstance(event, Error):
                raise ProtocolQualificationError("Модель вернула ошибку tool protocol.")
            if isinstance(event, TextDelta):
                result.text += event.text
            elif isinstance(event, ToolCallCompleted):
                result.tool_calls.append(event.call)
            elif isinstance(event, Done):
                if event.finish_reason not in {"stop", "tool_calls"}:
                    raise ProtocolQualificationError("Проверка не получила завершённый ответ.")
                terminal = True
            if len(result.text) > 4096 or len(result.tool_calls) > 1:
                raise ProtocolQualificationError("Ответ превысил пределы проверки.")
    if not terminal:
        raise ProtocolQualificationError("Поток завершился без terminal result.")
    return result


def run_inert_qualification(
    brain: LocalHttpBrain,
    config: ModelConfig,
    token: asyncio.Event,
    is_current: Callable[[], bool],
) -> None:
    """Четыре synthetic requests; функция модели никогда не исполняется на host."""
    nonce = secrets.token_hex(8)
    cfg = replace(
        config, temperature=0.0, max_tokens=256, system_prompt=None, thinking_enabled=False
    )
    tool = ToolSpec(
        name="inert_fixture",
        description="Protocol fixture only; no host action.",
        parameters_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"nonce": {"type": "string", "enum": [nonce]}},
            "required": ["nonce"],
        },
    )
    for streaming in (False, True):
        if cancellation_requested(token) or not is_current():
            raise GenerationCancelled("Qualification cancelled")
        messages = [
            LLMMessage(
                role="user",
                content=(
                    f"Protocol check: call inert_fixture once with nonce {nonce}. "
                    "After the tool result, reply with exactly that nonce and no more tool calls."
                ),
            )
        ]
        result = (
            _stream_result(brain, messages, cfg, [tool], token)
            if streaming
            else _sync_result(brain, messages, cfg, [tool], token)
        )
        if len(result.tool_calls) != 1:
            raise ProtocolQualificationError("Модель не вернула единственный inert tool call.")
        call = result.tool_calls[0]
        if (
            not call.id
            or len(call.id) > 256
            or call.name != tool.name
            or call.arguments != {"nonce": nonce}
        ):
            raise ProtocolQualificationError("Модель вернула неверный tool call или arguments.")
        messages.extend(
            [
                LLMMessage(
                    role="assistant",
                    content=result.text,
                    tool_calls=[
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.raw_arguments or '{"nonce":"' + nonce + '"}',
                            },
                        }
                    ],
                ),
                LLMMessage(role="tool", content=nonce, tool_call_id=call.id),
            ]
        )
        if cancellation_requested(token) or not is_current():
            raise GenerationCancelled("Qualification cancelled")
        final = (
            _stream_result(brain, messages, cfg, [tool], token)
            if streaming
            else _sync_result(brain, messages, cfg, [tool], token)
        )
        if final.tool_calls or final.text.strip() != nonce:
            raise ProtocolQualificationError("Модель не подтвердила role=tool continuation.")
    if cancellation_requested(token) or not is_current():
        raise GenerationCancelled("Qualification cancelled")
