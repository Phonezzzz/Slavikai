from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

from llm.types import LLMResult, LLMUsage, ToolCall
from shared.models import JSONValue, ToolResult

StreamTextMode = Literal["append", "replace"]


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Базовый тип всех событий потока LLM."""


@dataclass(frozen=True, slots=True)
class TextDelta(StreamEvent):
    text: str
    mode: StreamTextMode = "append"
    meta: dict[str, JSONValue] | None = None


@dataclass(frozen=True, slots=True)
class ToolCallStarted(StreamEvent):
    call_id: str
    name: str
    index: int


@dataclass(frozen=True, slots=True)
class ToolCallArgumentsDelta(StreamEvent):
    call_id: str
    delta: str
    index: int


@dataclass(frozen=True, slots=True)
class ToolCallCompleted(StreamEvent):
    call: ToolCall
    result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class Usage(StreamEvent):
    usage: LLMUsage


@dataclass(frozen=True, slots=True)
class Error(StreamEvent):
    message: str
    code: str = "stream_error"


@dataclass(frozen=True, slots=True)
class Done(StreamEvent):
    finish_reason: str | None = None


@dataclass(slots=True)
class _PendingToolCall:
    index: int
    call_id: str = ""
    name: str = ""
    raw_arguments: str = ""
    started: bool = False
    completed: bool = False

    def resolved_id(self) -> str:
        return self.call_id or f"tool-call-{self.index}"


class OpenAIStreamEventAdapter:
    """Преобразует OpenAI-совместимый поток в типизированные события."""

    def __init__(self, *, text_mode: StreamTextMode = "append") -> None:
        self.text_mode = text_mode
        self._pending_calls: dict[int, _PendingToolCall] = {}
        self._finish_reason: str | None = None
        self._terminal_error = False

    def feed(self, payload: dict[str, JSONValue]) -> list[StreamEvent]:
        if self._terminal_error:
            return []
        provider_error = _provider_error(payload)
        if provider_error is not None:
            self._terminal_error = True
            return [Error(message=provider_error, code="provider_stream_error")]

        events: list[StreamEvent] = []
        choices_raw = payload.get("choices")
        if isinstance(choices_raw, list):
            for choice in choices_raw:
                if not isinstance(choice, dict):
                    continue
                delta_raw = choice.get("delta")
                if isinstance(delta_raw, dict):
                    text = _content_text(delta_raw.get("content"))
                    if text:
                        events.append(TextDelta(text=text, mode=self.text_mode))
                    tool_calls_raw = delta_raw.get("tool_calls")
                    if isinstance(tool_calls_raw, list):
                        events.extend(self._feed_tool_calls(tool_calls_raw))
                finish_reason_raw = choice.get("finish_reason")
                if isinstance(finish_reason_raw, str) and finish_reason_raw:
                    self._finish_reason = finish_reason_raw

        usage = _usage_from_payload(payload)
        if usage is not None:
            events.append(Usage(usage=usage))
        return events

    def finish(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if not self._terminal_error:
            for index in sorted(self._pending_calls):
                pending = self._pending_calls[index]
                if pending.completed:
                    continue
                if not pending.name:
                    events.append(
                        Error(
                            message=(
                                f"У вызова инструмента {pending.resolved_id()} "
                                "отсутствует имя функции."
                            ),
                            code="invalid_tool_call",
                        )
                    )
                    self._terminal_error = True
                    break
                if not pending.started:
                    events.append(
                        ToolCallStarted(
                            call_id=pending.resolved_id(),
                            name=pending.name,
                            index=pending.index,
                        )
                    )
                    pending.started = True
                try:
                    arguments = _parse_tool_arguments(pending.raw_arguments)
                except ValueError as exc:
                    events.append(Error(message=str(exc), code="invalid_tool_arguments"))
                    self._terminal_error = True
                    break
                pending.completed = True
                events.append(
                    ToolCallCompleted(
                        call=ToolCall(
                            id=pending.resolved_id(),
                            name=pending.name,
                            arguments=arguments,
                            raw_arguments=pending.raw_arguments or None,
                        )
                    )
                )
        events.append(Done(finish_reason="error" if self._terminal_error else self._finish_reason))
        return events

    def _feed_tool_calls(self, tool_calls: list[JSONValue]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for fallback_index, item in enumerate(tool_calls):
            if not isinstance(item, dict):
                continue
            index_raw = item.get("index")
            index = index_raw if isinstance(index_raw, int) and index_raw >= 0 else fallback_index
            pending = self._pending_calls.setdefault(index, _PendingToolCall(index=index))
            call_id_raw = item.get("id")
            if isinstance(call_id_raw, str) and call_id_raw:
                pending.call_id = _merge_stream_fragment(pending.call_id, call_id_raw)
            function_raw = item.get("function")
            if not isinstance(function_raw, dict):
                continue
            name_raw = function_raw.get("name")
            if isinstance(name_raw, str) and name_raw:
                pending.name = _merge_stream_fragment(pending.name, name_raw)
            if pending.name and not pending.started:
                events.append(
                    ToolCallStarted(
                        call_id=pending.resolved_id(),
                        name=pending.name,
                        index=pending.index,
                    )
                )
                pending.started = True
            arguments_raw = function_raw.get("arguments")
            if isinstance(arguments_raw, str) and arguments_raw:
                pending.raw_arguments += arguments_raw
                events.append(
                    ToolCallArgumentsDelta(
                        call_id=pending.resolved_id(),
                        delta=arguments_raw,
                        index=pending.index,
                    )
                )
        return events


def iter_openai_sse_events(
    raw_lines: Iterable[str | bytes],
    *,
    text_mode: StreamTextMode = "append",
) -> Iterator[StreamEvent]:
    adapter = OpenAIStreamEventAdapter(text_mode=text_mode)
    for raw_line in raw_lines:
        line = (
            raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
        )
        normalized = line.strip()
        if not normalized or normalized.startswith(":") or not normalized.startswith("data:"):
            continue
        data_part = normalized.removeprefix("data:").strip()
        if not data_part:
            continue
        if data_part == "[DONE]":
            yield from adapter.finish()
            return
        try:
            parsed_json = json.loads(data_part)
        except json.JSONDecodeError as exc:
            yield Error(
                message=f"Некорректный JSON в потоке: {exc.msg}",
                code="invalid_stream_json",
            )
            yield Done(finish_reason="error")
            return
        if not isinstance(parsed_json, dict):
            yield Error(
                message="Событие потока должно быть JSON-объектом.",
                code="invalid_stream_payload",
            )
            yield Done(finish_reason="error")
            return
        payload: dict[str, JSONValue] = parsed_json
        events = adapter.feed(payload)
        yield from events
        if any(isinstance(event, Error) for event in events):
            yield from adapter.finish()
            return
    yield from adapter.finish()


def stream_events_from_result(
    result: LLMResult,
    *,
    chunk_size: int = 80,
) -> Iterator[StreamEvent]:
    normalized_chunk_size = max(1, chunk_size)
    for index in range(0, len(result.text), normalized_chunk_size):
        yield TextDelta(text=result.text[index : index + normalized_chunk_size])
    for index, call in enumerate(result.tool_calls):
        yield ToolCallStarted(call_id=call.id, name=call.name, index=index)
        raw_arguments = call.raw_arguments
        if raw_arguments is None:
            raw_arguments = json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":"))
        if raw_arguments:
            yield ToolCallArgumentsDelta(call_id=call.id, delta=raw_arguments, index=index)
        yield ToolCallCompleted(call=call)
    if result.usage is not None:
        yield Usage(usage=result.usage)
    yield Done()


def _content_text(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text_raw = item.get("text")
        if isinstance(text_raw, str):
            parts.append(text_raw)
    return "".join(parts)


def _provider_error(payload: dict[str, JSONValue]) -> str | None:
    error_raw = payload.get("error")
    if isinstance(error_raw, str) and error_raw.strip():
        return error_raw.strip()
    if not isinstance(error_raw, dict):
        return None
    message_raw = error_raw.get("message")
    if isinstance(message_raw, str) and message_raw.strip():
        return message_raw.strip()
    return json.dumps(error_raw, ensure_ascii=False, sort_keys=True)


def _usage_from_payload(payload: dict[str, JSONValue]) -> LLMUsage | None:
    usage_raw = payload.get("usage")
    if not isinstance(usage_raw, dict):
        return None
    return LLMUsage(
        prompt_tokens=_non_negative_int(usage_raw.get("prompt_tokens")),
        completion_tokens=_non_negative_int(usage_raw.get("completion_tokens")),
        total_tokens=_non_negative_int(usage_raw.get("total_tokens")),
    )


def _non_negative_int(value: JSONValue) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return 0


def _parse_tool_arguments(raw_arguments: str) -> dict[str, JSONValue]:
    if not raw_arguments.strip():
        return {}
    try:
        parsed_json = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Некорректные JSON-аргументы потокового вызова инструмента: {exc.msg}"
        ) from exc
    if not isinstance(parsed_json, dict):
        raise ValueError("Аргументы потокового вызова инструмента должны быть JSON-объектом.")
    return {str(key): value for key, value in parsed_json.items()}


def _merge_stream_fragment(current: str, fragment: str) -> str:
    if not current:
        return fragment
    if fragment == current:
        return current
    return f"{current}{fragment}"
