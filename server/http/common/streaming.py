from __future__ import annotations

import asyncio
import json
from typing import Literal, Protocol

from llm.stream_model import (
    Error,
    StreamEvent,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    Usage,
)
from shared.models import JSONValue

CHAT_STREAM_CHUNK_SIZE = 80
CHAT_STREAM_WARMUP_CHARS = 220


class SessionPublisher(Protocol):
    async def publish(self, session_id: str, event: dict[str, JSONValue]) -> None: ...


def _split_chat_stream_chunks(content: str) -> list[str]:
    if not content:
        return []
    return [
        content[idx : idx + CHAT_STREAM_CHUNK_SIZE]
        for idx in range(0, len(content), CHAT_STREAM_CHUNK_SIZE)
    ]


def _stream_preview_ready_for_chat(preview_text: str, *, chat_stream_warmup_chars: int) -> bool:
    """Проверяет, достаточно ли текста для начала показа в чате."""
    normalized = preview_text.strip()
    if not normalized:
        return False
    if len(normalized) >= chat_stream_warmup_chars:
        return True
    if len(normalized) >= 96 and "```" not in normalized and normalized.count("\n") <= 1:
        return True
    return False


async def _publish_chat_stream_start(
    hub: SessionPublisher,
    *,
    session_id: str,
    stream_id: str,
    lane: str = "chat",
) -> None:
    await hub.publish(
        session_id,
        {
            "type": "chat.stream.start",
            "payload": {
                "session_id": session_id,
                "stream_id": stream_id,
                "lane": lane,
            },
        },
    )


async def _publish_chat_stream_delta(
    hub: SessionPublisher,
    *,
    session_id: str,
    stream_id: str,
    delta: str,
    mode: Literal["append", "replace"] = "append",
    lane: str = "chat",
) -> None:
    if not delta:
        return
    normalized_mode: Literal["append", "replace"] = "replace" if mode == "replace" else "append"
    await hub.publish(
        session_id,
        {
            "type": "chat.stream.delta",
            "payload": {
                "session_id": session_id,
                "stream_id": stream_id,
                "delta": delta,
                "mode": normalized_mode,
                "lane": lane,
            },
        },
    )


async def _publish_chat_stream_done(
    hub: SessionPublisher,
    *,
    session_id: str,
    stream_id: str,
    lane: str = "chat",
) -> None:
    await hub.publish(
        session_id,
        {
            "type": "chat.stream.done",
            "payload": {
                "session_id": session_id,
                "stream_id": stream_id,
                "lane": lane,
            },
        },
    )


async def _publish_chat_protocol_event(
    hub: SessionPublisher,
    *,
    session_id: str,
    stream_id: str,
    event: StreamEvent,
    lane: str = "chat",
) -> None:
    event_type: str
    payload: dict[str, JSONValue] = {
        "session_id": session_id,
        "stream_id": stream_id,
        "lane": lane,
    }
    if isinstance(event, ToolCallStarted):
        event_type = "chat.tool.started"
        payload.update(
            {
                "call_id": event.call_id,
                "name": event.name,
                "index": event.index,
            }
        )
    elif isinstance(event, ToolCallArgumentsDelta):
        event_type = "chat.tool.arguments.delta"
        payload.update(
            {
                "call_id": event.call_id,
                "delta": event.delta,
                "index": event.index,
            }
        )
    elif isinstance(event, ToolCallCompleted):
        event_type = "chat.tool.completed"
        payload.update(
            {
                "call_id": event.call.id,
                "name": event.call.name,
                "arguments": event.call.arguments,
                "result": _tool_result_payload(event),
            }
        )
    elif isinstance(event, Usage):
        event_type = "chat.stream.usage"
        payload.update(
            {
                "prompt_tokens": event.usage.prompt_tokens,
                "completion_tokens": event.usage.completion_tokens,
                "total_tokens": event.usage.total_tokens,
            }
        )
    elif isinstance(event, Error):
        event_type = "chat.stream.error"
        payload.update({"message": event.message, "code": event.code})
    else:
        raise TypeError(f"Unsupported auxiliary stream event: {type(event).__name__}")
    await hub.publish(session_id, {"type": event_type, "payload": payload})


def _tool_result_payload(event: ToolCallCompleted) -> dict[str, JSONValue] | None:
    result = event.result
    if result is None:
        return None
    return {
        "ok": result.ok,
        "error": result.error,
        "summary": _short_tool_result(result.data, result.error),
    }


def _short_tool_result(data: dict[str, JSONValue], error: str | None) -> str:
    if error:
        return error[:240]
    for key in ("output", "content", "result"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:240]
    serialized = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return serialized[:240]


async def _publish_chat_stream_from_text(
    hub: SessionPublisher,
    *,
    session_id: str,
    stream_id: str,
    content: str,
    lane: str = "chat",
) -> None:
    await _publish_chat_stream_start(hub, session_id=session_id, stream_id=stream_id, lane=lane)
    for chunk in _split_chat_stream_chunks(content):
        await _publish_chat_stream_delta(
            hub,
            session_id=session_id,
            stream_id=stream_id,
            delta=chunk,
            lane=lane,
        )
        await asyncio.sleep(0.01)
    await _publish_chat_stream_done(hub, session_id=session_id, stream_id=stream_id, lane=lane)


def _split_canvas_stream_chunks(content: str) -> list[str]:
    """Разбивает контент для Canvas-стрима на чанки.

    Использует построчное разбиение для лучшего UX.
    """
    if not content:
        return []
    lines = content.splitlines(keepends=True)
    if len(lines) <= 2:
        chunk_size = 120
        return [content[idx : idx + chunk_size] for idx in range(0, len(content), chunk_size)]
    chunks: list[str] = []
    lines_per_chunk = 4
    for start in range(0, len(lines), lines_per_chunk):
        chunks.append("".join(lines[start : start + lines_per_chunk]))
    return [chunk for chunk in chunks if chunk]


async def _publish_canvas_stream(
    hub: SessionPublisher,
    *,
    session_id: str,
    artifact_id: str,
    content: str,
) -> None:
    """Публикует контент в Canvas stream."""
    await hub.publish(
        session_id,
        {
            "type": "canvas.stream.start",
            "payload": {
                "session_id": session_id,
                "artifact_id": artifact_id,
            },
        },
    )
    chunks = _split_canvas_stream_chunks(content)
    if not chunks:
        await hub.publish(
            session_id,
            {
                "type": "canvas.stream.done",
                "payload": {
                    "session_id": session_id,
                    "artifact_id": artifact_id,
                },
            },
        )
        return
    for delta in chunks:
        await hub.publish(
            session_id,
            {
                "type": "canvas.stream.delta",
                "payload": {
                    "session_id": session_id,
                    "artifact_id": artifact_id,
                    "delta": delta,
                },
            },
        )
        await asyncio.sleep(0.02)
    await hub.publish(
        session_id,
        {
            "type": "canvas.stream.done",
            "payload": {
                "session_id": session_id,
                "artifact_id": artifact_id,
            },
        },
    )
