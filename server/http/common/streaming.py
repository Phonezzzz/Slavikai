from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Protocol

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


def _stream_preview_indicates_canvas(
    preview_text: str,
    *,
    canvas_char_threshold: int,
    canvas_code_line_threshold: int,
    extract_named_files_from_output_fn: Callable[[str], Sequence[object]],
    extract_named_file_markers_fn: Callable[[str], list[str]],
) -> bool:
    normalized = preview_text.strip()
    if not normalized:
        return False
    if len(normalized) >= canvas_char_threshold:
        return True
    if len(normalized.splitlines()) >= canvas_code_line_threshold:
        return True
    if extract_named_files_from_output_fn(normalized):
        return True
    tail = normalized[-320:]
    if "```" in tail and extract_named_file_markers_fn(tail):
        return True
    return False


def _stream_preview_ready_for_chat(preview_text: str, *, chat_stream_warmup_chars: int) -> bool:
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
) -> None:
    await hub.publish(
        session_id,
        {
            "type": "chat.stream.start",
            "payload": {
                "session_id": session_id,
                "stream_id": stream_id,
            },
        },
    )


async def _publish_chat_stream_delta(
    hub: SessionPublisher,
    *,
    session_id: str,
    stream_id: str,
    delta: str,
) -> None:
    if not delta:
        return
    await hub.publish(
        session_id,
        {
            "type": "chat.stream.delta",
            "payload": {
                "session_id": session_id,
                "stream_id": stream_id,
                "delta": delta,
            },
        },
    )


async def _publish_chat_stream_done(
    hub: SessionPublisher,
    *,
    session_id: str,
    stream_id: str,
) -> None:
    await hub.publish(
        session_id,
        {
            "type": "chat.stream.done",
            "payload": {
                "session_id": session_id,
                "stream_id": stream_id,
            },
        },
    )


async def _publish_chat_stream_from_text(
    hub: SessionPublisher,
    *,
    session_id: str,
    stream_id: str,
    content: str,
) -> None:
    await _publish_chat_stream_start(hub, session_id=session_id, stream_id=stream_id)
    for chunk in _split_chat_stream_chunks(content):
        await _publish_chat_stream_delta(
            hub,
            session_id=session_id,
            stream_id=stream_id,
            delta=chunk,
        )
        await asyncio.sleep(0.01)
    await _publish_chat_stream_done(hub, session_id=session_id, stream_id=stream_id)


def _split_canvas_stream_chunks(content: str) -> list[str]:
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
