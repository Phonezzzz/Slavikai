from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

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
    auto_detector: Callable[[str], dict[str, str] | None] | None = None,
) -> bool:
    """Определяет по превью, должен ли стрим идти в Canvas.

    Улучшенная версия с поддержкой AutoCanvasDetector.

    Args:
        preview_text: Текущий буфер текста
        canvas_char_threshold: Порог по символам
        canvas_code_line_threshold: Порог по строкам кода
        extract_named_files_from_output_fn: Функция извлечения файлов
        extract_named_file_markers_fn: Функция извлечения имён файлов
        auto_detector: Опциональная функция-детектор (chunk -> decision | None)
    """
    normalized = preview_text.strip()
    if not normalized:
        return False

    # Если есть auto_detector, используем его
    if auto_detector is not None:
        result = auto_detector(normalized)
        if result is not None:
            return result.get("action") == "promote_to_canvas"

    # Fallback на размер
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


async def _publish_canvas_switch(
    hub: SessionPublisher,
    *,
    session_id: str,
    artifact_id: str,
    language: str | None = None,
) -> None:
    """Публикует событие переключения в Canvas режим.

    Используется для mid-stream переключения когда детектор
    решает, что контент должен быть в Canvas.
    """
    await hub.publish(
        session_id,
        {
            "type": "canvas.switch",
            "payload": {
                "session_id": session_id,
                "artifact_id": artifact_id,
                "language": language,
            },
        },
    )
