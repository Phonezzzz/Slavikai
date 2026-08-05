from __future__ import annotations

import asyncio
from dataclasses import dataclass

from llm.cancellation import cancel_generation


@dataclass(frozen=True, slots=True)
class ActiveChatGeneration:
    session_id: str
    stream_id: str
    token: asyncio.Event
    finished: asyncio.Event


@dataclass(frozen=True, slots=True)
class ChatCancellationResult:
    generation: ActiveChatGeneration
    close_errors: tuple[str, ...]


class ChatGenerationAlreadyActive(RuntimeError):
    pass


class ChatCancellationRegistry:
    """Хранит ровно одну активную генерацию на UI-сессию."""

    def __init__(self) -> None:
        self._active: dict[str, ActiveChatGeneration] = {}
        self._lock = asyncio.Lock()

    async def start(self, *, session_id: str, stream_id: str) -> ActiveChatGeneration:
        generation = ActiveChatGeneration(
            session_id=session_id,
            stream_id=stream_id,
            token=asyncio.Event(),
            finished=asyncio.Event(),
        )
        async with self._lock:
            existing = self._active.get(session_id)
            if existing is not None and not existing.finished.is_set():
                raise ChatGenerationAlreadyActive(session_id)
            self._active[session_id] = generation
        return generation

    async def request_cancel(self, session_id: str) -> ChatCancellationResult | None:
        async with self._lock:
            generation = self._active.get(session_id)
            if generation is None or generation.finished.is_set():
                return None
            generation.token.set()
        close_errors = await asyncio.to_thread(cancel_generation, generation.token)
        return ChatCancellationResult(
            generation=generation,
            close_errors=tuple(close_errors),
        )

    async def finish(self, generation: ActiveChatGeneration) -> None:
        async with self._lock:
            current = self._active.get(generation.session_id)
            if current is generation:
                self._active.pop(generation.session_id, None)
            generation.finished.set()

    async def shutdown(self) -> tuple[str, ...]:
        async with self._lock:
            generations = list(self._active.values())
            for generation in generations:
                generation.token.set()
        error_groups = await asyncio.gather(
            *(asyncio.to_thread(cancel_generation, item.token) for item in generations),
        )
        errors = tuple(error for group in error_groups for error in group)
        if generations:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(item.finished.wait() for item in generations)),
                    timeout=10,
                )
            except TimeoutError:
                errors = (*errors, "Timed out waiting for active chat generations to stop.")
        return errors
