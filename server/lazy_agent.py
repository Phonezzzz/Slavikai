from __future__ import annotations

import asyncio
from collections.abc import Callable


class LazyAgentProvider[T]:
    def __init__(self, *, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._agent: T | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_instance(cls, agent: T) -> LazyAgentProvider[T]:
        provider = cls(factory=lambda: agent)
        provider._agent = agent
        return provider

    async def get(self) -> T:
        if self._agent is not None:
            return self._agent
        async with self._lock:
            if self._agent is None:
                self._agent = self._factory()
            return self._agent

    async def ensure(self, factory: Callable[[], T]) -> T:
        if self._agent is not None:
            return self._agent
        async with self._lock:
            if self._agent is None:
                self._agent = factory()
            return self._agent
