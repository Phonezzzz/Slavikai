from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from llm.types import ModelConfig


@dataclass(frozen=True, slots=True)
class AgentScope:
    principal_id: str
    session_id: str

    def __post_init__(self) -> None:
        principal_id = self.principal_id.strip()
        session_id = self.session_id.strip()
        if not principal_id:
            raise ValueError("principal_id must be non-empty")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        object.__setattr__(self, "principal_id", principal_id)
        object.__setattr__(self, "session_id", session_id)


class ScopedAgentProvider[T]:
    """Owns one mutable Agent instance and lock per principal/session scope."""

    def __init__(self, *, factory: Callable[[AgentScope, ModelConfig | None], T]) -> None:
        self._factory = factory
        self._agents: dict[AgentScope, T] = {}
        self._locks: dict[AgentScope, asyncio.Lock] = {}
        self._creation_locks: dict[AgentScope, asyncio.Lock] = {}
        self._shared_instance: T | None = None
        self._shared_lock: asyncio.Lock | None = None
        self._closed = False

    @classmethod
    def from_instance(cls, agent: T) -> ScopedAgentProvider[T]:
        provider = cls(factory=lambda _scope, _config: agent)
        provider._shared_instance = agent
        provider._shared_lock = asyncio.Lock()
        return provider

    @property
    def is_static(self) -> bool:
        return self._shared_instance is not None

    async def get(self, scope: AgentScope, main_config: ModelConfig | None) -> T:
        if self._closed:
            raise RuntimeError("Agent provider is closed")
        if self._shared_instance is not None:
            return self._shared_instance
        existing = self._agents.get(scope)
        if existing is not None:
            return existing
        creation_lock = self._creation_locks.setdefault(scope, asyncio.Lock())
        async with creation_lock:
            existing = self._agents.get(scope)
            if existing is None:
                existing = self._factory(scope, main_config)
                self._agents[scope] = existing
            return existing

    def lock_for(self, scope: AgentScope) -> asyncio.Lock:
        if self._shared_lock is not None:
            return self._shared_lock
        return self._locks.setdefault(scope, asyncio.Lock())

    async def release(self, scope: AgentScope) -> None:
        if self._shared_instance is not None:
            return
        creation_lock = self._creation_locks.setdefault(scope, asyncio.Lock())
        async with creation_lock:
            run_lock = self.lock_for(scope)
            async with run_lock:
                agent = self._agents.pop(scope, None)
                if agent is not None:
                    self._close_agent(agent)

    async def apply_to_existing(self, callback: Callable[[T], None]) -> None:
        if self._shared_instance is not None:
            if self._shared_lock is None:
                raise RuntimeError("Static agent lock is unavailable")
            async with self._shared_lock:
                callback(self._shared_instance)
            return
        for scope, agent in list(self._agents.items()):
            async with self.lock_for(scope):
                if self._agents.get(scope) is agent:
                    callback(agent)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._shared_instance is not None:
            if self._shared_lock is None:
                raise RuntimeError("Static agent lock is unavailable")
            async with self._shared_lock:
                self._close_agent(self._shared_instance)
            self._shared_instance = None
            return
        for scope in list(self._agents):
            await self.release(scope)

    @staticmethod
    def _close_agent(agent: T) -> None:
        close = getattr(agent, "close", None)
        if callable(close):
            close()
