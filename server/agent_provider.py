from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

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


@dataclass(slots=True)
class _AgentEntry[T]:
    agent: T
    borrowers: set[asyncio.Future[object]] = field(default_factory=set)
    idle: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.idle.set()


@dataclass(frozen=True, slots=True)
class AgentApplyFailure:
    scope: AgentScope | None
    error: Exception


class ScopedAgentProvider[T]:
    """Владеет одним mutable Agent и lock на principal/session scope."""

    def __init__(self, *, factory: Callable[[AgentScope, ModelConfig | None], T]) -> None:
        self._factory = factory
        self._agents: dict[AgentScope, _AgentEntry[T]] = {}
        self._locks: dict[AgentScope, asyncio.Lock] = {}
        self._creation_locks: dict[AgentScope, asyncio.Lock] = {}
        self._retirements: set[asyncio.Task[None]] = set()
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

    async def get_for_current_task(
        self,
        scope: AgentScope,
        main_config: ModelConfig | None,
    ) -> T:
        """Выдаёт Agent текущей task и удерживает его живым до завершения task."""
        if self._closed:
            raise RuntimeError("Agent provider is closed")
        if self._shared_instance is not None:
            return self._shared_instance
        creation_lock = self._creation_locks.setdefault(scope, asyncio.Lock())
        async with creation_lock:
            if self._closed:
                raise RuntimeError("Agent provider is closed")
            entry = self._agents.get(scope)
            if entry is None:
                entry = _AgentEntry(self._factory(scope, main_config))
                self._agents[scope] = entry
            self._borrow_for_current_task(entry)
            return entry.agent

    def lock_for(self, scope: AgentScope) -> asyncio.Lock:
        if self._shared_lock is not None:
            return self._shared_lock
        return self._locks.setdefault(scope, asyncio.Lock())

    async def release(self, scope: AgentScope) -> None:
        if self._shared_instance is not None:
            return
        creation_lock = self._creation_locks.setdefault(scope, asyncio.Lock())
        async with creation_lock:
            entry = self._agents.pop(scope, None)
        if entry is None:
            return
        await self._retire_removed_entry(scope, entry)

    async def apply_to_existing(
        self, callback: Callable[[T], None]
    ) -> tuple[AgentApplyFailure, ...]:
        if self._shared_instance is not None:
            if self._shared_lock is None:
                raise RuntimeError("Static agent lock is unavailable")
            async with self._shared_lock:
                try:
                    callback(self._shared_instance)
                except Exception as exc:  # noqa: BLE001
                    return (AgentApplyFailure(scope=None, error=exc),)
            return ()
        failures: list[AgentApplyFailure] = []
        retired: list[tuple[AgentScope, _AgentEntry[T]]] = []
        for scope, entry in list(self._agents.items()):
            async with self.lock_for(scope):
                if self._agents.get(scope) is not entry:
                    continue
                try:
                    callback(entry.agent)
                except Exception as exc:  # noqa: BLE001
                    failures.append(AgentApplyFailure(scope=scope, error=exc))
                    if self._agents.get(scope) is entry:
                        self._agents.pop(scope, None)
                        retired.append((scope, entry))
        for scope, entry in retired:
            try:
                await self._retire_removed_entry(scope, entry)
            except Exception as close_exc:  # noqa: BLE001
                failures.append(AgentApplyFailure(scope=scope, error=close_exc))
        return tuple(failures)

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
        if self._retirements:
            await asyncio.gather(*tuple(self._retirements), return_exceptions=True)

    def _borrow_for_current_task(self, entry: _AgentEntry[T]) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Scoped agent requires an active asyncio task")
        if task in entry.borrowers:
            return
        if not entry.borrowers:
            entry.idle.clear()
        entry.borrowers.add(task)
        task.add_done_callback(lambda completed: self._return_borrower(entry, completed))

    @staticmethod
    def _return_borrower(
        entry: _AgentEntry[T],
        task: asyncio.Future[object],
    ) -> None:
        entry.borrowers.discard(task)
        if not entry.borrowers:
            entry.idle.set()

    async def _retire_removed_entry(
        self,
        scope: AgentScope,
        entry: _AgentEntry[T],
    ) -> None:
        current_task = asyncio.current_task()
        if current_task is not None:
            self._return_borrower(entry, current_task)
        retirement = asyncio.create_task(
            self._retire_entry(entry, self.lock_for(scope)),
            name=f"retire-agent:{scope.principal_id}:{scope.session_id}",
        )
        self._retirements.add(retirement)
        retirement.add_done_callback(self._retirements.discard)
        await asyncio.shield(retirement)

    async def _retire_entry(self, entry: _AgentEntry[T], run_lock: asyncio.Lock) -> None:
        await entry.idle.wait()
        async with run_lock:
            self._close_agent(entry.agent)

    @staticmethod
    def _close_agent(agent: T) -> None:
        close = getattr(agent, "close", None)
        if callable(close):
            close()
