from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from server.agent_provider import AgentScope


@dataclass(frozen=True)
class PlanExecution:
    task_id: str
    token: asyncio.Event
    finished: asyncio.Event


class PlanCancellationRegistry:
    """Сигнал отмены matching task до ожидания scoped Agent lock."""

    def __init__(self) -> None:
        self._active: dict[AgentScope, PlanExecution] = {}
        self._closed = False

    @asynccontextmanager
    async def running(
        self, scope: AgentScope, task_id: str, token: asyncio.Event | None
    ) -> AsyncIterator[asyncio.Event]:
        if self._closed:
            raise RuntimeError("plan_cancellation_registry_closed")
        if scope in self._active:
            raise RuntimeError("plan_execution_already_active")
        execution = PlanExecution(task_id, token or asyncio.Event(), asyncio.Event())
        self._active[scope] = execution
        try:
            yield execution.token
        finally:
            if self._active.get(scope) is execution:
                self._active.pop(scope)
            execution.finished.set()

    def request_cancel(self, scope: AgentScope, task_id: str) -> bool:
        execution = self._active.get(scope)
        if execution is None or execution.task_id != task_id:
            return False
        execution.token.set()
        return True

    async def shutdown(self) -> tuple[str, ...]:
        self._closed = True
        executions = tuple(self._active.values())
        for execution in executions:
            execution.token.set()
        if executions:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(item.finished.wait() for item in executions)), timeout=10
                )
            except TimeoutError:
                return ("Timed out waiting for active plan executions to stop.",)
        return ()
