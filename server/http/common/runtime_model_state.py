from __future__ import annotations

import asyncio
from collections.abc import Callable

from llm.types import ModelConfig


class RuntimeModelStateStore:
    """Server-owned runtime model state.

    Persisted config remains the durable source for cold-start hydration only.
    Runtime active model state lives here and will become the future source of
    truth for request-time model resolution.
    """

    def __init__(self, global_main: ModelConfig | None = None) -> None:
        self._global_main = global_main
        self._session_overrides: dict[str, ModelConfig] = {}
        self._lock = asyncio.Lock()

    def peek_global_main(self) -> ModelConfig | None:
        return self._global_main

    async def get_global_main(self) -> ModelConfig | None:
        async with self._lock:
            return self._global_main

    async def set_global_main(self, main: ModelConfig | None) -> None:
        async with self._lock:
            self._global_main = main

    async def get_session_override(self, session_id: str) -> ModelConfig | None:
        normalized = _normalize_session_id(session_id)
        if normalized is None:
            return None
        async with self._lock:
            return self._session_overrides.get(normalized)

    async def set_session_override(self, session_id: str, main: ModelConfig | None) -> None:
        normalized = _normalize_session_id(session_id)
        if normalized is None:
            return
        async with self._lock:
            if main is None:
                self._session_overrides.pop(normalized, None)
            else:
                self._session_overrides[normalized] = main

    async def clear_session_override(self, session_id: str) -> None:
        normalized = _normalize_session_id(session_id)
        if normalized is None:
            return
        async with self._lock:
            self._session_overrides.pop(normalized, None)


class RuntimeModelResolver:
    """Resolve runtime main model with fixed precedence.

    Resolution order is:
    1. Session override
    2. Global runtime main model
    3. None
    """

    def __init__(self, store: RuntimeModelStateStore) -> None:
        self._store = store

    async def resolve_main(self, session_id: str | None) -> ModelConfig | None:
        normalized = _normalize_session_id(session_id)
        if normalized is not None:
            session_override = await self._store.get_session_override(normalized)
            if session_override is not None:
                return session_override
        return await self._store.get_global_main()


def build_runtime_model_state_from_persisted(
    *,
    load_model_configs_fn: Callable[[], ModelConfig | None],
) -> RuntimeModelStateStore:
    """Hydrate global runtime state once at app startup."""

    return RuntimeModelStateStore(global_main=load_model_configs_fn())


def _normalize_session_id(session_id: str | None) -> str | None:
    if not isinstance(session_id, str):
        return None
    normalized = session_id.strip()
    if not normalized:
        return None
    return normalized
