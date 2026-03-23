from __future__ import annotations

import asyncio

from llm.types import ModelConfig
from server.http.common.runtime_model_state import (
    RuntimeModelResolver,
    RuntimeModelStateStore,
    build_runtime_model_state_from_persisted,
)


def _model(model_id: str) -> ModelConfig:
    return ModelConfig(provider="xai", model=model_id)


def test_runtime_model_state_store_starts_empty() -> None:
    store = RuntimeModelStateStore()

    assert asyncio.run(store.get_global_main()) is None


def test_runtime_model_state_store_roundtrips_global_main() -> None:
    store = RuntimeModelStateStore()
    expected = _model("global-model")

    asyncio.run(store.set_global_main(expected))

    assert asyncio.run(store.get_global_main()) == expected


def test_runtime_model_state_store_roundtrips_session_override() -> None:
    store = RuntimeModelStateStore()
    expected = _model("session-model")

    asyncio.run(store.set_session_override("session-1", expected))

    assert asyncio.run(store.get_session_override("session-1")) == expected


def test_runtime_model_state_store_clears_session_override() -> None:
    store = RuntimeModelStateStore()
    expected = _model("session-model")
    asyncio.run(store.set_session_override("session-1", expected))

    asyncio.run(store.clear_session_override("session-1"))

    assert asyncio.run(store.get_session_override("session-1")) is None


def test_runtime_model_state_store_clear_session_override_is_idempotent() -> None:
    store = RuntimeModelStateStore()

    asyncio.run(store.clear_session_override("session-1"))
    asyncio.run(store.clear_session_override("session-1"))

    assert asyncio.run(store.get_session_override("session-1")) is None


def test_runtime_model_resolver_prefers_session_override() -> None:
    store = RuntimeModelStateStore(global_main=_model("global-model"))
    asyncio.run(store.set_session_override("session-1", _model("session-model")))
    resolver = RuntimeModelResolver(store)

    assert asyncio.run(resolver.resolve_main("session-1")) == _model("session-model")


def test_runtime_model_resolver_falls_back_to_global_main() -> None:
    store = RuntimeModelStateStore(global_main=_model("global-model"))
    resolver = RuntimeModelResolver(store)

    assert asyncio.run(resolver.resolve_main("session-1")) == _model("global-model")


def test_runtime_model_resolver_returns_none_when_empty() -> None:
    resolver = RuntimeModelResolver(RuntimeModelStateStore())

    assert asyncio.run(resolver.resolve_main("session-1")) is None


def test_runtime_model_resolver_blank_session_id_behaves_like_no_session() -> None:
    store = RuntimeModelStateStore(global_main=_model("global-model"))
    resolver = RuntimeModelResolver(store)

    assert asyncio.run(resolver.resolve_main("   ")) == _model("global-model")


def test_build_runtime_model_state_from_persisted_hydrates_global_main() -> None:
    expected = _model("hydrated-model")

    store = build_runtime_model_state_from_persisted(load_model_configs_fn=lambda: expected)

    assert asyncio.run(store.get_global_main()) == expected
