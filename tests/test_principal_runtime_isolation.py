from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from pytest import MonkeyPatch

from core.agent import Agent
from core.desktop_policy import DesktopPolicyStore
from core.desktop_runtime import DesktopRunCoordinator
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig, ToolSpec
from server.agent_provider import AgentScope, ScopedAgentProvider
from server.http.common.runtime_contract import (
    SessionApprovalStore,
    _agent_lock_for_request,
    _resolve_agent_for_ui_session,
)
from server.http.common.runtime_model_state import RuntimeModelResolver, RuntimeModelStateStore
from server.principal_storage import principal_storage_paths
from server.ui_hub import UIHub
from shared.models import LLMMessage, MemoryItem


class _NoopBrain(Brain):
    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        del messages, config, tools
        return LLMResult(text="ok")


class _DummyEmbeddingModel:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


def test_principal_storage_keeps_owner_legacy_paths_and_hashes_members(tmp_path: Path) -> None:
    memory_root = tmp_path / "memory"
    owner = principal_storage_paths(
        principal_id="email:owner@example.com",
        owner_principal_id="email:owner@example.com",
        memory_root=memory_root,
    )
    member = principal_storage_paths(
        principal_id="email:member@example.com",
        owner_principal_id="email:owner@example.com",
        memory_root=memory_root,
    )

    assert owner.memory_db == memory_root / "memory.db"
    assert owner.vectors_db == memory_root / "vectors.db"
    assert owner.memory_companion_db == memory_root / "memory_companion.db"
    assert owner.memory_categories_db == memory_root / "memory_categories.db"
    assert owner.canonical_atoms_db == memory_root / "canonical_atoms.db"
    assert member.memory_db.parent.parent == memory_root / "principals"
    assert "member@example.com" not in str(member.memory_db)
    assert member != owner


def test_scoped_provider_owns_distinct_agents_and_locks() -> None:
    async def _run() -> None:
        created: list[AgentScope] = []

        class _Agent:
            def __init__(self, scope: AgentScope) -> None:
                self.scope = scope
                self.closed = False

            def close(self) -> None:
                self.closed = True

        def _factory(scope: AgentScope, _config: ModelConfig | None) -> _Agent:
            created.append(scope)
            return _Agent(scope)

        provider = ScopedAgentProvider(factory=_factory)
        config = ModelConfig(provider="local", model="test-model")
        scope_a = AgentScope("principal-a", "session-a")
        scope_b = AgentScope("principal-a", "session-b")
        first_a, second_a = await asyncio.gather(
            provider.get_for_current_task(scope_a, config),
            provider.get_for_current_task(scope_a, config),
        )
        agent_b = await provider.get_for_current_task(scope_b, config)

        assert first_a is second_a
        assert first_a is not agent_b
        assert created == [scope_a, scope_b]
        scope_a_lock = provider.lock_for(scope_a)
        assert scope_a_lock is provider.lock_for(scope_a)
        assert scope_a_lock is not provider.lock_for(scope_b)
        visited: list[AgentScope] = []
        failures = await provider.apply_to_existing(lambda item: visited.append(item.scope))
        assert visited == [scope_a, scope_b]
        assert failures == ()

        await provider.release(scope_a)
        assert first_a.closed is True
        assert provider.lock_for(scope_a) is scope_a_lock
        replacement_a = await provider.get_for_current_task(scope_a, config)
        assert replacement_a is not first_a
        await provider.close()
        assert replacement_a.closed is True
        assert agent_b.closed is True

    asyncio.run(_run())


def test_scoped_provider_release_does_not_wait_for_request_borrower() -> None:
    async def _run() -> None:
        class _Agent:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        provider = ScopedAgentProvider(factory=lambda _scope, _config: _Agent())
        scope = AgentScope("principal-a", "session-a")
        borrowed = asyncio.Event()
        finish_request = asyncio.Event()
        original: _Agent | None = None

        async def _request() -> None:
            nonlocal original
            original = await provider.get_for_current_task(scope, None)
            borrowed.set()
            await finish_request.wait()
            assert original.closed is False

        request_task = asyncio.create_task(_request())
        await borrowed.wait()
        assert original is not None
        release_task = asyncio.create_task(provider.release(scope))
        await asyncio.sleep(0)

        await release_task
        assert original.closed is False
        replacement = await provider.get_for_current_task(scope, None)
        assert replacement is not original

        finish_request.set()
        await request_task
        await asyncio.sleep(0)
        assert original.closed is True
        assert replacement.closed is False
        await provider.close()
        assert replacement.closed is True

    asyncio.run(_run())


def test_ui_session_pruning_retires_its_scoped_agent() -> None:
    async def _run() -> None:
        class _Agent:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        provider = ScopedAgentProvider(factory=lambda _scope, _config: _Agent())
        hub = UIHub(
            max_sessions=1,
            on_session_pruned=lambda principal_id, session_id: provider.schedule_release(
                AgentScope(principal_id, session_id)
            ),
        )
        first_session = await hub.create_session("principal-a")
        first_scope = AgentScope("principal-a", first_session)
        first_agent = await asyncio.create_task(provider.get_for_current_task(first_scope, None))

        await hub.create_session("principal-a")
        await hub.list_sessions("principal-a")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert first_agent.closed is True
        replacement = await provider.get_for_current_task(first_scope, None)
        assert replacement is not first_agent
        await provider.close()

    asyncio.run(_run())


def test_scoped_provider_shutdown_forces_bounded_retirement() -> None:
    async def _run() -> None:
        class _Agent:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        provider = ScopedAgentProvider(
            factory=lambda _scope, _config: _Agent(),
            retirement_timeout_seconds=0.01,
        )
        scope = AgentScope("principal-a", "session-a")
        borrowed = asyncio.Event()
        never_finish = asyncio.Event()
        agent: _Agent | None = None

        async def _request() -> None:
            nonlocal agent
            agent = await provider.get_for_current_task(scope, None)
            borrowed.set()
            await never_finish.wait()

        request_task = asyncio.create_task(_request())
        await borrowed.wait()
        assert agent is not None

        await provider.release(scope)
        await asyncio.wait_for(provider.close(), timeout=0.2)

        assert agent.closed is True
        request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)

    asyncio.run(_run())


def test_scoped_provider_settings_failure_evicts_only_failed_agent() -> None:
    async def _run() -> None:
        class _Agent:
            def __init__(self, scope: AgentScope) -> None:
                self.scope = scope
                self.closed = False
                self.updated = False

            def close(self) -> None:
                self.closed = True

        provider = ScopedAgentProvider(
            factory=lambda scope, _config: _Agent(scope),
        )
        failed_scope = AgentScope("principal-a", "failed")
        healthy_scope = AgentScope("principal-a", "healthy")
        failed = await provider.get_for_current_task(failed_scope, None)
        healthy = await provider.get_for_current_task(healthy_scope, None)

        def _apply(agent: _Agent) -> None:
            if agent.scope == failed_scope:
                raise RuntimeError("reconfigure failed")
            agent.updated = True

        failures = await provider.apply_to_existing(_apply)

        assert len(failures) == 1
        assert failures[0].scope == failed_scope
        assert str(failures[0].error) == "reconfigure failed"
        assert failed.closed is True
        assert healthy.updated is True
        assert healthy.closed is False
        replacement = await provider.get_for_current_task(failed_scope, None)
        assert replacement is not failed
        await provider.close()

    asyncio.run(_run())


def test_session_approvals_are_isolated_by_principal_and_session() -> None:
    async def _run() -> None:
        store = SessionApprovalStore()
        owner_scope = AgentScope("email:owner@example.com", "shared-session")
        member_scope = AgentScope("email:member@example.com", "shared-session")

        await store.approve(owner_scope, {"EXEC_ARBITRARY"})

        assert await store.get_categories(owner_scope) == {"EXEC_ARBITRARY"}
        assert await store.get_categories(member_scope) == set()

    asyncio.run(_run())


def test_ui_session_resolution_uses_request_principal_and_session_scope() -> None:
    async def _run() -> None:
        created: list[AgentScope] = []

        class _ResolvedAgent:
            def reconfigure_models(
                self,
                main_config: ModelConfig,
                main_api_key: str | None = None,
                *,
                persist: bool = True,
            ) -> None:
                del main_config, main_api_key, persist

        def _factory(scope: AgentScope, _config: ModelConfig | None) -> _ResolvedAgent:
            created.append(scope)
            return _ResolvedAgent()

        provider = ScopedAgentProvider(factory=_factory)
        model = ModelConfig(provider="local", model="test-model")
        state = RuntimeModelStateStore(global_main=model)
        app = web.Application()
        app["agent_provider"] = provider
        app["runtime_model_resolver"] = RuntimeModelResolver(state)
        request = make_mocked_request("GET", "/ui/api/chat/send", app=app)
        request["principal_id"] = "email:member@example.com"

        session_a, selected_a = await _resolve_agent_for_ui_session(request, "session-a")
        session_b, selected_b = await _resolve_agent_for_ui_session(request, "session-b")

        assert session_a is not session_b
        assert selected_a == model
        assert selected_b == model
        assert created == [
            AgentScope("email:member@example.com", "session-a"),
            AgentScope("email:member@example.com", "session-b"),
        ]
        assert _agent_lock_for_request(request, "session-a") is not _agent_lock_for_request(
            request, "session-b"
        )

    asyncio.run(_run())


def test_agent_memory_vectors_and_runtime_state_are_principal_session_isolated(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model",
        lambda _self, _name: _DummyEmbeddingModel(),
    )
    owner_principal = "email:owner@example.com"
    member_principal = "email:member@example.com"
    memory_root = tmp_path / "memory"
    desktop_coordinator = DesktopRunCoordinator()
    desktop_policy_store = DesktopPolicyStore(tmp_path / "desktop-approvals.json")

    def _factory(scope: AgentScope, main_config: ModelConfig | None) -> Agent:
        assert main_config is not None
        storage = principal_storage_paths(
            principal_id=scope.principal_id,
            owner_principal_id=owner_principal,
            memory_root=memory_root,
        )
        return Agent(
            brain=_NoopBrain(),
            main_config=main_config,
            user_id=scope.principal_id,
            memory_db_path=str(storage.memory_db),
            vectors_db_path=str(storage.vectors_db),
            memory_companion_db_path=str(storage.memory_companion_db),
            memory_inbox_db_path=str(storage.memory_categories_db),
            canonical_atoms_db_path=str(storage.canonical_atoms_db),
            desktop_policy_store=desktop_policy_store,
            desktop_home=tmp_path,
            desktop_run_coordinator=desktop_coordinator,
        )

    async def _run() -> None:
        provider = ScopedAgentProvider(factory=_factory)
        config = ModelConfig(provider="local", model="test-model")
        owner_a = await provider.get_for_current_task(
            AgentScope(owner_principal, "owner-a"), config
        )
        owner_b = await provider.get_for_current_task(
            AgentScope(owner_principal, "owner-b"), config
        )
        member = await provider.get_for_current_task(
            AgentScope(member_principal, "member-a"), config
        )

        assert owner_a is not owner_b
        assert owner_a.memory is not owner_b.memory
        assert owner_a.memory.db_path == owner_b.memory.db_path
        assert owner_a.session_id is None
        assert owner_b.session_id is None
        owner_a.set_session_context("owner-a", set())
        assert owner_a.session_id == "owner-a"
        assert owner_b.session_id is None
        assert member.user_id == member_principal
        assert owner_a.desktop_runtime.run_coordinator is desktop_coordinator
        assert member.desktop_runtime.run_coordinator is desktop_coordinator

        owner_a.memory.save(
            MemoryItem(
                id="owner-canary",
                content="owner private memory",
                tags=["private"],
                timestamp="2026-08-23T00:00:00+00:00",
            )
        )
        assert [item.id for item in owner_b.memory.search("owner private")] == ["owner-canary"]
        assert member.memory.search("owner private") == []

        owner_a.vectors.index_text("owner-vector", "private vector", namespace="memory")
        assert owner_b.vectors.search("private", namespace="memory", top_k=5)
        assert member.vectors.search("private", namespace="memory", top_k=5) == []

        await provider.close()

    asyncio.run(_run())
