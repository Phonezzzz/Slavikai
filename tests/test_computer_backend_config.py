"""Tests for ComputerBackendConfig / resolve_computer_backend_config() and
Agent.make_computer_runtime() env-driven backend selection.

No Docker daemon needed — ContainerComputerBackend construction does not
invoke docker; only methods like run_command/check do.  FakeContainerRunner
is NOT used here: tests verify that the production path selects the correct
backend type without calling docker at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config.computer_backend_config import resolve_computer_backend_config
from core.agent_computer import AgentComputerRuntime
from core.computer_backend import LocalComputerBackend
from core.container_computer_backend import ContainerComputerBackend

# ── resolve_computer_backend_config ─────────────────────────────────────────


def test_no_env_returns_local_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLAVIK_COMPUTER_BACKEND", raising=False)
    cfg = resolve_computer_backend_config()
    assert cfg.mode == "local"


def test_explicit_local_env_returns_local_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "local")
    cfg = resolve_computer_backend_config()
    assert cfg.mode == "local"


def test_explicit_container_with_image_returns_container_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "python:3.12-slim")
    cfg = resolve_computer_backend_config()
    assert cfg.mode == "container"
    assert cfg.container_image == "python:3.12-slim"


def test_default_engine_is_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLAVIK_COMPUTER_CONTAINER_ENGINE", raising=False)
    cfg = resolve_computer_backend_config()
    assert cfg.container_engine == "docker"


def test_podman_engine_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "img")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_ENGINE", "podman")
    cfg = resolve_computer_backend_config()
    assert cfg.container_engine == "podman"


def test_invalid_backend_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "k8s")
    with pytest.raises(ValueError, match="SLAVIK_COMPUTER_BACKEND"):
        resolve_computer_backend_config()


def test_invalid_engine_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "img")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_ENGINE", "nerdctl")
    with pytest.raises(ValueError, match="SLAVIK_COMPUTER_CONTAINER_ENGINE"):
        resolve_computer_backend_config()


def test_container_without_image_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.delenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", raising=False)
    with pytest.raises(ValueError, match="SLAVIK_COMPUTER_CONTAINER_IMAGE"):
        resolve_computer_backend_config()


def test_custom_workspace_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "img")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_WORKSPACE", "/code")
    cfg = resolve_computer_backend_config()
    assert cfg.container_workspace == "/code"


def test_default_workspace_is_slash_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLAVIK_COMPUTER_CONTAINER_WORKSPACE", raising=False)
    cfg = resolve_computer_backend_config()
    assert cfg.container_workspace == "/workspace"


def test_config_is_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLAVIK_COMPUTER_BACKEND", raising=False)
    cfg = resolve_computer_backend_config()
    with pytest.raises((AttributeError, TypeError)):
        cfg.mode = "container"  # type: ignore[misc]


# ── make_computer_runtime: default local ─────────────────────────────────────


def test_make_computer_runtime_default_creates_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var → LocalComputerBackend (unchanged behaviour)."""
    from core.agent import Agent

    monkeypatch.delenv("SLAVIK_COMPUTER_BACKEND", raising=False)
    gateway_mock = MagicMock()

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return gateway_mock

        make_computer_runtime = Agent.make_computer_runtime

    runtime = _StubAgent().make_computer_runtime()
    assert isinstance(runtime, AgentComputerRuntime)
    assert isinstance(runtime.backend, LocalComputerBackend)
    assert runtime.backend.gateway is gateway_mock


def test_make_computer_runtime_explicit_local_creates_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agent import Agent

    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "local")
    gateway_mock = MagicMock()

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return gateway_mock

        make_computer_runtime = Agent.make_computer_runtime

    runtime = _StubAgent().make_computer_runtime()
    assert isinstance(runtime.backend, LocalComputerBackend)


# ── make_computer_runtime: container opt-in ──────────────────────────────────


def test_make_computer_runtime_container_creates_container_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SLAVIK_COMPUTER_BACKEND=container → ContainerComputerBackend (no docker call)."""
    from core.agent import Agent

    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "python:3.12-slim")

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return MagicMock()

        make_computer_runtime = Agent.make_computer_runtime

    runtime = _StubAgent().make_computer_runtime()
    assert isinstance(runtime, AgentComputerRuntime)
    assert isinstance(runtime.backend, ContainerComputerBackend)


def test_make_computer_runtime_container_uses_configured_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agent import Agent

    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "python:3.12-slim")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_ENGINE", "docker")

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return MagicMock()

        make_computer_runtime = Agent.make_computer_runtime

    runtime = _StubAgent().make_computer_runtime()
    assert isinstance(runtime.backend, ContainerComputerBackend)
    assert runtime.backend.image == "python:3.12-slim"
    assert runtime.backend.runtime == "docker"


def test_make_computer_runtime_container_uses_configured_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agent import Agent

    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "img")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_WORKSPACE", "/code")

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return MagicMock()

        make_computer_runtime = Agent.make_computer_runtime

    runtime = _StubAgent().make_computer_runtime()
    assert isinstance(runtime.backend, ContainerComputerBackend)
    assert runtime.backend.container_workspace == "/code"


def test_make_computer_runtime_container_uses_production_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production path uses SubprocessContainerRunner, not FakeContainerRunner."""
    from core.agent import Agent
    from core.container_computer_backend import SubprocessContainerRunner

    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "img")

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return MagicMock()

        make_computer_runtime = Agent.make_computer_runtime

    runtime = _StubAgent().make_computer_runtime()
    assert isinstance(runtime.backend, ContainerComputerBackend)
    assert isinstance(runtime.backend.runner, SubprocessContainerRunner)


def test_make_computer_runtime_invalid_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agent import Agent

    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "bad-value")

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return MagicMock()

        make_computer_runtime = Agent.make_computer_runtime

    with pytest.raises(ValueError, match="SLAVIK_COMPUTER_BACKEND"):
        _StubAgent().make_computer_runtime()


def test_make_computer_runtime_container_missing_image_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agent import Agent

    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.delenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", raising=False)

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return MagicMock()

        make_computer_runtime = Agent.make_computer_runtime

    with pytest.raises(ValueError, match="SLAVIK_COMPUTER_CONTAINER_IMAGE"):
        _StubAgent().make_computer_runtime()
