"""Tests for AgentComputerRuntime facade and ComputerBackend boundary.

Verifies that:
- AgentComputerRuntime delegates every operation to its ComputerBackend.
- LocalComputerBackend constructs the correct ToolRequest and calls ToolGateway.
- Agent.make_computer_runtime() wires a LocalComputerBackend into AgentComputerRuntime.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.agent_computer import AgentComputerRuntime
from core.computer_backend import ComputerBackend, LocalComputerBackend
from shared.models import ToolRequest, ToolResult


@pytest.fixture()
def gateway() -> MagicMock:
    mock = MagicMock()
    mock.call.return_value = ToolResult.success({"output": "ok"})
    return mock


@pytest.fixture()
def backend(gateway: MagicMock) -> LocalComputerBackend:
    return LocalComputerBackend(gateway=gateway)


@pytest.fixture()
def computer(backend: LocalComputerBackend) -> AgentComputerRuntime:
    return AgentComputerRuntime(backend=backend)


# ── AgentComputerRuntime delegates to backend ────────────────────────────────


def test_runtime_delegates_list_files(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    computer.list_files("src/")
    gateway.call.assert_called_once_with(ToolRequest("workspace_list", {"path": "src/"}))


def test_runtime_delegates_list_files_default(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.list_files()
    gateway.call.assert_called_once_with(ToolRequest("workspace_list", {"path": ""}))


def test_runtime_delegates_read_file(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    computer.read_file("src/main.py")
    gateway.call.assert_called_once_with(ToolRequest("workspace_read", {"path": "src/main.py"}))


def test_runtime_delegates_write_file(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    computer.write_file("src/foo.py", "hello")
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_write", {"path": "src/foo.py", "content": "hello"})
    )


def test_runtime_delegates_apply_patch(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    patch = "@@ -1 +1 @@\n-old\n+new"
    computer.apply_patch("src/foo.py", patch)
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_patch", {"path": "src/foo.py", "patch": patch})
    )


def test_runtime_delegates_run_command(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    computer.run_command("pytest tests/")
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "pytest tests/"})
    )


def test_runtime_delegates_git_diff_with_path(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.git_diff("src/foo.py")
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "git diff -- src/foo.py"})
    )


def test_runtime_delegates_git_diff_no_path(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.git_diff()
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "git diff"})
    )


def test_runtime_delegates_run_tests_with_path(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.run_tests("tests/test_foo.py")
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "pytest tests/test_foo.py"})
    )


def test_runtime_delegates_run_tests_no_path(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.run_tests()
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "pytest"})
    )


def test_runtime_delegates_check(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    computer.check()
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "make check"})
    )


def test_runtime_returns_backend_result(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    expected = ToolResult.success({"output": "test output"})
    gateway.call.return_value = expected
    result = computer.read_file("foo.py")
    assert result is expected


def test_runtime_propagates_failure(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    gateway.call.return_value = ToolResult.failure("tool error")
    result = computer.write_file("foo.py", "x")
    assert not result.ok
    assert result.error == "tool error"


def test_runtime_calls_gateway_once_per_operation(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.list_files()
    computer.read_file("a.py")
    computer.write_file("b.py", "")
    computer.apply_patch("c.py", "@@ @@")
    computer.run_command("echo hi")
    assert gateway.call.call_count == 5


# ── LocalComputerBackend satisfies ComputerBackend protocol ──────────────────


def test_local_backend_is_computer_backend(backend: LocalComputerBackend) -> None:
    assert isinstance(backend, ComputerBackend)


def test_local_backend_holds_gateway(backend: LocalComputerBackend, gateway: MagicMock) -> None:
    assert backend.gateway is gateway


# ── Agent.make_computer_runtime wires LocalComputerBackend ───────────────────


def test_make_computer_runtime_returns_runtime_with_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.agent import Agent

    monkeypatch.delenv("SLAVIK_COMPUTER_BACKEND", raising=False)
    gateway_mock = MagicMock()

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return gateway_mock

        make_computer_runtime = Agent.make_computer_runtime

    stub = _StubAgent()
    runtime = stub.make_computer_runtime()
    assert isinstance(runtime, AgentComputerRuntime)
    assert isinstance(runtime.backend, LocalComputerBackend)
    assert runtime.backend.gateway is gateway_mock
