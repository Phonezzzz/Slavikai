"""Tests for AgentComputerRuntime facade.

Verifies that each method constructs the correct ToolRequest and delegates
through ToolGateway — testing the mechanism, not just the effect.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.agent_computer import AgentComputerRuntime
from shared.models import ToolRequest, ToolResult


@pytest.fixture()
def gateway() -> MagicMock:
    mock = MagicMock()
    mock.call.return_value = ToolResult.success({"output": "ok"})
    return mock


@pytest.fixture()
def computer(gateway: MagicMock) -> AgentComputerRuntime:
    return AgentComputerRuntime(gateway=gateway)


def test_list_files_calls_workspace_list(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.list_files("src/")
    gateway.call.assert_called_once_with(ToolRequest("workspace_list", {"path": "src/"}))


def test_list_files_default_path(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    computer.list_files()
    gateway.call.assert_called_once_with(ToolRequest("workspace_list", {"path": ""}))


def test_read_file_calls_workspace_read(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    computer.read_file("src/main.py")
    gateway.call.assert_called_once_with(ToolRequest("workspace_read", {"path": "src/main.py"}))


def test_write_file_calls_workspace_write(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.write_file("src/foo.py", "hello")
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_write", {"path": "src/foo.py", "content": "hello"})
    )


def test_apply_patch_calls_workspace_patch(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    patch = "@@ -1 +1 @@\n-old\n+new"
    computer.apply_patch("src/foo.py", patch)
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_patch", {"path": "src/foo.py", "patch": patch})
    )


def test_run_command_calls_workspace_terminal_run(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.run_command("pytest tests/")
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "pytest tests/"})
    )


def test_facade_returns_gateway_result(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    expected = ToolResult.success({"output": "test output"})
    gateway.call.return_value = expected
    result = computer.read_file("foo.py")
    assert result is expected


def test_facade_propagates_failure(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    gateway.call.return_value = ToolResult.failure("tool error")
    result = computer.write_file("foo.py", "x")
    assert not result.ok
    assert result.error == "tool error"


def test_each_method_calls_gateway_exactly_once(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.list_files()
    assert gateway.call.call_count == 1

    computer.read_file("a.py")
    assert gateway.call.call_count == 2

    computer.write_file("b.py", "")
    assert gateway.call.call_count == 3

    computer.apply_patch("c.py", "@@ @@")
    assert gateway.call.call_count == 4

    computer.run_command("echo hi")
    assert gateway.call.call_count == 5


def test_git_diff_with_path_calls_git_diff_path(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.git_diff("src/foo.py")
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "git diff -- src/foo.py"})
    )


def test_git_diff_no_path_calls_git_diff(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.git_diff()
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "git diff"})
    )


def test_run_tests_with_path_calls_pytest_path(
    computer: AgentComputerRuntime, gateway: MagicMock
) -> None:
    computer.run_tests("tests/test_foo.py")
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "pytest tests/test_foo.py"})
    )


def test_run_tests_no_path_calls_pytest(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    computer.run_tests()
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "pytest"})
    )


def test_check_calls_make_check(computer: AgentComputerRuntime, gateway: MagicMock) -> None:
    computer.check()
    gateway.call.assert_called_once_with(
        ToolRequest("workspace_terminal_run", {"command": "make check"})
    )


def test_make_computer_runtime_returns_runtime_with_gateway() -> None:
    from core.agent import Agent

    gateway_mock = MagicMock()

    class _StubAgent:
        def _build_tool_gateway(self) -> MagicMock:
            return gateway_mock

        make_computer_runtime = Agent.make_computer_runtime

    stub = _StubAgent()
    runtime = stub.make_computer_runtime()
    assert isinstance(runtime, AgentComputerRuntime)
    assert runtime.gateway is gateway_mock
