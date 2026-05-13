from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.tool_gateway import ToolGateway
from shared.models import ToolRequest, ToolResult


@runtime_checkable
class ComputerBackend(Protocol):
    """Execution boundary for Computer operations.

    Implementations route operations to local workspace tools, a container,
    or any other backend without callers knowing the difference.
    """

    def list_files(self, path: str = "") -> ToolResult: ...

    def read_file(self, path: str) -> ToolResult: ...

    def write_file(self, path: str, content: str) -> ToolResult: ...

    def apply_patch(self, path: str, patch: str) -> ToolResult: ...

    def run_command(self, command: str) -> ToolResult: ...

    def git_diff(self, path: str = "") -> ToolResult: ...

    def run_tests(self, path: str = "") -> ToolResult: ...

    def check(self) -> ToolResult: ...


@dataclass
class LocalComputerBackend:
    """Thin wrapper over ToolGateway for local workspace execution.

    All operations pass through the gateway, preserving approval policy,
    safe-mode, and activity logging hooks already in place.
    """

    gateway: ToolGateway

    def list_files(self, path: str = "") -> ToolResult:
        return self.gateway.call(ToolRequest("workspace_list", {"path": path}))

    def read_file(self, path: str) -> ToolResult:
        return self.gateway.call(ToolRequest("workspace_read", {"path": path}))

    def write_file(self, path: str, content: str) -> ToolResult:
        return self.gateway.call(ToolRequest("workspace_write", {"path": path, "content": content}))

    def apply_patch(self, path: str, patch: str) -> ToolResult:
        return self.gateway.call(ToolRequest("workspace_patch", {"path": path, "patch": patch}))

    def run_command(self, command: str) -> ToolResult:
        return self.gateway.call(ToolRequest("workspace_terminal_run", {"command": command}))

    def git_diff(self, path: str = "") -> ToolResult:
        cmd = f"git diff -- {path}" if path else "git diff"
        return self.gateway.call(ToolRequest("workspace_terminal_run", {"command": cmd}))

    def run_tests(self, path: str = "") -> ToolResult:
        cmd = f"pytest {path}" if path else "pytest"
        return self.gateway.call(ToolRequest("workspace_terminal_run", {"command": cmd}))

    def check(self) -> ToolResult:
        return self.gateway.call(ToolRequest("workspace_terminal_run", {"command": "make check"}))
