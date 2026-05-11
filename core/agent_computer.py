from __future__ import annotations

from dataclasses import dataclass

from core.tool_gateway import ToolGateway
from shared.models import ToolRequest, ToolResult


@dataclass
class AgentComputerRuntime:
    """Facade over existing workspace tools via ToolGateway.

    Each method maps a named Computer operation to a ToolRequest and delegates
    through the gateway. No new tools are registered, no new runtime paths exist.
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
