from __future__ import annotations

from dataclasses import dataclass

from core.computer_backend import ComputerBackend
from shared.models import ToolResult


@dataclass
class AgentComputerRuntime:
    """Backend-agnostic facade for Computer operations.

    Callers use this class without knowing whether execution is local,
    containerised, or sandboxed. The backend decides the execution path.
    """

    backend: ComputerBackend

    def list_files(self, path: str = "") -> ToolResult:
        return self.backend.list_files(path)

    def read_file(self, path: str) -> ToolResult:
        return self.backend.read_file(path)

    def write_file(self, path: str, content: str) -> ToolResult:
        return self.backend.write_file(path, content)

    def apply_patch(self, path: str, patch: str) -> ToolResult:
        return self.backend.apply_patch(path, patch)

    def run_command(self, command: str) -> ToolResult:
        return self.backend.run_command(command)

    def git_diff(self, path: str = "") -> ToolResult:
        return self.backend.git_diff(path)

    def run_tests(self, path: str = "") -> ToolResult:
        return self.backend.run_tests(path)

    def check(self) -> ToolResult:
        return self.backend.check()
