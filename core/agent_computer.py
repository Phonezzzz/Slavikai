from __future__ import annotations

import uuid
from dataclasses import dataclass

from core.computer_backend import ComputerBackend
from shared.models import JSONValue, ToolResult


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


def build_computer_changes_review_decision(
    changed_files: list[str],
    diff_summary: str,
    commit_message: str,
) -> dict[str, JSONValue] | None:
    """Build a decision packet for 'changes ready for review'.

    Returns None when there is nothing to review (empty files or blank commit message).
    Pure function — no I/O, no git, no commit side effects.
    """
    if not changed_files or not commit_message.strip():
        return None

    return {
        "id": str(uuid.uuid4()),
        "kind": "decision",
        "decision_type": "agent_decision",
        "status": "pending",
        "blocking": True,
        "reason": "computer_changes_review",
        "summary": f"Changes ready to commit: {commit_message.strip()[:72]}",
        "proposed_action": {
            "category": "computer_changes_review",
            "changed_files": changed_files,
            "diff_summary": diff_summary,
            "proposed_commit_message": commit_message.strip(),
        },
        "options": [
            {
                "id": "approve_once",
                "title": "Commit",
                "action": "approve",
                "payload": {},
                "risk": "low",
            },
            {
                "id": "reject",
                "title": "Cancel",
                "action": "reject",
                "payload": {},
                "risk": "low",
            },
        ],
        "default_option_id": "approve_once",
        "context": {},
        "created_at": None,
        "expires_at": None,
        "updated_at": None,
        "resolved_at": None,
    }
