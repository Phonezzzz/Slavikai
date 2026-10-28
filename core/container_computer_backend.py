from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from core.computer_backend import ComputerBackend  # noqa: F401 (re-exported for type checks)
from shared.models import JSONValue, ToolResult
from shared.sandbox_execution import SandboxExecutionUnavailable, sandbox_execution_failure


class ContainerCommandRunner(Protocol):
    """Injectable CLI runner. Production uses subprocess; tests use a fake."""

    def run(self, args: list[str]) -> tuple[int, str, str]:
        """Return (returncode, stdout, stderr)."""
        ...


@dataclass
class SubprocessContainerRunner:
    """Неквалифицированный runner: host CLI не запускается."""

    timeout: int = 60

    def run(self, args: list[str]) -> tuple[int, str, str]:
        raise SandboxExecutionUnavailable()


@dataclass
class ContainerComputerBackend:
    """Opt-in factory candidate без квалифицированной process environment.

    File operations используют прежние host path guards. Command operations
    закрыты I0 fence независимо от image/engine/injected runner.
    """

    image: str
    project_root: Path
    container_workspace: str = "/workspace"
    runtime: str = "docker"
    runner: ContainerCommandRunner = field(default_factory=SubprocessContainerRunner)

    # ── Path guards ───────────────────────────────────────────────────────────

    def _ensure_in_project(self, rel_path: str) -> Path:
        """Resolve rel_path under project_root, rejecting traversal."""
        if not rel_path:
            return self.project_root
        candidate = (self.project_root / rel_path).resolve()
        try:
            candidate.relative_to(self.project_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Path outside project root: {rel_path!r}") from exc
        return candidate

    def _container_path(self, rel_path: str) -> str:
        """Map a relative host path to its container-side equivalent."""
        if not rel_path:
            return self.container_workspace
        normalized = rel_path.strip().lstrip("/")
        if ".." in Path(normalized).parts:
            raise ValueError(f"Path traversal not allowed: {rel_path!r}")
        return f"{self.container_workspace}/{normalized}"

    # ── Container command helper ──────────────────────────────────────────────

    def _run_in_container(self, command: str) -> ToolResult:
        return sandbox_execution_failure()

    # ── ComputerBackend implementation ────────────────────────────────────────

    def list_files(self, path: str = "") -> ToolResult:
        target = self._ensure_in_project(path)
        if not target.exists():
            return ToolResult.failure(f"Path not found: {path!r}")
        entries: list[JSONValue] = []
        if target.is_file():
            entries.append({"name": target.name, "path": path or target.name, "type": "file"})
        else:
            for child in sorted(target.iterdir()):
                rel = str(child.relative_to(self.project_root))
                entries.append(
                    {
                        "name": child.name,
                        "path": rel,
                        "type": "dir" if child.is_dir() else "file",
                    }
                )
        return ToolResult.success(
            {"tree": entries, "meta": {"truncated": False, "truncated_reasons": []}}
        )

    def read_file(self, path: str) -> ToolResult:
        target = self._ensure_in_project(path)
        if not target.exists() or not target.is_file():
            return ToolResult.failure(f"File not found: {path!r}")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult.failure(f"Read failed: {exc}")
        return ToolResult.success({"output": content})

    def write_file(self, path: str, content: str) -> ToolResult:
        target = self._ensure_in_project(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult.failure(f"Write failed: {exc}")
        return ToolResult.success({"output": f"written: {path}"})

    def apply_patch(self, path: str, patch: str) -> ToolResult:
        self._ensure_in_project(path)
        self._container_path(path)
        return sandbox_execution_failure()

    def run_command(self, command: str) -> ToolResult:
        return self._run_in_container(command)

    def git_diff(self, path: str = "") -> ToolResult:
        cmd = f"git diff -- {path}" if path else "git diff"
        return self._run_in_container(cmd)

    def run_tests(self, path: str = "") -> ToolResult:
        cmd = f"pytest {path}" if path else "pytest"
        return self._run_in_container(cmd)

    def check(self) -> ToolResult:
        return self._run_in_container("make check")
