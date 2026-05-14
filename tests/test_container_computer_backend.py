"""Tests for ContainerComputerBackend.

No Docker daemon is required — all container invocations are intercepted by
FakeContainerRunner. File operations (list/read/write) use a real tmp_path
directory because they access the host filesystem directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from core.computer_backend import ComputerBackend
from core.container_computer_backend import (
    ContainerComputerBackend,
    SubprocessContainerRunner,
)

# ── Fake runner (no Docker needed) ──────────────────────────────────────────


@dataclass
class FakeContainerRunner:
    """Records every call and returns pre-configured responses."""

    responses: list[tuple[int, str, str]] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def run(self, args: list[str]) -> tuple[int, str, str]:
        self.calls.append(list(args))
        if self.responses:
            return self.responses.pop(0)
        return 0, "fake output", ""


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def fake_runner() -> FakeContainerRunner:
    return FakeContainerRunner()


@pytest.fixture()
def backend(tmp_path: Path, fake_runner: FakeContainerRunner) -> ContainerComputerBackend:
    return ContainerComputerBackend(
        image="test-image:latest",
        project_root=tmp_path,
        runner=fake_runner,
    )


# ── Protocol compliance ──────────────────────────────────────────────────────


def test_container_backend_satisfies_computer_backend_protocol(
    backend: ContainerComputerBackend,
) -> None:
    assert isinstance(backend, ComputerBackend)


def test_subprocess_runner_satisfies_protocol() -> None:
    runner = SubprocessContainerRunner()
    # structural check: has a run() method
    assert callable(runner.run)


# ── list_files ───────────────────────────────────────────────────────────────


def test_list_files_root_returns_entries(tmp_path: Path, fake_runner: FakeContainerRunner) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("hi")
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    result = b.list_files()
    assert result.ok
    tree = result.data["tree"]
    assert isinstance(tree, list)
    names = [e["name"] for e in tree]  # type: ignore[index]
    assert "src" in names
    assert "README.md" in names


def test_list_files_subdir(tmp_path: Path, fake_runner: FakeContainerRunner) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("")
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    result = b.list_files("pkg")
    assert result.ok
    names = [e["name"] for e in result.data["tree"]]  # type: ignore[index]
    assert "a.py" in names


def test_list_files_missing_path_returns_failure(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    result = b.list_files("does_not_exist")
    assert not result.ok


def test_list_files_path_traversal_rejected(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    with pytest.raises(ValueError, match="outside project root"):
        b.list_files("../../etc/passwd")


# ── read_file ────────────────────────────────────────────────────────────────


def test_read_file_returns_content(tmp_path: Path, fake_runner: FakeContainerRunner) -> None:
    (tmp_path / "hello.py").write_text("print('hello')")
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    result = b.read_file("hello.py")
    assert result.ok
    assert result.data["output"] == "print('hello')"


def test_read_file_missing_returns_failure(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    result = b.read_file("no_such_file.py")
    assert not result.ok


def test_read_file_path_traversal_rejected(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    with pytest.raises(ValueError, match="outside project root"):
        b.read_file("../../etc/shadow")


# ── write_file ───────────────────────────────────────────────────────────────


def test_write_file_creates_file_on_host(tmp_path: Path, fake_runner: FakeContainerRunner) -> None:
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    result = b.write_file("out.txt", "content")
    assert result.ok
    assert (tmp_path / "out.txt").read_text() == "content"


def test_write_file_creates_subdirectories(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    result = b.write_file("deep/nested/file.txt", "data")
    assert result.ok
    assert (tmp_path / "deep" / "nested" / "file.txt").read_text() == "data"


def test_write_file_path_traversal_rejected(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    with pytest.raises(ValueError, match="outside project root"):
        b.write_file("../../evil.txt", "bad")


# ── apply_patch ──────────────────────────────────────────────────────────────


def test_apply_patch_calls_runner_with_patch_command(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    (tmp_path / "src.txt").write_text("original")
    fake_runner.responses = [(0, "", "")]
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    b.apply_patch("src.txt", "@@ -1 +1 @@\n-original\n+replaced\n")
    assert len(fake_runner.calls) == 1
    cmd_args = fake_runner.calls[0]
    # verify docker run invocation and path mapping
    assert "docker" in cmd_args[0]
    assert "run" in cmd_args
    assert "--rm" in cmd_args
    # the command passed to sh -c must contain "patch"
    shell_cmd = cmd_args[-1]
    assert "patch" in shell_cmd
    assert "/workspace/src.txt" in shell_cmd


def test_apply_patch_path_traversal_rejected(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    with pytest.raises(ValueError, match="outside project root"):
        b.apply_patch("../../evil.txt", "diff")


# ── run_command ──────────────────────────────────────────────────────────────


def test_run_command_calls_runner_with_docker_run(
    backend: ContainerComputerBackend, fake_runner: FakeContainerRunner
) -> None:
    fake_runner.responses = [(0, "hello", "")]
    result = backend.run_command("echo hello")
    assert result.ok
    assert result.data["output"] == "hello"
    assert len(fake_runner.calls) == 1
    args = fake_runner.calls[0]
    assert args[0] == "docker"
    assert "run" in args
    assert "--rm" in args
    assert "test-image:latest" in args
    assert "echo hello" in args


def test_run_command_mounts_project_root(
    backend: ContainerComputerBackend,
    fake_runner: FakeContainerRunner,
    tmp_path: Path,
) -> None:
    backend.run_command("ls")
    args = fake_runner.calls[0]
    volume_arg = f"{tmp_path}:/workspace"
    assert volume_arg in args


def test_run_command_sets_workdir_to_container_workspace(
    backend: ContainerComputerBackend, fake_runner: FakeContainerRunner
) -> None:
    backend.run_command("pwd")
    args = fake_runner.calls[0]
    idx = args.index("-w")
    assert args[idx + 1] == "/workspace"


def test_run_command_failure_returns_error(
    backend: ContainerComputerBackend, fake_runner: FakeContainerRunner
) -> None:
    fake_runner.responses = [(1, "", "command not found")]
    result = backend.run_command("no_such_cmd")
    assert not result.ok
    assert "command not found" in (result.error or "")


def test_run_command_runner_exception_returns_failure(
    backend: ContainerComputerBackend, fake_runner: FakeContainerRunner
) -> None:
    fake_runner.responses = []

    def _boom(args: list[str]) -> tuple[int, str, str]:
        raise RuntimeError("docker not found")

    fake_runner.run = _boom  # type: ignore[method-assign]
    result = backend.run_command("ls")
    assert not result.ok
    assert "docker not found" in (result.error or "")


# ── git_diff ─────────────────────────────────────────────────────────────────


def test_git_diff_no_path_calls_git_diff(
    backend: ContainerComputerBackend, fake_runner: FakeContainerRunner
) -> None:
    backend.git_diff()
    shell_cmd = fake_runner.calls[0][-1]
    assert shell_cmd == "git diff"


def test_git_diff_with_path_includes_path(
    backend: ContainerComputerBackend, fake_runner: FakeContainerRunner
) -> None:
    backend.git_diff("src/main.py")
    shell_cmd = fake_runner.calls[0][-1]
    assert shell_cmd == "git diff -- src/main.py"


# ── run_tests ────────────────────────────────────────────────────────────────


def test_run_tests_no_path_calls_pytest(
    backend: ContainerComputerBackend, fake_runner: FakeContainerRunner
) -> None:
    backend.run_tests()
    shell_cmd = fake_runner.calls[0][-1]
    assert shell_cmd == "pytest"


def test_run_tests_with_path_calls_pytest_path(
    backend: ContainerComputerBackend, fake_runner: FakeContainerRunner
) -> None:
    backend.run_tests("tests/test_foo.py")
    shell_cmd = fake_runner.calls[0][-1]
    assert shell_cmd == "pytest tests/test_foo.py"


# ── check ────────────────────────────────────────────────────────────────────


def test_check_calls_make_check(
    backend: ContainerComputerBackend, fake_runner: FakeContainerRunner
) -> None:
    backend.check()
    shell_cmd = fake_runner.calls[0][-1]
    assert shell_cmd == "make check"


# ── podman support ───────────────────────────────────────────────────────────


def test_podman_runtime_used_in_args(tmp_path: Path, fake_runner: FakeContainerRunner) -> None:
    b = ContainerComputerBackend(
        image="img", project_root=tmp_path, runtime="podman", runner=fake_runner
    )
    b.run_command("ls")
    args = fake_runner.calls[0]
    assert args[0] == "podman"


# ── custom container_workspace ───────────────────────────────────────────────


def test_custom_container_workspace_used_in_volume_and_workdir(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    b = ContainerComputerBackend(
        image="img",
        project_root=tmp_path,
        container_workspace="/code",
        runner=fake_runner,
    )
    b.run_command("ls")
    args = fake_runner.calls[0]
    assert f"{tmp_path}:/code" in args
    idx = args.index("-w")
    assert args[idx + 1] == "/code"
