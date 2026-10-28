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


def test_apply_patch_path_traversal_rejected(
    tmp_path: Path, fake_runner: FakeContainerRunner
) -> None:
    b = ContainerComputerBackend(image="img", project_root=tmp_path, runner=fake_runner)
    with pytest.raises(ValueError, match="outside project root"):
        b.apply_patch("../../evil.txt", "diff")


@pytest.mark.parametrize("engine", ["docker", "podman"])
@pytest.mark.parametrize(
    "operation", ["run_command", "git_diff", "run_tests", "check", "apply_patch"]
)
def test_container_process_operations_cannot_use_even_injected_runner(
    tmp_path, fake_runner, engine, operation
):
    b = ContainerComputerBackend(
        image="unqualified", project_root=tmp_path, runtime=engine, runner=fake_runner
    )
    before = list(tmp_path.iterdir())
    if operation == "apply_patch":
        result = b.apply_patch("file.txt", "patch")
    elif operation == "check":
        result = b.check()
    else:
        result = getattr(b, operation)("echo data" if operation == "run_command" else "")
    assert not result.ok
    assert result.meta["reason_code"] == "sandbox_execution_unavailable"
    assert fake_runner.calls == []
    assert list(tmp_path.iterdir()) == before
