from __future__ import annotations

import asyncio

import pytest

from server.http.common import github_import


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path, monkeypatch) -> None:
    github_root = tmp_path / "sandbox" / "project" / "github"
    project_root = tmp_path / "sandbox" / "project"
    github_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(github_import, "GITHUB_ROOT", github_root.resolve())
    monkeypatch.setattr(github_import, "SANDBOX_PROJECT_ROOT", project_root.resolve())


def test_resolve_github_target_returns_same_path_for_existing_git_repo() -> None:
    repo_dir = github_import.GITHUB_ROOT / "owner" / "myrepo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / ".git").mkdir()

    target, relative = github_import.resolve_github_target("https://github.com/owner/myrepo.git")

    assert target == repo_dir
    assert relative.startswith("github/owner/myrepo")


def test_resolve_github_target_suffix_only_for_non_git_existing_dir() -> None:
    existing = github_import.GITHUB_ROOT / "owner" / "other"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "user_data.txt").write_text("keep me", encoding="utf-8")

    target, _ = github_import.resolve_github_target("https://github.com/owner/other")

    assert target != existing
    assert target.name == "other-1"
    assert (existing / "user_data.txt").read_text(encoding="utf-8") == "keep me"


def test_resolve_github_target_fresh_clone_target() -> None:
    target, relative = github_import.resolve_github_target("https://github.com/o/fresh")

    assert not target.exists()
    assert target.name == "fresh"


def test_clone_github_repository_refreshes_existing_git_repo(monkeypatch) -> None:
    repo_dir = github_import.GITHUB_ROOT / "owner" / "existing"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / ".git").mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd, **_: object):
        calls.append(cmd)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})

    monkeypatch.setattr("asyncio.to_thread", lambda fn, *a, **k: _run_inline(fn, a, k))
    monkeypatch.setattr("subprocess.run", fake_run)

    ok, result = asyncio.run(
        github_import.clone_github_repository(
            repo_url="https://github.com/owner/existing",
            branch="main",
            target_path=repo_dir,
        )
    )

    assert ok
    assert result == "ok"
    assert calls == [
        ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin"],
        ["git", "-C", str(repo_dir), "reset", "--hard", "origin/main"],
    ]


async def _run_inline(fn, args, kwargs):
    return fn(*args, **kwargs)
