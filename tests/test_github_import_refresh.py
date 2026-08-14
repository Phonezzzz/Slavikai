from __future__ import annotations

import pytest

from server.http.common import github_import


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path, monkeypatch) -> None:
    github_root = tmp_path / "sandbox" / "project" / "github"
    project_root = tmp_path / "sandbox" / "project"
    github_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(github_import, "GITHUB_ROOT", github_root.resolve())
    monkeypatch.setattr(github_import, "SANDBOX_PROJECT_ROOT", project_root.resolve())


def test_repeated_import_of_existing_git_repo_is_refused() -> None:
    repo_dir = github_import.GITHUB_ROOT / "owner" / "myrepo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / ".git").mkdir()
    user_file = repo_dir / "user_note.md"
    user_file.write_text("user edits", encoding="utf-8")

    with pytest.raises(ValueError, match="уже импортирован"):
        github_import.resolve_github_target("https://github.com/owner/myrepo.git")

    assert (repo_dir / "user_note.md").read_text(encoding="utf-8") == "user edits"
    assert not (repo_dir.parent / "myrepo-1").exists()


def test_non_git_existing_dir_gets_suffix_and_preserves_data() -> None:
    existing = github_import.GITHUB_ROOT / "owner" / "other"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "user_data.txt").write_text("keep me", encoding="utf-8")

    target, _ = github_import.resolve_github_target("https://github.com/owner/other")

    assert target != existing
    assert target.name == "other-1"
    assert (existing / "user_data.txt").read_text(encoding="utf-8") == "keep me"


def test_fresh_clone_target() -> None:
    target, relative = github_import.resolve_github_target("https://github.com/o/fresh")

    assert not target.exists()
    assert target.name == "fresh"
    assert relative.startswith("github/o/fresh")


def test_clone_failure_does_not_delete_preexisting_target(tmp_path, monkeypatch) -> None:
    preexisting = tmp_path / "sandbox" / "project" / "existing_dir"
    preexisting.mkdir(parents=True)
    (preexisting / "important.txt").write_text("data", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run(cmd, **_: object):
        calls.append(cmd)
        return type(
            "R",
            (),
            {"returncode": 1, "stdout": "clone failed", "stderr": ""},
        )

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("asyncio.to_thread", lambda fn, *a, **k: fn(*a, **k))

    import asyncio

    ok, _ = asyncio.run(
        github_import.clone_github_repository(
            repo_url="https://github.com/o/x",
            branch=None,
            target_path=preexisting,
        )
    )

    assert not ok
    assert (preexisting / "important.txt").read_text(encoding="utf-8") == "data"
