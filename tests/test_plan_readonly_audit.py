from __future__ import annotations

from pathlib import Path

import pytest

from server.http.common.workspace_runtime import _run_plan_readonly_audit


def _audit(root: Path) -> tuple[list[dict], dict[str, int]]:
    return _run_plan_readonly_audit(
        root=root,
        plan_audit_timeout_seconds=5,
        workspace_index_ignored_dirs={"venv", "__pycache__", ".git", "node_modules"},
        workspace_index_allowed_extensions={".py", ".md", ".txt"},
        plan_audit_max_total_bytes=100_000,
        plan_audit_max_read_files=100,
    )


def test_plan_audit_reads_allowed_files(tmp_path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("print('x')", encoding="utf-8")
    (root / "doc.md").write_text("# doc", encoding="utf-8")
    (root / "blob.bin").write_bytes(b"\x00\x01\x02")

    entries, usage = _audit(root)
    paths = [entry["path"] for entry in entries]
    assert "a.py" in paths
    assert "doc.md" in paths
    assert not any(path.endswith("blob.bin") for path in paths)
    assert usage["read_files"] == 2


def test_plan_audit_skips_symlink_escape(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "ok.py").write_text("print('ok')", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.md"
    secret.write_text("TOP SECRET", encoding="utf-8")
    link = root / "leak.md"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("Symlink недоступен в этом окружении.")

    entries, _ = _audit(root)
    paths = [entry["path"] for entry in entries]
    assert not any(path.endswith("leak.md") for path in paths), paths
    assert any(path.endswith("ok.py") for path in paths)
