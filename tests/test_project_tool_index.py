from __future__ import annotations

import pytest

from shared.models import ToolRequest
from tools.project_tool import handle_project_request


class DummyVectorIndex:
    def __init__(self, db_path: str = "memory/vectors.db", **_: object):  # noqa: ARG002
        self.indexed = []

    def ensure_runtime_ready(self) -> None:
        return

    def index_text(self, path: str, content: str, namespace: str = "default", meta=None) -> None:
        self.indexed.append((path, namespace, content[:10]))

    def search(self, query: str, namespace: str = "default", top_k: int = 5):  # noqa: ARG002
        return []


def test_project_index_uses_vector_index(tmp_path, monkeypatch) -> None:
    sandbox_root = (tmp_path / "sandbox" / "project").resolve()
    base = sandbox_root / "proj"
    base.mkdir(parents=True)
    (base / "a.py").write_text("print('hi')", encoding="utf-8")
    (base / "readme.md").write_text("# doc", encoding="utf-8")

    monkeypatch.setattr("tools.project_tool.SANDBOX_ROOT", sandbox_root)
    monkeypatch.setattr("tools.project_tool.VectorIndex", DummyVectorIndex)

    req = ToolRequest(name="project", args={"cmd": "index", "args": ["proj"]})
    result = handle_project_request(req)
    assert result.ok
    assert result.data.get("indexed_code") == 1
    assert result.data.get("indexed_docs") == 1


def test_project_index_rejects_absolute_path(tmp_path, monkeypatch) -> None:
    sandbox_root = (tmp_path / "sandbox" / "project").resolve()
    sandbox_root.mkdir(parents=True)
    monkeypatch.setattr("tools.project_tool.SANDBOX_ROOT", sandbox_root)
    monkeypatch.setattr("tools.project_tool.VectorIndex", DummyVectorIndex)

    req = ToolRequest(name="project", args={"cmd": "index", "args": [str(tmp_path.resolve())]})
    result = handle_project_request(req)
    assert not result.ok
    assert "sandbox/project" in (result.error or "")


def test_project_index_rejects_symlink_escape(tmp_path, monkeypatch) -> None:
    sandbox_root = (tmp_path / "sandbox" / "project").resolve()
    sandbox_root.mkdir(parents=True)
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    (outside / "leak.py").write_text("print('leak')", encoding="utf-8")
    escape_link = sandbox_root / "escape"
    try:
        escape_link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink недоступен в этом окружении.")

    monkeypatch.setattr("tools.project_tool.SANDBOX_ROOT", sandbox_root)
    monkeypatch.setattr("tools.project_tool.VectorIndex", DummyVectorIndex)

    req = ToolRequest(name="project", args={"cmd": "index", "args": ["escape"]})
    result = handle_project_request(req)
    assert not result.ok
    assert "sandbox/project" in (result.error or "")
