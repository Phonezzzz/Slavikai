from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from shared.models import ToolRequest
from shared.sandbox import WORKSPACE_ROOT
from tools.project_tool import handle_project_request


class _DummyVectorIndex:
    def __init__(self, db_path: str = "memory/vectors.db") -> None:  # noqa: ARG002
        self.indexed: list[tuple[str, str]] = []

    def index_text(self, path: str, content: str, namespace: str = "default", meta=None) -> None:  # noqa: ANN001, ARG002
        self.indexed.append((path, namespace))

    def search(self, query: str, namespace: str = "default", top_k: int = 5):  # noqa: ARG002
        return []


def test_project_index_rejects_parent_reference(monkeypatch) -> None:
    monkeypatch.setattr("tools.project_tool.VectorIndex", _DummyVectorIndex)
    req = ToolRequest(name="project", args={"cmd": "index", "args": ["../outside"]})
    result = handle_project_request(req)
    assert not result.ok
    assert "sandbox/project" in (result.error or "")


def test_project_index_rejects_absolute_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("tools.project_tool.VectorIndex", _DummyVectorIndex)
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    req = ToolRequest(name="project", args={"cmd": "index", "args": [str(outside.resolve())]})
    result = handle_project_request(req)
    assert not result.ok
    assert "sandbox/project" in (result.error or "")


def test_project_index_allows_workspace_relative_path(monkeypatch) -> None:
    monkeypatch.setattr("tools.project_tool.VectorIndex", _DummyVectorIndex)
    rel_dir = Path(f"test_project_tool_{uuid.uuid4().hex}")
    base = WORKSPACE_ROOT / rel_dir
    try:
        base.mkdir(parents=True, exist_ok=True)
        (base / "a.py").write_text("print('ok')\n", encoding="utf-8")

        req = ToolRequest(name="project", args={"cmd": "index", "args": [str(rel_dir)]})
        result = handle_project_request(req)
        assert result.ok
        assert result.data.get("indexed_code") == 1
    finally:
        shutil.rmtree(base, ignore_errors=True)
