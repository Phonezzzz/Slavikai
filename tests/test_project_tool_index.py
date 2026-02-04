from __future__ import annotations

import shutil
import uuid

from shared.models import ToolRequest
from shared.sandbox import WORKSPACE_ROOT
from tools.project_tool import handle_project_request


class DummyVectorIndex:
    def __init__(self, db_path: str = "memory/vectors.db"):  # noqa: ARG002
        self.indexed = []

    def index_text(self, path: str, content: str, namespace: str = "default", meta=None) -> None:
        self.indexed.append((path, namespace, content[:10]))

    def search(self, query: str, namespace: str = "default", top_k: int = 5):  # noqa: ARG002
        return []


def test_project_index_uses_vector_index(tmp_path, monkeypatch) -> None:
    del tmp_path
    base = WORKSPACE_ROOT / f"test_project_index_{uuid.uuid4().hex}"
    try:
        base.mkdir()
        (base / "a.py").write_text("print('hi')\n", encoding="utf-8")
        (base / "readme.md").write_text("# doc\n", encoding="utf-8")

        monkeypatch.setattr("tools.project_tool.VectorIndex", DummyVectorIndex)

        req = ToolRequest(
            name="project",
            args={"cmd": "index", "args": [str(base.relative_to(WORKSPACE_ROOT))]},
        )
        result = handle_project_request(req)
        assert result.ok
        assert result.data.get("indexed_code") == 1
        assert result.data.get("indexed_docs") == 1
    finally:
        shutil.rmtree(base, ignore_errors=True)
