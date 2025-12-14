from __future__ import annotations

from typing import Any

from shared.models import ToolRequest, VectorSearchResult
from tools.project_tool import handle_project_request


class FakeVectorIndex:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def search(self, query: str, namespace: str = "default", top_k: int = 5):
        return [
            VectorSearchResult(path=f"{namespace}/file.py", snippet=f"{query} snippet", score=0.9)
        ]


def test_project_find_uses_vectors(monkeypatch) -> None:
    monkeypatch.setattr("tools.project_tool.VectorIndex", FakeVectorIndex)
    req = ToolRequest(name="project", args={"cmd": "find", "args": ["foo"]})
    result = handle_project_request(req)
    assert result.ok
    assert "matches" in result.data
    assert result.meta and result.meta.get("matches") == 2  # code + docs namespaces
