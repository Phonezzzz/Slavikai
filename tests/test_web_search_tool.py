from __future__ import annotations

from typing import Any

from config.web_search_config import WebSearchConfig
from shared.models import ToolRequest
from tools.http_client import HttpResult
from tools.web_search_tool import SERPER_ENDPOINT, WebSearchTool


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post_json(self, url: str, **kwargs: Any) -> HttpResult:
        self.calls.append((url, kwargs))
        payload = {
            "organic": [
                {"title": "Result 1", "link": "https://example.com/1", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "https://example.com/2", "snippet": "Snippet 2"},
            ]
        }
        return HttpResult(ok=True, data=payload, status_code=200, error=None, headers={}, meta={})

    def get_text(self, url: str, **kwargs: Any) -> HttpResult:
        self.calls.append((url, kwargs))
        return HttpResult(
            ok=True,
            data="hello world",
            status_code=200,
            error=None,
            headers={},
            meta={},
        )


def test_web_search_serper() -> None:
    fake_http = FakeHttpClient()
    config = WebSearchConfig(api_key="test-key")
    tool = WebSearchTool(config=config, http_client=fake_http)
    result = tool.handle(ToolRequest(name="web", args={"query": "hello"}))
    assert result.ok
    assert fake_http.calls[0][0] == SERPER_ENDPOINT
    assert fake_http.calls[0][1]["headers"]["X-API-KEY"] == "test-key"
    assert len(result.data.get("results") or []) == 2
    scores = [entry["score"] for entry in result.data.get("results") or []]
    assert scores == sorted(scores, reverse=True)


def test_web_search_bad_url() -> None:
    tool = WebSearchTool()
    result = tool.handle(ToolRequest(name="web", args={"query": "ftp://example.com"}))
    assert not result.ok


def test_web_fetch() -> None:
    fake_http = FakeHttpClient()
    tool = WebSearchTool(http_client=fake_http)
    result = tool.handle(ToolRequest(name="web", args={"query": "https://example.com"}))
    assert result.ok
    assert "hello world" in str(result.data.get("output"))
