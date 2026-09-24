from __future__ import annotations

from tools.http_client import _redact_text, _redact_url


def test_redact_url_strips_sensitive_query_param() -> None:
    url = "https://serpapi.com/search.json?engine=google&q=hello&api_key=SECRETKEY"
    redacted = _redact_url(url)
    assert "SECRETKEY" not in redacted
    assert "api_key=%5BREDACTED%5D" in redacted or "api_key=[REDACTED]" in redacted


def test_redact_text_strips_sensitive_urls() -> None:
    text = "500 Server Error for url: https://serpapi.com/search.json?q=hello&api_key=SECRETKEY"
    redacted = _redact_text(text)
    assert "SECRETKEY" not in redacted
