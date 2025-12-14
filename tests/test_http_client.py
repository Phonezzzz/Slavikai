from __future__ import annotations

from tools.http_client import HttpClient, HttpConfig


class DummyResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self._text = text
        self.status_code = status_code
        self.headers = {}

    def iter_content(self, chunk_size=4096, decode_unicode=True):  # noqa: ANN001
        yield self._text

    def raise_for_status(self) -> None:
        return None


def test_http_client_json_limit(monkeypatch) -> None:
    payload = "{" + '"k":"v"' * 600_000 + "}"
    response = DummyResponse(payload)

    def fake_request(method, url, timeout, stream, **kwargs):  # noqa: ANN001
        return response

    monkeypatch.setattr("tools.http_client.requests.request", fake_request)
    client = HttpClient(HttpConfig(max_json_bytes=1024))
    result = client.post_json("https://example.com")
    assert not result.ok
    assert result.error == "payload_too_large"


def test_http_client_json_ok(monkeypatch) -> None:
    response = DummyResponse('{"ok": true}')

    def fake_request(method, url, timeout, stream, **kwargs):  # noqa: ANN001
        return response

    monkeypatch.setattr("tools.http_client.requests.request", fake_request)
    client = HttpClient()
    result = client.post_json("https://example.com")
    assert result.ok
    assert isinstance(result.data, dict)
