from __future__ import annotations

from tools.http_client import HttpClient, HttpConfig


class DummyResponse:
    def __init__(self, body: bytes, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        return

    def iter_content(self, chunk_size: int = 4096, decode_unicode: bool = False):  # noqa: ARG002
        yield self.body


def test_http_client_payload_too_large(monkeypatch) -> None:
    client = HttpClient(HttpConfig(max_bytes=5, max_json_bytes=5))

    def fake_request(method: str, url: str, timeout: int, stream: bool, **kwargs):  # noqa: ANN001
        return DummyResponse(b'{"a": 123456}', status_code=200)

    monkeypatch.setattr("tools.http_client.requests.request", fake_request)
    result = client.post_json("http://example.com")
    assert not result.ok
    assert result.error == "payload_too_large"
    assert result.meta and result.meta.get("truncated") is True
