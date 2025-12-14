from __future__ import annotations

from tools.http_client import HttpClient, HttpConfig


class DummyResponse:
    def __init__(self, body: bytes, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self) -> None:
        return

    def iter_content(self, chunk_size: int = 4096, decode_unicode: bool = False):
        yield self.body


def test_http_client_reads_bytes(monkeypatch) -> None:
    client = HttpClient(HttpConfig(max_bytes=20))

    def fake_request(**kwargs):
        return DummyResponse(b"0123456789", status_code=200)

    monkeypatch.setattr(
        "tools.http_client.requests.request", lambda **kwargs: fake_request(**kwargs)
    )
    result = client.post_bytes("http://example.com")
    assert result.ok
    assert isinstance(result.data, (bytes, bytearray))
