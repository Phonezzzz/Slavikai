from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from shared.models import ToolRequest
from tools.http_client import HttpResult
from tools.image_analyze_tool import ImageAnalyzeTool
from tools.image_generate_tool import ImageGenerateTool


class DummyHttpClient:
    def __init__(self, result: HttpResult) -> None:
        self.result = result
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None
        self.last_json: dict[str, object] | None = None

    def post_json(self, url: str, **kwargs: object) -> HttpResult:
        self.last_url = url
        headers_raw = kwargs.get("headers")
        json_raw = kwargs.get("json")
        if isinstance(headers_raw, dict):
            self.last_headers = {str(k): str(v) for k, v in headers_raw.items()}
        if isinstance(json_raw, dict):
            self.last_json = {str(k): v for k, v in json_raw.items()}
        return self.result


def test_image_generate_returns_base64(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    stub_http = DummyHttpClient(
        HttpResult(
            ok=True,
            data={"data": [{"b64_json": "ZmFrZV9pbWFnZQ=="}]},
            status_code=200,
            error=None,
            headers={},
            meta={},
        )
    )
    tool = ImageGenerateTool(http_client=stub_http)
    req = ToolRequest(
        name="img_gen",
        args={"prompt": "square", "width": 64, "height": 64, "model": "grok-imagine-image"},
    )
    result = tool.handle(req)
    assert result.ok
    b64 = result.data.get("base64")
    assert isinstance(b64, str) and len(b64) > 10
    meta = result.meta or {}
    assert meta.get("width") == 64
    assert meta.get("height") == 64
    assert meta.get("provider") == "xai"
    assert stub_http.last_url is not None
    assert stub_http.last_url.endswith("/images/generations")
    assert stub_http.last_headers is not None
    assert stub_http.last_headers.get("Authorization") == "Bearer test-key"
    assert stub_http.last_json is not None
    assert stub_http.last_json.get("prompt") == "square"
    assert stub_http.last_json.get("size") == "64x64"
    assert stub_http.last_json.get("model") == "grok-imagine-image"


def test_image_generate_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    stub_http = DummyHttpClient(
        HttpResult(
            ok=True,
            data={"data": [{"b64_json": "ZmFrZQ=="}]},
            status_code=200,
            error=None,
            headers={},
            meta={},
        )
    )
    result = ImageGenerateTool(http_client=stub_http).handle(
        ToolRequest(name="img_gen", args={"prompt": "no-key"})
    )
    assert not result.ok
    assert "xai_api_key" in (result.error or "").lower()


def test_image_analyze_handles_invalid_input() -> None:
    req = ToolRequest(name="img_analyze", args={"base64": "not_base64"})
    result = ImageAnalyzeTool().handle(req)
    assert not result.ok


def test_image_analyze_valid_base64(monkeypatch) -> None:
    # generate small valid image
    img = Image.new("RGB", (8, 8), color="#ff00ff")
    buf = BytesIO()
    img.save(buf, format="PNG")
    data_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    req = ToolRequest(name="img_analyze", args={"base64": data_b64})
    result = ImageAnalyzeTool().handle(req)
    assert result.ok
    assert result.data.get("width") == 8
    assert result.data.get("height") == 8


def test_image_analyze_rejects_outside_sandbox(tmp_path: Path) -> None:
    sandbox_root = (tmp_path / "sandbox").resolve()
    sandbox_root.mkdir(parents=True)
    evil_dir = tmp_path / "outside"
    evil_dir.mkdir(parents=True)
    img = Image.new("RGB", (4, 4), color="#ff0000")
    evil_file = evil_dir / "evil.png"
    img.save(str(evil_file), format="PNG")

    tool = ImageAnalyzeTool(sandbox_root=sandbox_root)
    result = tool.handle(ToolRequest(name="img_analyze", args={"path": "../outside/evil.png"}))
    assert not result.ok
    assert "sandbox" in (result.error or "").lower()


def test_image_analyze_rejects_absolute_path(tmp_path: Path) -> None:
    sandbox_root = (tmp_path / "sandbox").resolve()
    sandbox_root.mkdir(parents=True)

    tool = ImageAnalyzeTool(sandbox_root=sandbox_root)
    result = tool.handle(ToolRequest(name="img_analyze", args={"path": "/etc/passwd"}))
    assert not result.ok
    assert "sandbox" in (result.error or "").lower()


def test_image_analyze_allows_sandbox_png(tmp_path: Path) -> None:
    sandbox_root = (tmp_path / "sandbox").resolve()
    sandbox_root.mkdir(parents=True)
    img = Image.new("RGB", (4, 4), color="#00ff00")
    png_path = sandbox_root / "ok.png"
    img.save(str(png_path), format="PNG")

    tool = ImageAnalyzeTool(sandbox_root=sandbox_root)
    result = tool.handle(ToolRequest(name="img_analyze", args={"path": "ok.png"}))
    assert result.ok, result.error
    assert result.data.get("width") == 4
    assert result.data.get("height") == 4
