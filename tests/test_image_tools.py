from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from shared.models import ToolRequest
from tools.image_analyze_tool import ImageAnalyzeTool
from tools.image_generate_tool import ImageGenerateTool


def test_image_generate_returns_base64() -> None:
    req = ToolRequest(
        name="img_gen",
        args={"prompt": "square", "width": 64, "height": 64, "color": "#112233"},
    )
    result = ImageGenerateTool().handle(req)
    assert result.ok
    b64 = result.data.get("base64")
    assert isinstance(b64, str) and len(b64) > 10
    meta = result.meta or {}
    assert meta.get("width") == 64
    assert meta.get("height") == 64


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
