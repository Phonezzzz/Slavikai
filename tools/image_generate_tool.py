from __future__ import annotations

import base64
import logging
from io import BytesIO

from PIL import Image

from shared.models import ToolRequest, ToolResult

logger = logging.getLogger("SlavikAI.ImageGenerateTool")


class ImageGenerateTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        """
        Генерация изображения (локальная-заглушка) по цвету и размеру.
        Без сетевых вызовов, возвращает base64 PNG.
        """
        prompt = str(request.args.get("prompt") or "").strip()
        width_raw = request.args.get("width")
        height_raw = request.args.get("height")
        color_raw = request.args.get("color")
        width = int(width_raw) if isinstance(width_raw, (int, float, str)) else 512
        height = int(height_raw) if isinstance(height_raw, (int, float, str)) else 512
        color = str(color_raw) if color_raw is not None else "#888888"

        if width <= 0 or height <= 0 or width > 2048 or height > 2048:
            return ToolResult.failure("Неверные размеры. Диапазон 1..2048.")

        try:
            img = Image.new("RGB", (width, height), color=color)
            buf = BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.error("Image generation error: %s", exc)
            return ToolResult.failure(f"Ошибка генерации изображения: {exc}")

        return ToolResult.success(
            {"base64": b64, "format": "PNG", "prompt": prompt},
            meta={"width": width, "height": height, "color": color},
        )
