from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image

from shared.models import ToolRequest, ToolResult

logger = logging.getLogger("SlavikAI.ImageAnalyzeTool")


class ImageAnalyzeTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        """
        Анализ изображения (base64 или локальный файл).
        Возвращает размеры и базовые метаданные.
        """
        data_b64 = str(request.args.get("base64") or "").strip()
        path_value = str(request.args.get("path") or "").strip()

        try:
            if data_b64:
                image_bytes = base64.b64decode(data_b64, validate=True)
            elif path_value:
                image_bytes = Path(path_value).read_bytes()
            else:
                return ToolResult.failure("Нужно передать base64 или path для анализа.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Image decode error: %s", exc)
            return ToolResult.failure(f"Ошибка декодирования изображения: {exc}")

        try:
            with Image.open(BytesIO(image_bytes)) as img:
                width, height = img.size
                mode = img.mode
                format_name = img.format
        except Exception as exc:  # noqa: BLE001
            logger.error("Image open error: %s", exc)
            return ToolResult.failure(f"Ошибка чтения изображения: {exc}")

        return ToolResult.success(
            {
                "width": width,
                "height": height,
                "mode": mode,
                "format": format_name,
            },
            meta={"source": "base64" if data_b64 else "file"},
        )
