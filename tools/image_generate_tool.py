from __future__ import annotations

import logging
import os
from typing import Final

from shared.models import ToolRequest, ToolResult
from tools.http_client import HttpClient

logger = logging.getLogger("SlavikAI.ImageGenerateTool")

XAI_IMAGE_ENDPOINT: Final[str] = "https://api.x.ai/v1/images/generations"
DEFAULT_XAI_IMAGE_MODEL: Final[str] = "grok-imagine-image"
DEFAULT_SIZE: Final[int] = 1024
MIN_SIZE: Final[int] = 1
MAX_SIZE: Final[int] = 2048
MAX_IMAGES_PER_REQUEST: Final[int] = 10


class ImageGenerateTool:
    def __init__(
        self,
        http_client: HttpClient | None = None,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.http_client = http_client or HttpClient()
        self._api_key_override = api_key
        self._endpoint_override = endpoint

    def handle(self, request: ToolRequest) -> ToolResult:
        """
        Генерация изображения через xAI Images API.
        Возвращает base64 (b64_json) или URL (если явно запрошено).
        """
        prompt = str(request.args.get("prompt") or "").strip()
        if not prompt:
            return ToolResult.failure("Нужен prompt для генерации изображения.")

        try:
            width, height = _resolve_size(request)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        try:
            n = _parse_int_like(request.args.get("n"), 1, "n")
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        if n < 1 or n > MAX_IMAGES_PER_REQUEST:
            return ToolResult.failure("n должен быть в диапазоне 1..10.")

        api_key_raw = self._api_key_override
        if api_key_raw is None:
            api_key_raw = os.getenv("XAI_API_KEY", "")
        api_key = str(api_key_raw).strip()
        if not api_key:
            return ToolResult.failure("Не задан XAI_API_KEY для image_generate.")

        endpoint = (
            self._endpoint_override
            or os.getenv("XAI_IMAGE_API_URL", "").strip()
            or XAI_IMAGE_ENDPOINT
        )
        model = str(request.args.get("model") or os.getenv("XAI_IMAGE_MODEL") or "").strip()
        if not model:
            model = DEFAULT_XAI_IMAGE_MODEL

        response_format = str(request.args.get("response_format") or "b64_json").strip()
        payload: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "n": n,
            "response_format": response_format,
        }

        aspect_ratio = str(request.args.get("aspect_ratio") or "").strip()
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio

        result = self.http_client.post_json(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if not result.ok:
            error = result.error or "unknown_error"
            return ToolResult.failure(f"xAI image generation failed: {error}")

        data = result.data
        if not isinstance(data, dict):
            return ToolResult.failure("Некорректный ответ xAI: ожидался JSON объект.")
        items = data.get("data")
        if not isinstance(items, list) or not items:
            return ToolResult.failure("Некорректный ответ xAI: отсутствует data[].")
        first_item = items[0]
        if not isinstance(first_item, dict):
            return ToolResult.failure("Некорректный ответ xAI: data[0] не объект.")

        b64_json = first_item.get("b64_json")
        if isinstance(b64_json, str) and b64_json.strip():
            return ToolResult.success(
                {
                    "base64": b64_json,
                    "format": "PNG",
                    "prompt": prompt,
                    "model": model,
                },
                meta={
                    "provider": "xai",
                    "width": width,
                    "height": height,
                    "n": n,
                    "response_format": "b64_json",
                },
            )

        image_url = first_item.get("url")
        if isinstance(image_url, str) and image_url.strip():
            return ToolResult.success(
                {
                    "output": image_url,
                    "url": image_url,
                    "prompt": prompt,
                    "model": model,
                },
                meta={
                    "provider": "xai",
                    "width": width,
                    "height": height,
                    "n": n,
                    "response_format": "url",
                },
            )

        logger.error("Image generation response has no b64_json/url: %s", data)
        return ToolResult.failure("xAI вернул ответ без b64_json и без url.")


def _parse_int_like(raw: object, default: int, name: str) -> int:
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        raise ValueError(f"{name} должен быть целым числом.")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not raw.is_integer():
            raise ValueError(f"{name} должен быть целым числом.")
        return int(raw)
    if isinstance(raw, str):
        value_raw = raw.strip()
        if not value_raw:
            return default
        try:
            return int(value_raw)
        except ValueError as exc:
            raise ValueError(f"{name} должен быть целым числом.") from exc
    raise ValueError(f"{name} должен быть целым числом.")


def _parse_dimension(raw: object, default: int, name: str) -> int:
    value = _parse_int_like(raw, default, name)
    if value < MIN_SIZE or value > MAX_SIZE:
        raise ValueError(f"{name} должен быть в диапазоне {MIN_SIZE}..{MAX_SIZE}.")
    return value


def _resolve_size(request: ToolRequest) -> tuple[int, int]:
    size_raw = str(request.args.get("size") or "").strip().lower()
    if size_raw:
        if "x" not in size_raw:
            raise ValueError("size должен быть в формате <width>x<height>.")
        width_str, height_str = size_raw.split("x", maxsplit=1)
        width = _parse_dimension(width_str, DEFAULT_SIZE, "width")
        height = _parse_dimension(height_str, DEFAULT_SIZE, "height")
        return width, height

    width = _parse_dimension(request.args.get("width"), DEFAULT_SIZE, "width")
    height = _parse_dimension(request.args.get("height"), DEFAULT_SIZE, "height")
    return width, height
