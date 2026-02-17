from __future__ import annotations

from aiohttp import web
from aiohttp.multipart import BodyPartReader

from server import http_api as api
from server.http.common.responses import error_response, json_response


def _openai_error_message(response: object) -> str | None:
    if not hasattr(response, "json"):
        return None
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    error_raw = payload.get("error")
    if not isinstance(error_raw, dict):
        return None
    message_raw = error_raw.get("message")
    if not isinstance(message_raw, str):
        return None
    normalized = message_raw.strip()
    return normalized or None


async def handle_ui_stt_transcribe(request: web.Request) -> web.Response:
    api_key = api._resolve_provider_api_key("openai")
    if not api_key:
        return error_response(
            status=409,
            message="Не задан OpenAI API key для STT (settings.providers.openai.api_key).",
            error_type="configuration_error",
            code="stt_api_key_missing",
        )

    try:
        reader = await request.multipart()
    except Exception:  # noqa: BLE001
        return error_response(
            status=400,
            message="Ожидался multipart/form-data с полем audio.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    audio_bytes: bytes | None = None
    audio_filename = "recording.webm"
    audio_content_type = "application/octet-stream"
    language = "ru"
    while True:
        part = await reader.next()
        if part is None:
            break
        if not isinstance(part, BodyPartReader):
            continue
        name = str(getattr(part, "name", "") or "").strip()
        if not name:
            continue
        if name == "language":
            try:
                language_raw = await part.text()
            except Exception:  # noqa: BLE001
                language_raw = ""
            normalized_language = language_raw.strip()
            if normalized_language:
                language = normalized_language
            continue
        if name != "audio":
            continue
        audio_filename_raw = getattr(part, "filename", None)
        if isinstance(audio_filename_raw, str) and audio_filename_raw.strip():
            audio_filename = audio_filename_raw.strip()
        part_content_type = part.headers.get("Content-Type")
        if isinstance(part_content_type, str) and part_content_type.strip():
            audio_content_type = part_content_type.strip()
        chunks: list[bytes] = []
        total_size = 0
        while True:
            chunk = await part.read_chunk()
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > api.MAX_STT_AUDIO_BYTES:
                return error_response(
                    status=413,
                    message="Аудиофайл слишком большой.",
                    error_type="invalid_request_error",
                    code="payload_too_large",
                )
            chunks.append(chunk)
        if chunks:
            audio_bytes = b"".join(chunks)

    if audio_bytes is None:
        return error_response(
            status=400,
            message="Поле audio обязательно.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    try:
        requests_module = api._requests_module()
        response = requests_module.post(
            api.OPENAI_STT_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": "whisper-1",
                "language": language,
                "response_format": "json",
            },
            files={"file": (audio_filename, audio_bytes, audio_content_type)},
            timeout=api.MODEL_FETCH_TIMEOUT,
        )
    except Exception:  # noqa: BLE001
        return error_response(
            status=502,
            message="Не удалось связаться с STT-провайдером.",
            error_type="upstream_error",
            code="upstream_error",
        )

    if response.status_code >= 400:
        upstream_message = _openai_error_message(response)
        if response.status_code in {400, 415, 422}:
            return error_response(
                status=400,
                message=upstream_message or "Неподдерживаемый формат аудио.",
                error_type="invalid_request_error",
                code="unsupported_audio_format",
            )
        return error_response(
            status=502,
            message=upstream_message or "STT-провайдер вернул ошибку.",
            error_type="upstream_error",
            code="upstream_error",
        )

    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = None
    if not isinstance(payload, dict):
        return error_response(
            status=502,
            message="STT-провайдер вернул неожиданный ответ.",
            error_type="upstream_error",
            code="upstream_error",
        )
    text_raw = payload.get("text")
    if not isinstance(text_raw, str) or not text_raw.strip():
        return error_response(
            status=502,
            message="STT-провайдер не вернул текст распознавания.",
            error_type="upstream_error",
            code="upstream_error",
        )
    return json_response(
        {
            "text": text_raw.strip(),
            "model": "whisper-1",
            "language": language,
        }
    )
