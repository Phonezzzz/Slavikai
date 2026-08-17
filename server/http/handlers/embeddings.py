from __future__ import annotations

from aiohttp import web

from server.embedding_download import EmbeddingDownloadManager
from server.http.common.responses import error_response, json_response
from server.http_api import _load_embeddings_settings


async def handle_embeddings_status(request: web.Request) -> web.Response:
    manager: EmbeddingDownloadManager = request.app["embedding_download_manager"]
    settings = _load_embeddings_settings()
    snapshot = await manager.status(settings.local_model)
    return json_response(
        {
            **snapshot.to_dict(),
            "provider": settings.provider,
            "install_command": "make install-beta",
        }
    )


async def handle_embeddings_download(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict) or payload.get("confirm") is not True:
        return error_response(
            status=400,
            message="Для загрузки embeddings требуется confirm=true.",
            error_type="invalid_request_error",
            code="confirm_required",
        )
    settings = _load_embeddings_settings()
    model_raw = payload.get("model", settings.local_model)
    if not isinstance(model_raw, str) or not model_raw.strip():
        return error_response(
            status=400,
            message="model должен быть непустой строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if model_raw.strip() != settings.local_model:
        return error_response(
            status=409,
            message="Сначала сохраните выбранную local embeddings model в Settings.",
            error_type="configuration_error",
            code="embeddings_model_not_saved",
        )
    manager: EmbeddingDownloadManager = request.app["embedding_download_manager"]
    snapshot, started = await manager.start(model_raw)
    if snapshot.state == "package_missing":
        return error_response(
            status=409,
            message="Local embeddings dependencies are missing. Run: make install-beta",
            error_type="configuration_error",
            code="embeddings_package_missing",
        )
    return json_response({**snapshot.to_dict(), "started": started}, status=202 if started else 200)
