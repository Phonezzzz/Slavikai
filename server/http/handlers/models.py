from __future__ import annotations

from aiohttp import web

from server.http.common.proxy_model import PUBLIC_PROXY_MODEL_ID
from server.http.common.responses import json_response


async def handle_models(request: web.Request) -> web.Response:
    del request
    models = [
        {
            "id": PUBLIC_PROXY_MODEL_ID,
            "object": "model",
            "owned_by": PUBLIC_PROXY_MODEL_ID,
        },
    ]
    return json_response({"object": "list", "data": models})
