from __future__ import annotations

from aiohttp import web

from server.http.common.responses import json_response


async def handle_models(request: web.Request) -> web.Response:
    del request
    models = [
        {"id": "slavik", "object": "model", "owned_by": "slavik"},
    ]
    return json_response({"object": "list", "data": models})
