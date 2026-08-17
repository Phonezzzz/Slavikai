from __future__ import annotations

from pathlib import Path

from aiohttp import web

from server.http.common.responses import json_response
from server.version import VERSION, build_sha


async def handle_health(request: web.Request) -> web.Response:
    dist_path: Path = request.app["ui_dist_path"]
    return json_response(
        {
            "status": "ok",
            "version": VERSION,
            "build_sha": build_sha(),
            "ui_built": (dist_path / "index.html").is_file(),
        }
    )
