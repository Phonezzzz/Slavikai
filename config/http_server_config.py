from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_REQUEST_BYTES = 1_000_000
DEFAULT_PATH = Path("config/http_server.json")


@dataclass(frozen=True)
class HttpServerConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES

    def to_dict(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "max_request_bytes": self.max_request_bytes,
        }


def load_http_server_config(path: Path = DEFAULT_PATH) -> HttpServerConfig:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return HttpServerConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ошибка чтения http_server.json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("http_server.json должен содержать объект.")
    host = data.get("host", DEFAULT_HOST)
    port = data.get("port", DEFAULT_PORT)
    max_request_bytes = data.get("max_request_bytes", DEFAULT_MAX_REQUEST_BYTES)
    if not isinstance(host, str) or not host.strip():
        raise ValueError("http_server.host должен быть непустой строкой.")
    if not isinstance(port, int):
        raise ValueError("http_server.port должен быть int.")
    if not isinstance(max_request_bytes, int):
        raise ValueError("http_server.max_request_bytes должен быть int.")
    return HttpServerConfig(
        host=host.strip(),
        port=port,
        max_request_bytes=max_request_bytes,
    )


def resolve_http_server_config(path: Path = DEFAULT_PATH) -> HttpServerConfig:
    config = load_http_server_config(path)
    host_raw = os.getenv("SLAVIK_HTTP_HOST")
    port_raw = os.getenv("SLAVIK_HTTP_PORT")
    max_bytes_raw = os.getenv("SLAVIK_HTTP_MAX_REQUEST_BYTES")

    host = config.host
    if isinstance(host_raw, str) and host_raw.strip():
        host = host_raw.strip()

    port = config.port
    if isinstance(port_raw, str) and port_raw.strip():
        try:
            port = int(port_raw.strip())
        except ValueError as exc:
            raise ValueError("SLAVIK_HTTP_PORT должен быть int.") from exc

    max_bytes = config.max_request_bytes
    if isinstance(max_bytes_raw, str) and max_bytes_raw.strip():
        try:
            max_bytes = int(max_bytes_raw.strip())
        except ValueError as exc:
            raise ValueError("SLAVIK_HTTP_MAX_REQUEST_BYTES должен быть int.") from exc

    return HttpServerConfig(host=host, port=port, max_request_bytes=max_bytes)
