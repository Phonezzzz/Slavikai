from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shared.identity import normalize_email

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_REQUEST_BYTES = 1_000_000
DEFAULT_PATH = Path("config/http_server.json")
DEFAULT_ALLOW_UNAUTH_LOCAL = False
BrowserAuthMode = Literal["token", "cloudflare"]


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


@dataclass(frozen=True)
class HttpAuthConfig:
    api_token: str
    allow_unauth_local: bool = DEFAULT_ALLOW_UNAUTH_LOCAL
    browser_auth_mode: BrowserAuthMode = "token"
    cloudflare_access_issuer: str = ""
    cloudflare_access_aud: str = ""
    owner_email: str = ""


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


def resolve_http_auth_config() -> HttpAuthConfig:
    api_token_raw = os.getenv("SLAVIK_API_TOKEN", "")
    allow_unauth_raw = os.getenv("SLAVIK_ALLOW_UNAUTH_LOCAL", "")
    allow_unauth_local = allow_unauth_raw.strip().lower() == "true"
    browser_auth_raw = os.getenv("SLAVIK_BROWSER_AUTH", "token").strip().lower()
    browser_auth_mode: BrowserAuthMode
    if browser_auth_raw == "token":
        browser_auth_mode = "token"
    elif browser_auth_raw == "cloudflare":
        browser_auth_mode = "cloudflare"
    else:
        raise ValueError("SLAVIK_BROWSER_AUTH должен быть token|cloudflare.")
    team_domain = os.getenv("SLAVIK_CLOUDFLARE_ACCESS_TEAM_DOMAIN", "").strip()
    issuer = _cloudflare_issuer(team_domain)
    return HttpAuthConfig(
        api_token=api_token_raw.strip(),
        allow_unauth_local=allow_unauth_local,
        browser_auth_mode=browser_auth_mode,
        cloudflare_access_issuer=issuer,
        cloudflare_access_aud=os.getenv("SLAVIK_CLOUDFLARE_ACCESS_AUD", "").strip(),
        owner_email=os.getenv("SLAVIK_OWNER_EMAIL", "").strip().casefold(),
    )


def ensure_http_auth_boot_config(auth_config: HttpAuthConfig | None = None) -> HttpAuthConfig:
    resolved = auth_config or resolve_http_auth_config()
    if not resolved.allow_unauth_local and not resolved.api_token:
        raise RuntimeError("SLAVIK_API_TOKEN is required unless SLAVIK_ALLOW_UNAUTH_LOCAL=true.")
    if resolved.browser_auth_mode == "cloudflare":
        if resolved.allow_unauth_local:
            raise RuntimeError("SLAVIK_ALLOW_UNAUTH_LOCAL must be false with Cloudflare auth.")
        if not resolved.cloudflare_access_issuer:
            raise RuntimeError("SLAVIK_CLOUDFLARE_ACCESS_TEAM_DOMAIN is required.")
        if not resolved.cloudflare_access_aud:
            raise RuntimeError("SLAVIK_CLOUDFLARE_ACCESS_AUD is required.")
        if normalize_email(resolved.owner_email) is None:
            raise RuntimeError("SLAVIK_OWNER_EMAIL is required and must be a valid email.")
    return resolved


def _cloudflare_issuer(team_domain: str) -> str:
    normalized = team_domain.strip().rstrip("/")
    if not normalized:
        return ""
    if normalized.startswith("https://"):
        return normalized
    if "://" in normalized:
        raise ValueError("Cloudflare Access team domain должен использовать https.")
    return f"https://{normalized}"
