from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import requests

from shared.models import JSONValue

logger = logging.getLogger("SlavikAI.HTTPClient")


@dataclass
class HttpConfig:
    timeout: int = 15
    max_bytes: int = 500_000
    max_json_bytes: int = 1_000_000


@dataclass(frozen=True)
class HttpResult:
    ok: bool
    data: JSONValue | None
    status_code: int | None
    error: str | None = None
    headers: dict[str, str] | None = None
    meta: dict[str, JSONValue] | None = None


class HttpClient:
    def __init__(self, config: HttpConfig | None = None) -> None:
        self.config = config or HttpConfig()

    def get_text(self, url: str, **kwargs: Any) -> HttpResult:
        return self._request("GET", url, expect_json=False, as_bytes=False, **kwargs)

    def post_json(self, url: str, **kwargs: Any) -> HttpResult:
        return self._request("POST", url, expect_json=True, as_bytes=False, **kwargs)

    def post_bytes(self, url: str, **kwargs: Any) -> HttpResult:
        return self._request("POST", url, expect_json=False, as_bytes=True, **kwargs)

    def _request(
        self, method: str, url: str, expect_json: bool, as_bytes: bool, **kwargs: Any
    ) -> HttpResult:
        timeout = kwargs.pop("timeout", self.config.timeout)
        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=timeout,
                stream=True,
                **kwargs,
            )
            status = response.status_code
            response.raise_for_status()
        except requests.Timeout:
            logger.error("HTTP %s timeout for %s", method, url)
            return HttpResult(ok=False, data=None, status_code=None, error="timeout")
        except requests.RequestException as exc:
            logger.error("HTTP %s error for %s: %s", method, url, exc)
            return HttpResult(ok=False, data=None, status_code=None, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.error("HTTP %s unexpected error for %s: %s", method, url, exc)
            return HttpResult(ok=False, data=None, status_code=None, error=str(exc))

        body_raw, truncated = self._read_limited(response, as_bytes=as_bytes)
        meta: dict[str, JSONValue] = {"truncated": truncated, "status_code": status}

        if expect_json:
            if as_bytes and isinstance(body_raw, bytes):
                body_str = body_raw.decode("utf-8", errors="ignore")
            else:
                body_str = cast(str, body_raw)
            if truncated:
                logger.error("HTTP %s JSON body truncated for %s (payload too large)", method, url)
                return HttpResult(
                    ok=False,
                    data=None,
                    status_code=status,
                    error="payload_too_large",
                    headers=dict(response.headers),
                    meta=meta,
                )
            if len(body_str.encode("utf-8")) > self.config.max_json_bytes:
                logger.error("HTTP %s JSON exceeds max_json_bytes for %s", method, url)
                return HttpResult(
                    ok=False,
                    data=None,
                    status_code=status,
                    error="payload_too_large",
                    headers=dict(response.headers),
                    meta=meta,
                )
            try:
                import json

                parsed = json.loads(body_str)
            except Exception as exc:  # noqa: BLE001
                logger.error("HTTP %s JSON decode error for %s: %s", method, url, exc)
                return HttpResult(
                    ok=False,
                    data=None,
                    status_code=status,
                    error=f"json_decode_error: {exc}",
                    headers=dict(response.headers),
                    meta=meta,
                )
            return HttpResult(
                ok=True,
                data=parsed,
                status_code=status,
                headers=dict(response.headers),
                meta=meta,
            )

        return HttpResult(
            ok=True,
            data=cast(JSONValue | None, body_raw),
            status_code=status,
            headers=dict(response.headers),
            meta=meta,
        )

    def _read_limited(
        self, response: requests.Response, as_bytes: bool
    ) -> tuple[str, bool] | tuple[bytes, bool]:
        total = 0
        truncated = False
        if as_bytes:
            collected_bytes: list[bytes] = []
            for chunk in response.iter_content(chunk_size=4096, decode_unicode=False):
                if chunk is None:
                    continue
                total += len(chunk)
                if total > self.config.max_bytes:
                    collected_bytes.append(b"\n...[response truncated]")
                    truncated = True
                    break
                collected_bytes.append(chunk)
            return b"".join(collected_bytes), truncated

        collected_str: list[str] = []
        for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
            if chunk is None:
                continue
            encoded = str(chunk).encode("utf-8")
            total += len(encoded)
            if total > self.config.max_bytes:
                collected_str.append("\n...[response truncated]")
                truncated = True
                break
            collected_str.append(str(chunk))
        return "".join(collected_str), truncated
