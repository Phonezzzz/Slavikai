#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _get_json(base_url: str, path: str, token: str | None = None) -> dict[str, object]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{base_url}{path}", headers=headers)  # noqa: S310
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} returned a non-object JSON payload")
    return payload


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(override=False)
    base_url = os.getenv("SLAVIK_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    token = os.getenv("SLAVIK_API_TOKEN", "").strip()
    if not token:
        print("ERROR: set SLAVIK_API_TOKEN before running smoke-prod", file=sys.stderr)
        return 2
    try:
        health = _get_json(base_url, "/healthz")
        settings = _get_json(base_url, "/ui/api/settings", token)
        models = _get_json(base_url, "/v1/models", token)
    except (OSError, ValueError, RuntimeError, urllib.error.HTTPError) as exc:
        print(f"ERROR: beta smoke check failed: {exc}", file=sys.stderr)
        return 1
    if health.get("status") != "ok" or "settings" not in settings or "data" not in models:
        print("ERROR: beta smoke check received an incomplete response", file=sys.stderr)
        return 1
    print(
        f"OK: SlavikAI {health.get('version', 'unknown')} is healthy at {base_url}; "
        "settings and models endpoints are authenticated and ready."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
