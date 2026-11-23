"""Minimal client for the Owner Host Bridge (E03-T02).

One request per connection: connect to the AF_UNIX socket, send one
capability line, read one response line. Used by tests today; the
SlavikAI server becomes the real client when ``owner_host`` is adopted
as an execution target (later E03 tickets).
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Final

from host_bridge.protocol import (
    CAPABILITY_CAPABILITIES,
    CAPABILITY_PING,
    CAPABILITY_VERSION,
    MAX_FRAME_BYTES,
    encode_request,
)

_DEFAULT_TIMEOUT_S: Final[float] = 5.0


class BridgeClientError(Exception):
    """Typed bridge failure; ``code`` is the wire error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def call(
    socket_path: Path,
    capability: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, object]:
    """Send one capability request and return its result object."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(socket_path))
            conn.sendall(encode_request(capability))
            reader = conn.makefile("rb")
            line = reader.readline(MAX_FRAME_BYTES + 1)
    except (BrokenPipeError, ConnectionResetError) as exc:
        raise BridgeClientError(
            "connection_closed", "bridge closed the connection without a response"
        ) from exc
    if not line:
        raise BridgeClientError(
            "connection_closed", "bridge closed the connection without a response"
        )
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeClientError("malformed_response", "bridge response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise BridgeClientError("malformed_response", "bridge response must be a JSON object")
    if payload.get("ok") is True:
        result = payload.get("result")
        if not isinstance(result, dict):
            raise BridgeClientError("malformed_response", "bridge result must be a JSON object")
        return result
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        raise BridgeClientError(
            code if isinstance(code, str) else "bridge_error",
            message if isinstance(message, str) else "bridge returned an error",
        )
    raise BridgeClientError("malformed_response", "bridge response carries no result or error")


def ping(socket_path: Path, *, timeout: float = _DEFAULT_TIMEOUT_S) -> dict[str, object]:
    return call(socket_path, CAPABILITY_PING, timeout=timeout)


def version(socket_path: Path, *, timeout: float = _DEFAULT_TIMEOUT_S) -> dict[str, object]:
    return call(socket_path, CAPABILITY_VERSION, timeout=timeout)


def capabilities(socket_path: Path, *, timeout: float = _DEFAULT_TIMEOUT_S) -> list[str]:
    result = call(socket_path, CAPABILITY_CAPABILITIES, timeout=timeout)
    raw = result.get("capabilities")
    if not isinstance(raw, list):
        raise BridgeClientError("malformed_response", "capabilities must be a list of strings")
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise BridgeClientError("malformed_response", "capabilities must be a list of strings")
        names.append(item)
    return names
