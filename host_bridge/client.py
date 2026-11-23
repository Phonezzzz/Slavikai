"""Minimal client for the Owner Host Bridge (E03-T03 schema).

One request per connection: connect to the AF_UNIX socket, send one
versioned request frame (protocol version, client-generated request ID,
capability, optional attribution fields), read one response frame and
validate it against the schema (protocol version, echoed request ID).
Used by tests today; the SlavikAI server becomes the real client when
``owner_host`` is adopted as an execution target (later E03 tickets).
"""

from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path
from typing import Final

from host_bridge.protocol import (
    CAPABILITY_CAPABILITIES,
    CAPABILITY_PING,
    CAPABILITY_VERSION,
    PROTOCOL_VERSION,
    FrameTooLargeError,
    encode_request,
    read_frame,
)

_DEFAULT_TIMEOUT_S: Final[float] = 5.0

# Typed rejections the daemon emits before a request ID could be
# extracted -- the frame was rejected before parsing
# (``frame_too_large``), the frame was not a JSON object
# (``malformed_request``), or the request ID itself was invalid
# (``invalid_request_id``) -- carry ``request_id: null`` by design.
# The client surfaces their typed code instead of masking them as a
# correlation failure. Successful responses and errors echoing an ID
# keep strict correlation.
_PRE_ID_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {"frame_too_large", "invalid_request_id", "malformed_request"}
)


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
    request_id: str | None = None,
    principal_id: str | None = None,
    run_id: str | None = None,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Send one capability request and return its result object."""
    rid = request_id if request_id is not None else uuid.uuid4().hex
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(timeout)
            conn.connect(str(socket_path))
            conn.sendall(
                encode_request(
                    capability,
                    request_id=rid,
                    principal_id=principal_id,
                    run_id=run_id,
                    params=params,
                )
            )
            reader = conn.makefile("rb")
            frame = read_frame(reader)
    except TimeoutError as exc:
        raise BridgeClientError(
            "request_timeout", "bridge did not answer within the timeout"
        ) from exc
    except FrameTooLargeError as exc:
        raise BridgeClientError(
            "malformed_response", "bridge response exceeds the frame limit"
        ) from exc
    except (BrokenPipeError, ConnectionResetError) as exc:
        raise BridgeClientError(
            "connection_closed", "bridge closed the connection without a response"
        ) from exc
    if frame is None:
        raise BridgeClientError(
            "connection_closed", "bridge closed the connection without a response"
        )
    try:
        payload = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeClientError("malformed_response", "bridge response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise BridgeClientError("malformed_response", "bridge response must be a JSON object")
    version = payload.get("version")
    if isinstance(version, bool) or version != PROTOCOL_VERSION:
        raise BridgeClientError(
            "malformed_response", f"bridge response carries unsupported version: {version!r}"
        )
    response_rid = payload.get("request_id")
    if response_rid != rid:
        error = payload.get("error")
        if response_rid is None and payload.get("ok") is False and isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, str) and code in _PRE_ID_ERROR_CODES:
                # The daemon rejected the frame before an ID could be
                # extracted (or the ID itself was invalid), so there is
                # nothing to correlate; surface the typed error instead
                # of masking it as a correlation failure.
                message = error.get("message")
                raise BridgeClientError(
                    code,
                    message if isinstance(message, str) else "bridge rejected the request",
                )
        raise BridgeClientError(
            "malformed_response", "bridge response request_id does not match the request"
        )
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
