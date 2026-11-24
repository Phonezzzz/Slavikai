"""Minimal client for the Owner Host Bridge (E03-T03 schema, E03-T04 FS).

One request per connection: connect to the AF_UNIX socket, send one
versioned request frame (protocol version, client-generated request ID,
capability, optional attribution fields), read one response frame and
validate it against the schema (protocol version, echoed request ID).
Used by tests today; the SlavikAI server becomes the real client when
``owner_host`` is adopted as an execution target (later E03 tickets).

Timeout semantics (E03-T04): the client never retries. A
``request_timeout`` on a mutation means the outcome is unknown — the
daemon guarantees cancel-before-commit (a timed-out mutation never
publishes late), but if the publish syscall already ran when the
response was lost, the mutation may have taken effect. Callers must
verify the outcome (e.g. ``host.fs.stat``) instead of blindly
retrying.
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
    CAPABILITY_PROCESS_STOP,
    CAPABILITY_VERSION,
    PROTOCOL_VERSION,
    FrameTooLargeError,
    capability_deadline,
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
    timeout: float | None = None,
    request_id: str | None = None,
    principal_id: str | None = None,
    run_id: str | None = None,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    """Send one capability request and return its result object.

    ``timeout=None`` uses the per-capability deadline shared with the
    daemon (``host_bridge.protocol.capability_deadline``) so the two
    sides stay aligned.
    """
    effective_timeout = timeout if timeout is not None else capability_deadline(capability)
    rid = request_id if request_id is not None else uuid.uuid4().hex
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(effective_timeout)
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


def ping(socket_path: Path, *, timeout: float | None = None) -> dict[str, object]:
    return call(socket_path, CAPABILITY_PING, timeout=timeout)


def version(socket_path: Path, *, timeout: float | None = None) -> dict[str, object]:
    return call(socket_path, CAPABILITY_VERSION, timeout=timeout)


def capabilities(socket_path: Path, *, timeout: float | None = None) -> list[str]:
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


def roots(socket_path: Path, *, timeout: float | None = None) -> list[dict[str, object]]:
    """Approved filesystem roots as ``{id, mode}`` (host paths never leak)."""
    result = call(socket_path, CAPABILITY_CAPABILITIES, timeout=timeout)
    raw = result.get("roots")
    if not isinstance(raw, list):
        raise BridgeClientError("malformed_response", "roots must be a list")
    entries: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise BridgeClientError("malformed_response", "roots must be a list of objects")
        entries.append(item)
    return entries


def stop(
    socket_path: Path,
    run_id: str,
    *,
    principal_id: str | None = None,
    timeout: float | None = None,
) -> dict[str, object]:
    """Kill every non-final process bound to one run (E03-T06).

    The ownership key is (principal_id, run_id); the principal
    goes in the envelope, the run id in the params — exactly
    where the daemon reads them.
    """
    return call(
        socket_path,
        CAPABILITY_PROCESS_STOP,
        principal_id=principal_id,
        run_id=run_id,
        params={"run_id": run_id},
        timeout=timeout,
    )
