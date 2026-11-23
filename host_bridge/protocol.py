"""Wire format for the Owner Host Bridge (E03-T02).

Minimal newline-delimited JSON: one request object per line, one response
object per line. A request names exactly one capability::

    {"capability": "bridge.ping"}

Responses are ``{"ok": true, "result": {...}}`` on success and
``{"ok": false, "error": {"code": ..., "message": ...}}`` on failure.

This is deliberately the smallest format that proves lifecycle/auth
(E03-T02). The versioned request schema, formal frame limit, request ID,
and timeouts are E03-T03; the only bound here is a hard per-line byte cap
so a peer cannot make the daemon buffer unbounded input. Unknown request
fields are ignored and never become command arguments.
"""

from __future__ import annotations

import json
from typing import Final

PROTOCOL_VERSION: Final[int] = 1
BRIDGE_VERSION: Final[str] = "0.1.0"

CAPABILITY_PING: Final[str] = "bridge.ping"
CAPABILITY_VERSION: Final[str] = "bridge.version"
CAPABILITY_CAPABILITIES: Final[str] = "bridge.capabilities"
CAPABILITIES: Final[tuple[str, ...]] = (
    CAPABILITY_PING,
    CAPABILITY_VERSION,
    CAPABILITY_CAPABILITIES,
)

# Hard safety bound on one NDJSON line. E03-T03 formalizes the frame limit.
MAX_FRAME_BYTES: Final[int] = 64 * 1024


class BridgeProtocolError(Exception):
    """Typed protocol failure; ``code`` is the wire error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MalformedRequestError(BridgeProtocolError):
    def __init__(self, message: str) -> None:
        super().__init__("malformed_request", message)


class UnknownCapabilityError(BridgeProtocolError):
    def __init__(self, capability: str) -> None:
        super().__init__("unknown_capability", f"unknown capability: {capability!r}")


def encode_request(capability: str) -> bytes:
    return json.dumps({"capability": capability}).encode("utf-8") + b"\n"


def decode_request(line: bytes) -> str:
    """Return the requested capability or raise MalformedRequestError."""
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedRequestError("request is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise MalformedRequestError("request must be a JSON object")
    capability = payload.get("capability")
    if not isinstance(capability, str) or not capability:
        raise MalformedRequestError("request must carry a non-empty string 'capability'")
    return capability


def encode_success(result: dict[str, object]) -> bytes:
    return json.dumps({"ok": True, "result": result}).encode("utf-8") + b"\n"


def encode_error(error: BridgeProtocolError) -> bytes:
    payload = {"ok": False, "error": {"code": error.code, "message": str(error)}}
    return json.dumps(payload).encode("utf-8") + b"\n"
