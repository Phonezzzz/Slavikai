"""Wire format for the Owner Host Bridge (E03-T03).

Newline-delimited JSON: one request object per line, one response object
per line. Since E03-T03 every frame follows the versioned schema below
(ADR-0013 §6: unknown capability or unknown protocol version is a
deterministic rejection; unknown request fields never become command
arguments).

Request (protocol version 1)::

    {
      "version": 1,                    // required, int, must equal 1
      "request_id": "client-chosen",   // required, 1..128 chars, echoed
      "capability": "bridge.ping",     // required, non-empty string
      "principal_id": "owner",         // optional, attribution only
      "run_id": "run-42",              // optional, attribution only
      "params": {}                     // optional object, capability args
    }

``principal_id`` and ``run_id`` are attribution/audit data only
(ADR-0013 §4): the bridge never authorizes from request JSON. Run
authorization is confirmed by the trusted runtime out-of-band; T03
defines the wire fields, not the auth mechanism.

Unknown top-level fields are rejected deterministically
(``unknown_field``) rather than ignored, so a typo'd or smuggled field
can never silently change meaning. Validation order is fixed: JSON
shape, protocol version, unknown fields, request ID, capability, then
optional-field types.

Response::

    {"version": 1, "request_id": "...", "ok": true,  "result": {...}}
    {"version": 1, "request_id": "...", "ok": false,
     "error": {"code": ..., "message": ...}}

``request_id`` echoes the request's ID whenever one could be extracted
from the frame; otherwise it is ``null``.

Frame limit: one frame (the JSON object bytes, excluding the newline
delimiter) must not exceed ``MAX_FRAME_BYTES``; larger frames are
rejected with ``frame_too_large`` and the connection is closed (framing
is lost). Per-request deadline enforcement lives in the daemon
(``request_timeout``). Error responses obey the same limit:
request-derived strings echoed in error messages are truncated, and
``encode_error`` never emits a frame larger than ``MAX_FRAME_BYTES``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import BinaryIO, Final

PROTOCOL_VERSION: Final[int] = 1
BRIDGE_VERSION: Final[str] = "0.1.0"

CAPABILITY_PING: Final[str] = "bridge.ping"
CAPABILITY_VERSION: Final[str] = "bridge.version"
CAPABILITY_CAPABILITIES: Final[str] = "bridge.capabilities"

# Bounded host filesystem API (E03-T04). Wire schema for the params of
# each capability lives in host_bridge.fs; the envelope is unchanged.
CAPABILITY_FS_STAT: Final[str] = "host.fs.stat"
CAPABILITY_FS_LIST: Final[str] = "host.fs.list"
CAPABILITY_FS_READ: Final[str] = "host.fs.read"
CAPABILITY_FS_WRITE: Final[str] = "host.fs.write"
CAPABILITY_FS_PATCH: Final[str] = "host.fs.patch"
CAPABILITY_FS_MOVE: Final[str] = "host.fs.move"
CAPABILITY_FS_DELETE: Final[str] = "host.fs.delete"
CAPABILITY_FS_SEARCH: Final[str] = "host.fs.search"

# Approved-profile process API (E03-T05). Wire schema for the params
# of each capability lives in host_bridge.process; the envelope is
# unchanged. Deadlines stay at the 5 s default (no CAPABILITY_DEADLINES
# entries): run/poll/kill are all instant by design. Run-scoped STOP
# is E03-T06: host.process.stop kills only the processes bound to one
# (principal_id, run_id) pair, under a single shared deadline.
CAPABILITY_PROCESS_RUN: Final[str] = "host.process.run"
CAPABILITY_PROCESS_POLL: Final[str] = "host.process.poll"
CAPABILITY_PROCESS_KILL: Final[str] = "host.process.kill"
CAPABILITY_PROCESS_STOP: Final[str] = "host.process.stop"

# Read-only host diagnostics (E03-T07, ADR-0013 §6). Wire schema for
# the params of each capability lives in host_bridge.diag; the
# envelope is unchanged. No deadlines entries: diag handlers are
# instant by design (systemctl 2 s, disk under a shared 2 s budget),
# so the 5 s default covers them with margin.
CAPABILITY_DIAG_SERVICE_STATUS: Final[str] = "host.diag.service_status"
CAPABILITY_DIAG_PROCESS_LIST: Final[str] = "host.diag.process_list"
CAPABILITY_DIAG_DISK: Final[str] = "host.diag.disk"
CAPABILITY_DIAG_HOST: Final[str] = "host.diag.host"

CAPABILITIES: Final[tuple[str, ...]] = (
    CAPABILITY_PING,
    CAPABILITY_VERSION,
    CAPABILITY_CAPABILITIES,
    CAPABILITY_FS_STAT,
    CAPABILITY_FS_LIST,
    CAPABILITY_FS_READ,
    CAPABILITY_FS_WRITE,
    CAPABILITY_FS_PATCH,
    CAPABILITY_FS_MOVE,
    CAPABILITY_FS_DELETE,
    CAPABILITY_FS_SEARCH,
    CAPABILITY_PROCESS_RUN,
    CAPABILITY_PROCESS_POLL,
    CAPABILITY_PROCESS_KILL,
    CAPABILITY_PROCESS_STOP,
    CAPABILITY_DIAG_SERVICE_STATUS,
    CAPABILITY_DIAG_PROCESS_LIST,
    CAPABILITY_DIAG_DISK,
    CAPABILITY_DIAG_HOST,
)

# Per-capability daemon deadlines (E03-T04): tree walks and moves over
# large trees legitimately exceed the 5 s default. The daemon enforces
# these; the client uses the same map for its default timeout so the
# two sides stay aligned.
DEFAULT_CAPABILITY_DEADLINE_S: Final[float] = 5.0
CAPABILITY_DEADLINES: Final[dict[str, float]] = {
    CAPABILITY_FS_SEARCH: 30.0,
    CAPABILITY_FS_MOVE: 30.0,
    CAPABILITY_FS_DELETE: 30.0,
}


def capability_deadline(capability: str, default: float = DEFAULT_CAPABILITY_DEADLINE_S) -> float:
    """Daemon deadline for a capability (seconds)."""
    return CAPABILITY_DEADLINES.get(capability, default)


FS_CAPABILITIES: Final[frozenset[str]] = frozenset(
    {
        CAPABILITY_FS_STAT,
        CAPABILITY_FS_LIST,
        CAPABILITY_FS_READ,
        CAPABILITY_FS_WRITE,
        CAPABILITY_FS_PATCH,
        CAPABILITY_FS_MOVE,
        CAPABILITY_FS_DELETE,
        CAPABILITY_FS_SEARCH,
    }
)

PROCESS_CAPABILITIES: Final[frozenset[str]] = frozenset(
    {
        CAPABILITY_PROCESS_RUN,
        CAPABILITY_PROCESS_POLL,
        CAPABILITY_PROCESS_KILL,
        CAPABILITY_PROCESS_STOP,
    }
)

# Read-only diagnostics (E03-T07): unlike the process capabilities,
# these are never hidden when the enforcement probe fails — the
# diagnosis must answer exactly when the diagnosed subsystem is
# broken (see daemon._dispatch).
DIAG_CAPABILITIES: Final[frozenset[str]] = frozenset(
    {
        CAPABILITY_DIAG_SERVICE_STATUS,
        CAPABILITY_DIAG_PROCESS_LIST,
        CAPABILITY_DIAG_DISK,
        CAPABILITY_DIAG_HOST,
    }
)

# Formal frame limit (E03-T03): maximum bytes of one JSON frame,
# excluding the newline delimiter.
MAX_FRAME_BYTES: Final[int] = 64 * 1024

MAX_REQUEST_ID_CHARS: Final[int] = 128
MAX_ATTRIBUTION_CHARS: Final[int] = 256

# Error messages may echo request-derived strings (field names,
# capability names). Bound them so an error response can never exceed
# the frame limit no matter what the request contained.
MAX_ERROR_ECHO_CHARS: Final[int] = 64
MAX_ERROR_FIELDS_LISTED: Final[int] = 8

_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {"version", "request_id", "capability", "principal_id", "run_id", "params"}
)


def _bounded_echo(text: str) -> str:
    """Truncate a request-derived string for safe echo in an error."""
    if len(text) <= MAX_ERROR_ECHO_CHARS:
        return text
    return text[: MAX_ERROR_ECHO_CHARS - 1] + "…"


class BridgeConfigError(Exception):
    """Fatal host-bridge configuration problem (daemon refuses to start)."""


class BridgeProtocolError(Exception):
    """Typed protocol failure; ``code`` is the wire error code.

    ``request_id`` carries the request's ID when one could be extracted
    from the frame, so the daemon can echo it in the error response.
    """

    def __init__(self, code: str, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id


class MalformedRequestError(BridgeProtocolError):
    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__("malformed_request", message, request_id=request_id)


class UnsupportedVersionError(BridgeProtocolError):
    def __init__(self, version: int, *, request_id: str | None = None) -> None:
        super().__init__(
            "unsupported_version",
            f"unsupported protocol version: {version} (supported: {PROTOCOL_VERSION})",
            request_id=request_id,
        )


class InvalidRequestIdError(BridgeProtocolError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_request_id", message)


class UnknownFieldError(BridgeProtocolError):
    def __init__(self, fields: list[str], *, request_id: str | None = None) -> None:
        ordered = sorted(fields)
        shown = [_bounded_echo(name) for name in ordered[:MAX_ERROR_FIELDS_LISTED]]
        suffix = ""
        if len(ordered) > MAX_ERROR_FIELDS_LISTED:
            suffix = f" (+{len(ordered) - MAX_ERROR_FIELDS_LISTED} more)"
        super().__init__(
            "unknown_field",
            f"unknown request field(s): {', '.join(shown)}{suffix}",
            request_id=request_id,
        )


class UnknownCapabilityError(BridgeProtocolError):
    def __init__(self, capability: str, *, request_id: str | None = None) -> None:
        super().__init__(
            "unknown_capability",
            f"unknown capability: {_bounded_echo(capability)!r}",
            request_id=request_id,
        )


class FrameTooLargeError(BridgeProtocolError):
    def __init__(self) -> None:
        super().__init__("frame_too_large", "request frame exceeds the frame limit")


class RequestTimeoutError(BridgeProtocolError):
    def __init__(self, *, request_id: str | None = None) -> None:
        super().__init__(
            "request_timeout", "request exceeded the daemon deadline", request_id=request_id
        )


@dataclass(frozen=True)
class BridgeRequest:
    """One validated bridge request (protocol version 1).

    ``principal_id`` / ``run_id`` are attribution data only (ADR-0013
    §4); nothing in the bridge may authorize from them.
    """

    version: int
    request_id: str
    capability: str
    principal_id: str | None = None
    run_id: str | None = None
    params: dict[str, object] = field(default_factory=dict)


def read_frame(reader: BinaryIO) -> bytes | None:
    """Read one NDJSON frame; ``None`` at clean EOF.

    Raises FrameTooLargeError when the frame content (excluding the
    newline delimiter) exceeds MAX_FRAME_BYTES.
    """
    line = reader.readline(MAX_FRAME_BYTES + 2)
    if not line:
        return None
    if line.endswith(b"\n"):
        frame = line[:-1]
        if len(frame) > MAX_FRAME_BYTES:
            raise FrameTooLargeError()
        return frame
    if len(line) > MAX_FRAME_BYTES:
        # No delimiter within the bound: the frame is oversized.
        raise FrameTooLargeError()
    return line  # final frame without trailing newline at EOF


def _valid_request_id(value: object) -> str | None:
    if isinstance(value, str) and 1 <= len(value) <= MAX_REQUEST_ID_CHARS:
        return value
    return None


def _optional_attribution(payload: dict[str, object], name: str, echo_id: str | None) -> str | None:
    if name not in payload:
        return None
    value = payload[name]
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_ATTRIBUTION_CHARS:
        raise MalformedRequestError(
            f"'{name}' must be a string of 1..{MAX_ATTRIBUTION_CHARS} characters",
            request_id=echo_id,
        )
    return value


def decode_request(frame: bytes) -> BridgeRequest:
    """Validate one frame against the versioned schema (E03-T03)."""
    try:
        payload = json.loads(frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedRequestError("request is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise MalformedRequestError("request must be a JSON object")
    echo_id = _valid_request_id(payload.get("request_id"))
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise MalformedRequestError("request must carry an integer 'version'", request_id=echo_id)
    if version != PROTOCOL_VERSION:
        raise UnsupportedVersionError(version, request_id=echo_id)
    unknown = sorted(set(payload) - _REQUEST_FIELDS)
    if unknown:
        raise UnknownFieldError(unknown, request_id=echo_id)
    if echo_id is None:
        raise InvalidRequestIdError(
            f"request must carry a string 'request_id' of 1..{MAX_REQUEST_ID_CHARS} characters"
        )
    capability = payload.get("capability")
    if not isinstance(capability, str) or not capability:
        raise MalformedRequestError(
            "request must carry a non-empty string 'capability'", request_id=echo_id
        )
    principal_id = _optional_attribution(payload, "principal_id", echo_id)
    run_id = _optional_attribution(payload, "run_id", echo_id)
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise MalformedRequestError("'params' must be a JSON object", request_id=echo_id)
    return BridgeRequest(
        version=version,
        request_id=echo_id,
        capability=capability,
        principal_id=principal_id,
        run_id=run_id,
        params=params,
    )


def encode_request(
    capability: str,
    *,
    request_id: str,
    principal_id: str | None = None,
    run_id: str | None = None,
    params: dict[str, object] | None = None,
) -> bytes:
    payload: dict[str, object] = {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "capability": capability,
    }
    if principal_id is not None:
        payload["principal_id"] = principal_id
    if run_id is not None:
        payload["run_id"] = run_id
    if params is not None:
        payload["params"] = params
    return json.dumps(payload).encode("utf-8") + b"\n"


def encode_success(result: dict[str, object], request_id: str) -> bytes:
    payload = {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": True,
        "result": result,
    }
    return json.dumps(payload).encode("utf-8") + b"\n"


def encode_error(error: BridgeProtocolError) -> bytes:
    payload = {
        "version": PROTOCOL_VERSION,
        "request_id": error.request_id,
        "ok": False,
        "error": {"code": error.code, "message": str(error)},
    }
    frame = json.dumps(payload).encode("utf-8")
    if len(frame) > MAX_FRAME_BYTES:
        # Backstop: no error response may exceed the frame limit,
        # whatever the request contained. The echoed request_id is
        # already bounded by validation, so the minimal payload fits.
        payload["error"] = {
            "code": error.code,
            "message": "error detail omitted (frame limit)",
        }
        frame = json.dumps(payload).encode("utf-8")
    return frame + b"\n"
