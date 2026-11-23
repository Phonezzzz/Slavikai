"""Owner Host Bridge protocol validation (E03-T03, ADR-0013 §6).

Versioned request schema, formal frame limit, required client-generated
request ID echoed in every response, deterministic typed rejections
(unsupported version, unknown field, invalid request ID, malformed
JSON, oversized frame), and the per-request daemon deadline. Unknown
fields never become command arguments; ``principal_id`` / ``run_id``
are attribution data only and never authorize anything.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from host_bridge import client
from host_bridge.daemon import HostBridgeDaemon
from host_bridge.protocol import (
    MAX_FRAME_BYTES,
    MAX_REQUEST_ID_CHARS,
    PROTOCOL_VERSION,
    BridgeProtocolError,
    BridgeRequest,
    FrameTooLargeError,
    InvalidRequestIdError,
    MalformedRequestError,
    UnknownFieldError,
    UnsupportedVersionError,
    decode_request,
    encode_error,
    encode_request,
    encode_success,
    read_frame,
)


def _frame(payload: dict[str, object]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": PROTOCOL_VERSION,
        "request_id": "req-1",
        "capability": "bridge.ping",
    }
    payload.update(overrides)
    return payload


# --- schema validation (unit) ---


def test_decode_minimal_valid_request() -> None:
    request = decode_request(_frame(_valid_payload()))
    assert request == BridgeRequest(version=1, request_id="req-1", capability="bridge.ping")
    assert request.principal_id is None
    assert request.run_id is None
    assert request.params == {}


def test_decode_full_valid_request() -> None:
    request = decode_request(
        _frame(
            _valid_payload(
                principal_id="owner",
                run_id="run-42",
                params={"path": "/tmp/x"},
            )
        )
    )
    assert request.principal_id == "owner"
    assert request.run_id == "run-42"
    assert request.params == {"path": "/tmp/x"}


def test_encode_decode_round_trip() -> None:
    wire = encode_request(
        "bridge.version",
        request_id="abc",
        principal_id="owner",
        run_id="run-7",
        params={"k": 1},
    )
    assert wire.endswith(b"\n")
    request = decode_request(wire.rstrip(b"\n"))
    assert request == BridgeRequest(
        version=1,
        request_id="abc",
        capability="bridge.version",
        principal_id="owner",
        run_id="run-7",
        params={"k": 1},
    )


def test_malformed_json_is_rejected_without_echo() -> None:
    with pytest.raises(MalformedRequestError) as excinfo:
        decode_request(b"this is not json")
    assert excinfo.value.code == "malformed_request"
    assert excinfo.value.request_id is None


def test_non_object_json_is_rejected() -> None:
    with pytest.raises(MalformedRequestError):
        decode_request(b'["bridge.ping"]')


def test_missing_version_is_malformed() -> None:
    payload = _valid_payload()
    del payload["version"]
    with pytest.raises(MalformedRequestError) as excinfo:
        decode_request(_frame(payload))
    assert excinfo.value.request_id == "req-1"  # extractable ID is echoed


@pytest.mark.parametrize("bad_version", ["1", 1.0, True, None])
def test_non_integer_version_is_malformed(bad_version: object) -> None:
    with pytest.raises(MalformedRequestError):
        decode_request(_frame(_valid_payload(version=bad_version)))


def test_unknown_version_is_deterministic_rejection() -> None:
    with pytest.raises(UnsupportedVersionError) as excinfo:
        decode_request(_frame(_valid_payload(version=2)))
    assert excinfo.value.code == "unsupported_version"
    assert excinfo.value.request_id == "req-1"


def test_unknown_field_is_rejected_not_ignored() -> None:
    with pytest.raises(UnknownFieldError) as excinfo:
        decode_request(_frame(_valid_payload(command="rm -rf /")))
    assert excinfo.value.code == "unknown_field"
    assert "command" in str(excinfo.value)
    assert excinfo.value.request_id == "req-1"


@pytest.mark.parametrize(
    "bad_id",
    ["", "x" * (MAX_REQUEST_ID_CHARS + 1), 42, None, ["req-1"]],
)
def test_invalid_request_id_is_rejected(bad_id: object) -> None:
    with pytest.raises(InvalidRequestIdError) as excinfo:
        decode_request(_frame(_valid_payload(request_id=bad_id)))
    assert excinfo.value.code == "invalid_request_id"


def test_missing_request_id_is_rejected() -> None:
    payload = _valid_payload()
    del payload["request_id"]
    with pytest.raises(InvalidRequestIdError):
        decode_request(_frame(payload))


def test_request_id_at_max_length_is_accepted() -> None:
    rid = "x" * MAX_REQUEST_ID_CHARS
    request = decode_request(_frame(_valid_payload(request_id=rid)))
    assert request.request_id == rid


@pytest.mark.parametrize("bad_capability", ["", 7, None])
def test_invalid_capability_is_malformed(bad_capability: object) -> None:
    with pytest.raises(MalformedRequestError):
        decode_request(_frame(_valid_payload(capability=bad_capability)))


@pytest.mark.parametrize("field_name", ["principal_id", "run_id"])
@pytest.mark.parametrize("bad_value", ["", 9, None])
def test_invalid_attribution_field_is_malformed(field_name: str, bad_value: object) -> None:
    with pytest.raises(MalformedRequestError):
        decode_request(_frame(_valid_payload(**{field_name: bad_value})))


def test_non_object_params_is_malformed() -> None:
    with pytest.raises(MalformedRequestError):
        decode_request(_frame(_valid_payload(params=["not", "an", "object"])))


def test_success_envelope_shape() -> None:
    payload = json.loads(encode_success({"pong": True}, "req-9"))
    assert payload == {
        "version": PROTOCOL_VERSION,
        "request_id": "req-9",
        "ok": True,
        "result": {"pong": True},
    }


def test_error_envelope_echoes_request_id() -> None:
    payload = json.loads(encode_error(UnsupportedVersionError(3, request_id="req-3")))
    assert payload["version"] == PROTOCOL_VERSION
    assert payload["request_id"] == "req-3"
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unsupported_version"


# --- frame limit (unit) ---


def test_read_frame_accepts_frame_at_limit() -> None:
    frame = b"x" * MAX_FRAME_BYTES
    reader = io.BytesIO(frame + b"\n")
    assert read_frame(reader) == frame


def test_read_frame_rejects_frame_over_limit() -> None:
    reader = io.BytesIO(b"x" * (MAX_FRAME_BYTES + 1) + b"\n")
    with pytest.raises(FrameTooLargeError):
        read_frame(reader)


def test_read_frame_rejects_unterminated_oversized_stream() -> None:
    reader = io.BytesIO(b"x" * (MAX_FRAME_BYTES + 100))
    with pytest.raises(FrameTooLargeError):
        read_frame(reader)


def test_read_frame_eof_and_unterminated_final_frame() -> None:
    assert read_frame(io.BytesIO(b"")) is None
    assert read_frame(io.BytesIO(b"{}")) == b"{}"


# --- over the real socket ---


def _start_daemon(daemon: HostBridgeDaemon) -> threading.Thread:
    daemon.start()
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    return thread


@pytest.fixture()
def bridge(tmp_path: Path) -> Iterator[Path]:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path)
    thread = _start_daemon(daemon)
    try:
        yield socket_path
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def _raw_roundtrip(socket_path: Path, payload: bytes) -> dict[str, object]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(5.0)
        conn.connect(str(socket_path))
        conn.sendall(payload)
        line = conn.makefile("rb").readline(MAX_FRAME_BYTES + 1)
    parsed = json.loads(line)
    assert isinstance(parsed, dict)
    return parsed


def test_client_round_trip_with_attribution_fields(bridge: Path) -> None:
    result = client.call(
        bridge,
        "bridge.ping",
        request_id="fixed-id",
        principal_id="owner",
        run_id="run-1",
    )
    assert result == {"pong": True}


def test_response_echoes_request_id(bridge: Path) -> None:
    payload = _raw_roundtrip(
        bridge,
        b'{"version": 1, "request_id": "echo-me", "capability": "bridge.version"}\n',
    )
    assert payload["request_id"] == "echo-me"
    assert payload["version"] == PROTOCOL_VERSION
    assert payload["ok"] is True


def test_unknown_version_over_socket_keeps_connection_usable(bridge: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(5.0)
        conn.connect(str(bridge))
        reader = conn.makefile("rb")
        conn.sendall(b'{"version": 99, "request_id": "r-1", "capability": "bridge.ping"}\n')
        rejected = json.loads(reader.readline(MAX_FRAME_BYTES + 1))
        assert rejected["error"]["code"] == "unsupported_version"
        assert rejected["request_id"] == "r-1"
        conn.sendall(b'{"version": 1, "request_id": "r-2", "capability": "bridge.ping"}\n')
        accepted = json.loads(reader.readline(MAX_FRAME_BYTES + 1))
        assert accepted["ok"] is True
        assert accepted["request_id"] == "r-2"


def test_unknown_field_over_socket_is_typed_rejection(bridge: Path) -> None:
    payload = _raw_roundtrip(
        bridge,
        b'{"version": 1, "request_id": "r-1", "capability": "bridge.ping",'
        b' "argv": ["rm", "-rf"]}\n',
    )
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unknown_field"


def test_missing_request_id_over_socket_is_typed_rejection(bridge: Path) -> None:
    payload = _raw_roundtrip(bridge, b'{"version": 1, "capability": "bridge.ping"}\n')
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_request_id"
    assert payload["request_id"] is None


def test_malformed_request_closes_connection(bridge: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(5.0)
        conn.connect(str(bridge))
        reader = conn.makefile("rb")
        conn.sendall(b"not json\n")
        rejected = json.loads(reader.readline(MAX_FRAME_BYTES + 1))
        assert rejected["error"]["code"] == "malformed_request"
        assert reader.readline(MAX_FRAME_BYTES + 1) == b""  # daemon closed


def test_slow_handler_hits_daemon_deadline(tmp_path: Path) -> None:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"

    def slow_dispatch(request: BridgeRequest) -> dict[str, object]:
        time.sleep(1.0)
        return {"pong": True}

    daemon = HostBridgeDaemon(
        socket_path,
        request_timeout_s=0.05,
        dispatch_fn=slow_dispatch,
    )
    thread = _start_daemon(daemon)
    try:
        started = time.monotonic()
        with pytest.raises(client.BridgeClientError) as excinfo:
            client.ping(socket_path, timeout=5.0)
        elapsed = time.monotonic() - started
        assert excinfo.value.code == "request_timeout"
        assert elapsed < 0.5  # answered by the deadline, not the handler
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def test_client_side_timeout_is_typed(tmp_path: Path) -> None:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"

    def slow_dispatch(request: BridgeRequest) -> dict[str, object]:
        time.sleep(1.0)
        return {"pong": True}

    daemon = HostBridgeDaemon(
        socket_path,
        request_timeout_s=30.0,  # daemon deadline far away; client gives up first
        dispatch_fn=slow_dispatch,
    )
    thread = _start_daemon(daemon)
    try:
        with pytest.raises(client.BridgeClientError) as excinfo:
            client.ping(socket_path, timeout=0.1)
        assert excinfo.value.code == "request_timeout"
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


# --- error-size bound and typed-error preservation (PR #195 rework) ---


def test_unknown_field_error_message_is_bounded() -> None:
    error = UnknownFieldError(["x" * 5000], request_id="r-1")
    assert len(str(error)) < 256
    assert len(encode_error(error)) <= MAX_FRAME_BYTES + 1


def test_encode_error_backstop_caps_oversized_message() -> None:
    error = BridgeProtocolError("some_code", "y" * 200_000, request_id="r-1")
    frame = encode_error(error)
    assert len(frame) <= MAX_FRAME_BYTES + 1
    payload = json.loads(frame)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "some_code"
    assert payload["request_id"] == "r-1"


def test_oversized_unknown_field_name_over_socket_stays_typed(bridge: Path) -> None:
    field_name = "f" * 60_000
    frame = (
        b'{"version": 1, "request_id": "r-1", "capability": "bridge.ping", "'
        + field_name.encode("ascii")
        + b'": 1}\n'
    )
    assert len(frame) - 1 <= MAX_FRAME_BYTES
    payload = _raw_roundtrip(bridge, frame)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unknown_field"
    assert payload["request_id"] == "r-1"


def test_client_empty_request_id_surfaces_typed_error(bridge: Path) -> None:
    with pytest.raises(client.BridgeClientError) as excinfo:
        client.call(bridge, "bridge.ping", request_id="")
    assert excinfo.value.code == "invalid_request_id"


def test_client_overlong_request_id_surfaces_typed_error(bridge: Path) -> None:
    with pytest.raises(client.BridgeClientError) as excinfo:
        client.call(bridge, "bridge.ping", request_id="x" * (MAX_REQUEST_ID_CHARS + 1))
    assert excinfo.value.code == "invalid_request_id"
