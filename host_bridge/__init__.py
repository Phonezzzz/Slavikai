"""Owner Host Bridge: bounded AF_UNIX daemon for owner-host capabilities.

E03-T02 implements lifecycle/auth only (``bridge.ping`` /
``bridge.version`` / ``bridge.capabilities``) per ADR-0013; E03-T03
adds the versioned protocol validation (schema, frame limit, request
ID, timeout). Filesystem and process capabilities land in E03-T04/T05.
"""

from __future__ import annotations

from host_bridge.client import BridgeClientError, call, capabilities, ping, version
from host_bridge.daemon import (
    DEFAULT_REQUEST_TIMEOUT_S,
    BridgeConfigError,
    HostBridgeDaemon,
    PeerCred,
    read_peer_cred,
    resolve_socket_path,
)
from host_bridge.protocol import (
    BRIDGE_VERSION,
    CAPABILITIES,
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    BridgeProtocolError,
    BridgeRequest,
    FrameTooLargeError,
    InvalidRequestIdError,
    MalformedRequestError,
    RequestTimeoutError,
    UnknownCapabilityError,
    UnknownFieldError,
    UnsupportedVersionError,
)

__all__ = [
    "BRIDGE_VERSION",
    "CAPABILITIES",
    "DEFAULT_REQUEST_TIMEOUT_S",
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "BridgeClientError",
    "BridgeConfigError",
    "BridgeProtocolError",
    "BridgeRequest",
    "FrameTooLargeError",
    "HostBridgeDaemon",
    "InvalidRequestIdError",
    "MalformedRequestError",
    "PeerCred",
    "RequestTimeoutError",
    "UnknownCapabilityError",
    "UnknownFieldError",
    "UnsupportedVersionError",
    "call",
    "capabilities",
    "ping",
    "read_peer_cred",
    "resolve_socket_path",
    "version",
]
