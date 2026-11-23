"""Owner Host Bridge: bounded AF_UNIX daemon for owner-host capabilities.

E03-T02 implements lifecycle/auth only (``bridge.ping`` /
``bridge.version`` / ``bridge.capabilities``) per ADR-0013. Filesystem
and process capabilities land in E03-T04/T05; protocol validation in
E03-T03.
"""

from __future__ import annotations

from host_bridge.client import BridgeClientError, call, capabilities, ping, version
from host_bridge.daemon import (
    BridgeConfigError,
    HostBridgeDaemon,
    PeerCred,
    read_peer_cred,
    resolve_socket_path,
)
from host_bridge.protocol import (
    BRIDGE_VERSION,
    CAPABILITIES,
    PROTOCOL_VERSION,
    BridgeProtocolError,
    MalformedRequestError,
    UnknownCapabilityError,
)

__all__ = [
    "BRIDGE_VERSION",
    "CAPABILITIES",
    "PROTOCOL_VERSION",
    "BridgeClientError",
    "BridgeConfigError",
    "BridgeProtocolError",
    "HostBridgeDaemon",
    "MalformedRequestError",
    "PeerCred",
    "UnknownCapabilityError",
    "call",
    "capabilities",
    "ping",
    "read_peer_cred",
    "resolve_socket_path",
    "version",
]
