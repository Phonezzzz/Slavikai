"""Entry point: ``python -m host_bridge`` (systemd user service, ADR-0013)."""

from __future__ import annotations

import signal
import sys
import threading
import types

from host_bridge.daemon import BridgeConfigError, HostBridgeDaemon, resolve_socket_path


def main() -> int:
    try:
        socket_path = resolve_socket_path()
    except BridgeConfigError as exc:
        print(f"host-bridge: {exc}", file=sys.stderr)
        return 2
    daemon = HostBridgeDaemon(socket_path)
    stop = threading.Event()

    def _request_stop(signum: int, frame: types.FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        daemon.start()
    except BridgeConfigError as exc:
        print(f"host-bridge: {exc}", file=sys.stderr)
        return 2
    server_thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    server_thread.start()
    stop.wait()
    daemon.shutdown()
    server_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
