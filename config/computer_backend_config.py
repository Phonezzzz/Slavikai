from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ComputerBackendMode = Literal["local", "container"]
ContainerEngine = Literal["docker", "podman"]


@dataclass(frozen=True)
class ComputerBackendConfig:
    """Configures which ComputerBackend Agent.make_computer_runtime() creates.

    Default: mode="local"     -> LocalComputerBackend via ToolGateway.
    Opt-in:  mode="container" -> ContainerComputerBackend via docker/podman run --rm.
    """

    mode: ComputerBackendMode = "local"
    container_image: str = ""
    container_engine: ContainerEngine = "docker"
    container_workspace: str = "/workspace"
    rootless_state_root: str | None = None
    capture_file_bytes: int | None = None
    capture_total_bytes: int | None = None
    capture_entries: int | None = None
    capture_depth: int | None = None
    capture_output_bytes: int | None = None


def _required_positive_env(suffix: str) -> int:
    name = "SLAVIK_COMPUTER_CAPTURE_" + suffix
    raw = os.getenv(name, "").strip()
    if not raw or any(character not in "0123456789" for character in raw):
        raise ValueError(f"{name} must be an explicit positive integer")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be an explicit positive integer")
    return value


def resolve_computer_backend_config() -> ComputerBackendConfig:
    """Read SLAVIK_COMPUTER_* env vars and return a validated config.

    Env vars:
      SLAVIK_COMPUTER_BACKEND            local|container  (default: local)
      SLAVIK_COMPUTER_CONTAINER_IMAGE    image name       (required if backend=container)
      SLAVIK_COMPUTER_CONTAINER_ENGINE   docker|podman    (default: docker)
      SLAVIK_COMPUTER_CONTAINER_WORKSPACE  path           (default: /workspace)

    Raises ValueError on invalid values or missing required fields.
    Never silently falls back to local when container is explicitly requested.
    """
    mode_raw = os.getenv("SLAVIK_COMPUTER_BACKEND", "").strip().lower()
    if not mode_raw or mode_raw == "local":
        mode: ComputerBackendMode = "local"
    elif mode_raw == "container":
        mode = "container"
    else:
        raise ValueError(
            f"SLAVIK_COMPUTER_BACKEND must be 'local' or 'container', got {mode_raw!r}"
        )

    engine_raw = os.getenv("SLAVIK_COMPUTER_CONTAINER_ENGINE", "docker").strip().lower()
    if engine_raw == "docker":
        engine: ContainerEngine = "docker"
    elif engine_raw == "podman":
        engine = "podman"
    else:
        raise ValueError(
            f"SLAVIK_COMPUTER_CONTAINER_ENGINE must be 'docker' or 'podman', got {engine_raw!r}"
        )

    container_image = os.getenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "").strip()
    workspace_raw = os.getenv("SLAVIK_COMPUTER_CONTAINER_WORKSPACE", "/workspace").strip()
    container_workspace = workspace_raw or "/workspace"

    if mode == "container" and not container_image:
        raise ValueError(
            "SLAVIK_COMPUTER_CONTAINER_IMAGE is required when SLAVIK_COMPUTER_BACKEND=container"
        )

    rootless_root = os.getenv("SLAVIK_COMPUTER_ROOTLESS_STATE", "").strip() or None
    file_bytes = total_bytes = entries = depth = output_bytes = None
    if rootless_root is not None:
        if mode != "container" or engine != "podman" or container_workspace != "/workspace":
            raise ValueError("Rootless state requires container/podman and /workspace")
        if not Path(rootless_root).is_absolute():
            raise ValueError("SLAVIK_COMPUTER_ROOTLESS_STATE must be absolute")
        if Path(rootless_root).resolve() != Path(rootless_root):
            raise ValueError("SLAVIK_COMPUTER_ROOTLESS_STATE must be canonical without aliases")
        file_bytes = _required_positive_env("FILE_BYTES")
        total_bytes = _required_positive_env("TOTAL_BYTES")
        entries = _required_positive_env("ENTRIES")
        depth = _required_positive_env("DEPTH")
        output_bytes = _required_positive_env("OUTPUT_BYTES")

    return ComputerBackendConfig(
        mode=mode,
        container_image=container_image,
        container_engine=engine,
        container_workspace=container_workspace,
        rootless_state_root=rootless_root,
        capture_file_bytes=file_bytes,
        capture_total_bytes=total_bytes,
        capture_entries=entries,
        capture_depth=depth,
        capture_output_bytes=output_bytes,
    )
