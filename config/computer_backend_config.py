from __future__ import annotations

import os
from dataclasses import dataclass
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

    return ComputerBackendConfig(
        mode=mode,
        container_image=container_image,
        container_engine=engine,
        container_workspace=container_workspace,
    )
