from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shared.sandbox import crosses_nested_mount, read_mount_bindings, source_mount_root

ProtectionLevel = Literal["normal", "ask", "deny"]


@dataclass(frozen=True, slots=True)
class ResolvedDesktopPath:
    raw: str
    lexical: Path
    canonical: Path
    protection: ProtectionLevel
    reason: str


class DesktopPathSecurity:
    """Canonical host path resolver shared by policy evaluation and host tools."""

    def __init__(
        self,
        *,
        home: Path | None = None,
        policy_store_path: Path | None = None,
        protected_paths: Sequence[Path] = (),
    ) -> None:
        self.home = (home or Path.home()).expanduser().resolve()
        self.policy_store_path = (
            policy_store_path.expanduser().resolve() if policy_store_path is not None else None
        )
        self.policy_store_root = (
            self.policy_store_path.parent if self.policy_store_path is not None else None
        )
        self._protected_prefixes = tuple(path.expanduser().resolve() for path in protected_paths)
        self._deny_prefixes = tuple(
            path.resolve()
            for path in (
                Path("/boot"),
                Path("/dev"),
                Path("/proc"),
                Path("/sys"),
                Path("/root"),
            )
        )
        self._ask_prefixes = (Path("/etc").resolve(),)
        self._credential_prefixes = tuple(
            (self.home / name).resolve()
            for name in (
                ".ssh",
                ".gnupg",
                ".aws",
                ".azure",
                ".config/gcloud",
                ".docker",
                ".kube",
                ".password-store",
            )
        )

    def resolve(
        self,
        raw: str,
        *,
        must_exist: bool = False,
        mutation: bool = False,
        includes_descendants: bool = False,
    ) -> ResolvedDesktopPath:
        normalized = raw.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("Путь пуст или содержит NUL")
        if normalized == "~":
            lexical = self.home
        elif normalized.startswith("~/"):
            lexical = self.home / normalized[2:]
        else:
            candidate = Path(normalized)
            lexical = candidate if candidate.is_absolute() else self.home / candidate
        lexical = Path(os.path.normpath(str(lexical))).absolute()
        canonical = lexical.resolve(strict=must_exist)
        protection, reason = self._classify(
            canonical, mutation=mutation, includes_descendants=includes_descendants
        )
        try:
            mounts = read_mount_bindings()
        except (OSError, ValueError):
            mounts = []
        if includes_descendants and not mounts:
            protection, reason = "deny", "mount_topology_unavailable"
        elif includes_descendants and crosses_nested_mount(canonical, mounts):
            protection, reason = "deny", "nested_mount"
        mount_root = source_mount_root(canonical, mounts)
        if mount_root is not None and protection != "deny":
            mounted_protection, mounted_reason = self._classify(
                mount_root,
                mutation=mutation,
                includes_descendants=includes_descendants,
            )
            if mounted_protection == "deny":
                protection = "deny"
                reason = f"protected_bind_mount:{mounted_reason}"
            elif mounted_protection == "ask" and protection == "normal":
                protection = "ask"
                reason = f"sensitive_bind_mount:{mounted_reason}"
        return ResolvedDesktopPath(
            raw=raw,
            lexical=lexical,
            canonical=canonical,
            protection=protection,
            reason=reason,
        )

    def require_not_denied(
        self,
        raw: str,
        *,
        must_exist: bool = False,
        mutation: bool = False,
        includes_descendants: bool = False,
    ) -> ResolvedDesktopPath:
        resolved = self.resolve(
            raw, must_exist=must_exist, mutation=mutation, includes_descendants=includes_descendants
        )
        if resolved.protection == "deny":
            raise PermissionError(f"PROTECTED_RESOURCE_DENY: {resolved.reason}")
        return resolved

    def _classify(
        self, path: Path, *, mutation: bool, includes_descendants: bool
    ) -> tuple[ProtectionLevel, str]:
        if path == Path("/") or (mutation and path == self.home):
            return "deny", "filesystem_or_home_root"
        if self.policy_store_root is not None and (
            _is_within(path, self.policy_store_root)
            or (includes_descendants and _is_within(self.policy_store_root, path))
        ):
            return "deny", "desktop_policy_store"
        for prefix in self._protected_prefixes:
            if _is_within(path, prefix):
                return "deny", "desktop_enforcement_or_config"
            if (mutation or includes_descendants) and _is_within(prefix, path):
                return "deny", "ancestor_of_desktop_enforcement_or_config"
        for prefix in self._deny_prefixes:
            if _is_within(path, prefix):
                return "deny", str(prefix)
        for prefix in self._credential_prefixes:
            if _is_within(path, prefix) or (includes_descendants and _is_within(prefix, path)):
                return "deny", "credentials"
        lowered_name = path.name.lower()
        if lowered_name in {
            "id_rsa",
            "id_ed25519",
            "credentials",
            "credentials.json",
            "secrets.json",
            ".env",
            ".netrc",
            ".npmrc",
            ".pypirc",
        }:
            return "deny", "credential_or_secret_file"
        if lowered_name.endswith((".pem", ".key")):
            return "deny", "private_key_file"
        for prefix in self._ask_prefixes:
            if _is_within(path, prefix):
                return "ask", str(prefix)
        return "normal", "ordinary_host_path"


def _is_within(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False
