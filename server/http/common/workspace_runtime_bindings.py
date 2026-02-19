from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from config.ui_embeddings_settings import UIEmbeddingsSettings
from server.http.common import workspace_index, workspace_paths, workspace_runtime
from server.ui_hub import UIHub
from shared.models import JSONValue


@dataclass(frozen=True)
class WorkspaceRuntimeBindings:
    workspace_root_getter: Callable[[], Path]
    load_embeddings_settings_fn: Callable[[], UIEmbeddingsSettings]
    resolve_provider_api_key_fn: Callable[[str], str | None]
    index_enabled_env: str
    workspace_index_ignored_dirs: set[str]
    workspace_index_allowed_extensions: set[str]
    workspace_index_max_file_bytes: int
    plan_audit_timeout_seconds: int
    plan_audit_max_total_bytes: int
    plan_audit_max_read_files: int

    def _workspace_root(self) -> Path:
        return self.workspace_root_getter()

    async def workspace_root_for_session(self, hub: UIHub, session_id: str) -> Path:
        return await workspace_paths.workspace_root_for_session(
            hub=hub,
            session_id=session_id,
            fallback_root=self._workspace_root(),
        )

    def resolve_workspace_root_candidate(self, path_raw: str, *, policy_profile: str) -> Path:
        return workspace_paths.resolve_workspace_root_candidate(
            path_raw,
            policy_profile=policy_profile,
            workspace_root=self._workspace_root(),
        )

    def index_workspace_root(self, root: Path) -> dict[str, JSONValue]:
        return workspace_index.index_workspace_root(
            root=root,
            load_embeddings_settings=self.load_embeddings_settings_fn,
            resolve_provider_api_key=self.resolve_provider_api_key_fn,
            index_enabled_env=self.index_enabled_env,
            ignored_dirs=self.workspace_index_ignored_dirs,
            allowed_extensions=self.workspace_index_allowed_extensions,
            max_file_bytes=self.workspace_index_max_file_bytes,
        )

    def workspace_git_diff(self, root: Path) -> tuple[str, str | None]:
        return workspace_index.workspace_git_diff(root)

    def run_plan_readonly_audit(
        self,
        *,
        root: Path,
    ) -> tuple[list[dict[str, JSONValue]], dict[str, int]]:
        return workspace_runtime._run_plan_readonly_audit(
            root=root,
            plan_audit_timeout_seconds=self.plan_audit_timeout_seconds,
            workspace_index_ignored_dirs=self.workspace_index_ignored_dirs,
            workspace_index_allowed_extensions=self.workspace_index_allowed_extensions,
            plan_audit_max_total_bytes=self.plan_audit_max_total_bytes,
            plan_audit_max_read_files=self.plan_audit_max_read_files,
        )
