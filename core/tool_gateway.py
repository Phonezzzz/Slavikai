from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

from core.approval_policy import (
    ApprovalContext,
    ApprovalRequest,
    ApprovalRequired,
    build_approval_request,
    decide_request,
)
from core.managed_workspace import ManagedWorkspacePool, WorkspaceAccess
from shared.models import JSONValue, ToolRequest, ToolResult
from shared.tool_capture import ToolCaptureFault
from shared.workspace_binding import (
    WorkspaceBinding,
    WorkspaceConflict,
    WorkspaceFileBudget,
    WorkspaceFileBudgetExceeded,
    WorkspaceUnavailable,
)
from tools.tool_registry import ToolRegistry

logger = logging.getLogger("SlavikAI.ToolGateway")

_WORKSPACE_WRITERS = frozenset(
    {
        "workspace_write",
        "workspace_create",
        "workspace_rename",
        "workspace_move",
        "workspace_delete",
        "workspace_patch",
    }
)
_WORKSPACE_READERS = frozenset({"workspace_read", "workspace_list"})
_WORKSPACE_OPERATIONS = _WORKSPACE_WRITERS | _WORKSPACE_READERS | {"workspace_run"}


def _workspace_failure(exc: OSError | WorkspaceUnavailable) -> ToolResult:
    return ToolResult.failure(
        str(exc),
        meta={
            "reason_code": (
                "workspace_conflict"
                if isinstance(exc, WorkspaceConflict)
                else "workspace_unavailable"
            ),
            "execution_started": False,
            "execution_review_required": isinstance(exc, WorkspaceConflict),
        },
    )


@dataclass
class AutoFileEffects:
    limit: int
    touched_paths: set[str] = field(default_factory=set)
    complete: bool = True

    @property
    def count(self) -> int | None:
        return len(self.touched_paths) if self.complete else None

    def observe(self, name: str, result: ToolResult) -> None:
        meta = result.meta or {}
        publication = result.data.get("publication")
        if name == "workspace_run" and publication == "published":
            paths = result.data.get("changed_paths")
        elif name in _WORKSPACE_WRITERS and result.ok:
            paths = meta.get("workspace_changed_paths")
        elif meta.get("execution_started") is False or (
            name == "workspace_run"
            and publication == "none"
            and meta.get("execution_review_required") is False
            and meta.get("reason_code") == WorkspaceFileBudgetExceeded.reason_code
        ):
            return
        else:
            paths = None
        if isinstance(paths, list) and all(
            isinstance(path, str)
            and path not in {"", "."}
            and "\x00" not in path
            and not PurePosixPath(path).is_absolute()
            and ".." not in PurePosixPath(path).parts
            and PurePosixPath(path).as_posix() == path
            for path in paths
        ):
            self.touched_paths.update(paths)
            return
        self.complete = False


@dataclass
class ToolGateway:
    registry: ToolRegistry
    pre_call: Callable[[ToolRequest], object | None] | None = None
    post_call: Callable[[ToolRequest, ToolResult, object | None], None] | None = None
    approval_context: ApprovalContext | None = None
    log_event: Callable[[str, str, dict[str, JSONValue] | None], None] | None = None
    confirmed_decision: bool = False
    workspace_access: WorkspaceAccess | None = None
    approved_workspace_binding: WorkspaceBinding | None = None
    workspace_pool: ManagedWorkspacePool | None = None
    workspace_principal_id: str | None = None
    auto_file_effects: AutoFileEffects | None = None

    def _workspace_binding(self, request: ToolRequest) -> WorkspaceBinding | None:
        if request.name not in _WORKSPACE_OPERATIONS:
            return None
        access = self.workspace_access
        if access is None and self.workspace_pool is None:
            return None
        from tools.workspace_tools import get_workspace_root

        root = get_workspace_root()
        if self.workspace_pool is not None:
            if self.workspace_principal_id is None:
                raise WorkspaceUnavailable("workspace principal unavailable")
            self.workspace_pool.resolve_root(self.workspace_principal_id, root)
        if access is None:
            return None
        if (
            self.approval_context is None
            or self.approval_context.session_id != access.session_id
            or self.approval_context.execution_target != "sandbox"
        ):
            raise WorkspaceUnavailable("workspace execution context mismatch")
        if root != access.workspace.root:
            parent = access.workspace.root.parent
            if root.is_relative_to(parent) or parent.is_relative_to(root):
                raise WorkspaceUnavailable("workspace root overlaps protected managed namespace")
            return None
        return access.bind()

    def call_approved_once(self, request: ToolRequest, approval: ApprovalRequest) -> ToolResult:
        context = self.approval_context
        subject = json.dumps(
            {"name": request.name, "args": request.args}, sort_keys=True, allow_nan=False
        )
        if (
            context is None
            or context.session_id != approval.session_id
            or subject != approval.request_json
        ):
            raise ValueError("approval_subject_mismatch")
        if request.name in _WORKSPACE_OPERATIONS:
            try:
                if self._workspace_binding(request) != approval.workspace_binding:
                    raise WorkspaceConflict("workspace approval binding changed")
            except (OSError, WorkspaceUnavailable) as exc:
                return _workspace_failure(exc)
        bound_gateway = replace(self, approved_workspace_binding=approval.workspace_binding)
        if context.execution_target == "desktop":
            # Scoped Desktop rules remain authoritative; never inject categories.
            return bound_gateway.call(request)
        decision, _ = decide_request(
            context=context,
            request=request,
            risk_classes=self.registry.get_risk_classes(request.name),
        )
        if decision.status != "require_approval":
            return bound_gateway.call(request)
        if set(decision.required_categories) != set(approval.required_categories):
            return bound_gateway.call(request)
        scoped_gateway = replace(
            bound_gateway,
            approval_context=replace(
                context,
                approved_categories=context.approved_categories | set(approval.required_categories),
            ),
        )
        return scoped_gateway.call(request)

    def call(self, request: ToolRequest) -> ToolResult:
        binding: WorkspaceBinding | None = None
        workspace_access = self.workspace_access
        try:
            binding = self._workspace_binding(request)
            if (
                self.approved_workspace_binding is not None
                and binding != self.approved_workspace_binding
            ):
                raise WorkspaceConflict("workspace approval binding changed")
        except (OSError, WorkspaceUnavailable) as exc:
            return _workspace_failure(exc)
        # Python-only binding всегда выдаётся этим scoped Gateway, не tool arguments.
        request = replace(request, workspace_binding=binding, workspace_file_budget=None)
        bypass_safe_mode = False
        confirmed_server_tool = False
        if self.confirmed_decision:
            descriptor = self.registry.get_descriptor(request.name)
            confirmed_server_tool = bool(
                descriptor is not None and descriptor.confirmed_decision_only
            )
        if self.approval_context is not None and not confirmed_server_tool:
            decision, scope = decide_request(
                context=self.approval_context,
                request=request,
                risk_classes=self.registry.get_risk_classes(request.name),
            )
            if decision.status == "require_approval":
                approval_request = build_approval_request(
                    context=self.approval_context,
                    decision=decision,
                    scope=scope,
                )
                if approval_request is None:
                    raise RuntimeError("Approval request was not built.")
                if self.log_event:
                    self.log_event(
                        "approval_required",
                        approval_request.prompt.what,
                        {
                            "category": approval_request.category,
                            "required_categories": approval_request.required_categories,
                            "tool": approval_request.tool,
                            "details": approval_request.details,
                            "session_id": approval_request.session_id,
                            "policy_rule_id": approval_request.policy_rule_id,
                        },
                    )
                approval_request = replace(
                    approval_request,
                    workspace_binding=binding,
                    request_json=json.dumps(
                        {"name": request.name, "args": request.args},
                        sort_keys=True,
                        allow_nan=False,
                    ),
                )
                raise ApprovalRequired(approval_request)
            if decision.status == "block":
                if self.log_event:
                    self.log_event(
                        "policy_denied",
                        decision.reason,
                        {
                            "tool": request.name,
                            "session_id": self.approval_context.session_id,
                            "execution_target": self.approval_context.execution_target,
                            "policy_rule_id": decision.policy_rule_id,
                        },
                    )
                block_message = f"POLICY_DENY: {decision.reason}"
                if decision.reason == "command_denied:hard_safety" and decision.intents:
                    command = str(decision.intents[0].details.get("command") or "")
                    block_message = (
                        "Команда запрещена политикой безопасности: sudo, rm -rf, shutdown, "
                        "reboot, mkfs, fork bomb и запись в системные файлы недоступны "
                        "даже в YOLO."
                    )
                    if command:
                        block_message = f"{block_message} Команда: {command}"
                return ToolResult.failure(
                    block_message,
                    meta={"policy_reason": decision.reason},
                )
            if decision.status == "allow" and decision.intents:
                bypass_safe_mode = True
                if self.log_event:
                    categories = [intent.category for intent in decision.intents]
                    self.log_event(
                        "policy_allowed",
                        decision.reason,
                        {
                            "categories": categories,
                            "tool": request.name,
                            "session_id": self.approval_context.session_id,
                            "execution_target": self.approval_context.execution_target,
                            "policy_rule_id": decision.policy_rule_id,
                        },
                    )
        effects = self.auto_file_effects
        if effects is not None and (
            request.name == "workspace_run" or request.name in _WORKSPACE_WRITERS
        ):
            if request.name in _WORKSPACE_WRITERS and binding is None:
                return _workspace_failure(
                    WorkspaceUnavailable("Файловый бюджет требует managed workspace.")
                )
            if not effects.complete:
                return ToolResult.failure(
                    "Нельзя проверить файловый лимит. Эффекты предыдущего вызова неизвестны.",
                    meta={
                        "reason_code": "workspace_file_effects_unknown",
                        "execution_started": False,
                    },
                )
            request = replace(
                request,
                workspace_file_budget=WorkspaceFileBudget(
                    effects.limit, frozenset(effects.touched_paths)
                ),
            )
        context = self.pre_call(request) if self.pre_call else None
        try:
            guard = None
            if workspace_access is not None and binding is not None:
                if request.name in _WORKSPACE_WRITERS:
                    guard = workspace_access.workspace.mutation(binding)
                elif request.name in _WORKSPACE_READERS:
                    guard = workspace_access.workspace.inspection(binding)
            if guard is not None:
                result = self.registry.call(
                    request,
                    bypass_safe_mode=bypass_safe_mode,
                    confirmed_decision=self.confirmed_decision,
                    effect_guard=guard,
                )
            else:
                result = self.registry.call(
                    request,
                    bypass_safe_mode=bypass_safe_mode,
                    confirmed_decision=self.confirmed_decision,
                )
        except WorkspaceUnavailable as exc:
            result = _workspace_failure(exc)
        except ToolCaptureFault:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка инструмента %s: %s", request.name, exc)
            result = ToolResult.failure(f"Ошибка инструмента {request.name}: {exc}")
        if effects is not None and (
            request.name == "workspace_run"
            or request.name in _WORKSPACE_WRITERS
            or self.registry.get_capability(request.name) in {"write", "exec"}
        ):
            effects.observe(request.name, result)
        if self.post_call:
            self.post_call(request, result, context)
        return result
