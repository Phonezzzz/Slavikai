from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from core.approval_policy import (
    ApprovalContext,
    ApprovalRequired,
    build_approval_request,
    decide_request,
)
from shared.models import JSONValue, ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry

logger = logging.getLogger("SlavikAI.ToolGateway")


@dataclass
class ToolGateway:
    registry: ToolRegistry
    pre_call: Callable[[ToolRequest], object | None] | None = None
    post_call: Callable[[ToolRequest, ToolResult, object | None], None] | None = None
    approval_context: ApprovalContext | None = None
    log_event: Callable[[str, str, dict[str, JSONValue] | None], None] | None = None
    confirmed_decision: bool = False

    def call(self, request: ToolRequest) -> ToolResult:
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
                return ToolResult.failure(
                    f"POLICY_DENY: {decision.reason}",
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
        context = self.pre_call(request) if self.pre_call else None
        try:
            result = self.registry.call(
                request,
                bypass_safe_mode=bypass_safe_mode,
                confirmed_decision=self.confirmed_decision,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка инструмента %s: %s", request.name, exc)
            result = ToolResult.failure(f"Ошибка инструмента {request.name}: {exc}")
        if self.post_call:
            self.post_call(request, result, context)
        return result
