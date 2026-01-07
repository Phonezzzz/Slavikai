from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from core.approval_policy import (
    ApprovalContext,
    ApprovalRequired,
    build_approval_request,
    decide_action,
    detect_action_intents,
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

    def call(self, request: ToolRequest) -> ToolResult:
        bypass_safe_mode = False
        if self.approval_context is not None:
            intents = detect_action_intents(request)
            decision = decide_action(context=self.approval_context, intents=intents)
            if decision.status == "require_approval":
                approval_request = build_approval_request(
                    context=self.approval_context,
                    decision=decision,
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
                        },
                    )
                raise ApprovalRequired(approval_request)
            if decision.status == "allow" and intents:
                bypass_safe_mode = True
                if self.log_event and not self.approval_context.safe_mode:
                    categories = [intent.category for intent in intents]
                    self.log_event(
                        "approval_skipped",
                        "Safe mode disabled",
                        {
                            "categories": categories,
                            "tool": request.name,
                            "session_id": self.approval_context.session_id,
                        },
                    )
        context = self.pre_call(request) if self.pre_call else None
        try:
            result = self.registry.call(request, bypass_safe_mode=bypass_safe_mode)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка инструмента %s: %s", request.name, exc)
            result = ToolResult.failure(f"Ошибка инструмента {request.name}: {exc}")
        if self.post_call:
            self.post_call(request, result, context)
        return result
