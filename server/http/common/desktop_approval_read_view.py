from __future__ import annotations

import hashlib
import re

from core.desktop_policy import DesktopApprovalRule
from shared.models import JSONValue


def project_saved_desktop_rule(rule: DesktopApprovalRule) -> dict[str, JSONValue]:
    scope = rule.scope
    target = scope.target_pattern
    path = (
        target
        if target is not None
        and target.startswith("/")
        and len(target) <= 4096
        and all(ord(char) >= 32 for char in target)
        else None
    )
    return {
        "reference": hashlib.sha256(rule.rule_id.encode()).hexdigest(),
        "effect": rule.effect,
        "source": rule.source,
        "tool": scope.tool if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", scope.tool) else None,
        "action": scope.action if re.fullmatch(r"[a-z][a-z0-9_]{0,79}", scope.action) else None,
        "target_path": path,
        "target_kind": "none" if target is None else "path" if path is not None else "hidden",
        "command_restriction_present": scope.command_exact is not None
        or scope.command_class is not None,
        "risk_restriction_present": scope.risk_class is not None,
    }
