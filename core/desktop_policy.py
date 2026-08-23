from __future__ import annotations

import json
import logging
import os
import shlex
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from shared.models import JSONValue, ToolRequest

PolicyEffect = Literal["allow", "ask", "deny"]
RuleSource = Literal["builtin", "once", "session", "persistent"]

DEFAULT_DESKTOP_POLICY_PATH = Path(".run/desktop_approvals.json")
logger = logging.getLogger("SlavikAI.DesktopPolicy")


@dataclass(frozen=True, slots=True)
class DesktopAction:
    tool: str
    action: str
    target: str | None = None
    command_class: str | None = None
    command: str | None = None
    risk_class: str | None = None
    execution_target: str = "desktop"


@dataclass(frozen=True, slots=True)
class DesktopApprovalScope:
    tool: str
    action: str
    target_pattern: str | None = None
    command_class: str | None = None
    command_exact: str | None = None
    risk_class: str | None = None
    execution_target: str = "desktop"

    def matches(self, action: DesktopAction) -> bool:
        if self.execution_target != action.execution_target:
            return False
        if self.tool != action.tool or self.action != action.action:
            return False
        if self.command_class is not None and self.command_class != action.command_class:
            return False
        if self.command_exact is not None and action.command != self.command_exact:
            return False
        if self.risk_class is not None and self.risk_class != action.risk_class:
            return False
        if self.target_pattern is None:
            return action.target is None
        if action.target is None:
            return False
        if self.target_pattern.endswith("/**"):
            root = self.target_pattern[:-3].rstrip("/") or "/"
            descendant_prefix = "/" if root == "/" else f"{root}/"
            if action.target != root and not action.target.startswith(descendant_prefix):
                return False
            return True
        return action.target == self.target_pattern

    @property
    def specificity(self) -> int:
        score = 20
        if self.target_pattern is not None:
            score += len(self.target_pattern.replace("*", ""))
        if self.command_class is not None:
            score += 10
        if self.command_exact is not None:
            score += len(self.command_exact)
        if self.risk_class is not None:
            score += 5
        return score

    @property
    def supports_persistent_allow(self) -> bool:
        non_persistent_actions = {
            ("desktop_clipboard", "read"),
            ("desktop_clipboard", "write"),
            ("desktop_clipboard", "clear"),
            ("desktop_package", "install"),
            ("desktop_package", "remove"),
            ("desktop_package", "update_metadata"),
            ("desktop_systemd", "enable"),
            ("desktop_systemd", "disable"),
            ("desktop_session", "notify"),
            ("desktop_session", "lock"),
        }
        return (
            self.tool not in {"desktop_shell", "desktop_launch", "desktop_browser", "desktop_gui"}
            and (self.tool, self.action) not in non_persistent_actions
            and self.command_class is None
            and self.command_exact is None
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "tool": self.tool,
            "action": self.action,
            "target_pattern": self.target_pattern,
            "command_class": self.command_class,
            "command_exact": self.command_exact,
            "risk_class": self.risk_class,
            "execution_target": self.execution_target,
        }

    @classmethod
    def from_dict(cls, raw: object) -> DesktopApprovalScope:
        if not isinstance(raw, dict):
            raise ValueError("scope должен быть JSON-объектом")
        if "command_pattern" in raw:
            raise ValueError(
                "scope.command_pattern больше не поддерживается; пересоздайте exact-command rule"
            )
        tool = raw.get("tool")
        action = raw.get("action")
        execution_target = raw.get("execution_target", "desktop")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError("scope.tool должен быть непустой строкой")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("scope.action должен быть непустой строкой")
        if execution_target != "desktop":
            raise ValueError("persistent scope разрешён только для desktop")
        return cls(
            tool=tool.strip(),
            action=action.strip(),
            target_pattern=_optional_string(raw.get("target_pattern")),
            command_class=_optional_string(raw.get("command_class")),
            command_exact=_optional_string(raw.get("command_exact")),
            risk_class=_optional_string(raw.get("risk_class")),
        )


@dataclass(frozen=True, slots=True)
class DesktopApprovalRule:
    rule_id: str
    effect: PolicyEffect
    scope: DesktopApprovalScope
    source: RuleSource
    created_at: str
    description: str = ""
    subject_principal_id: str = "legacy"

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "rule_id": self.rule_id,
            "effect": self.effect,
            "scope": self.scope.to_dict(),
            "source": self.source,
            "created_at": self.created_at,
            "description": self.description,
            "subject_principal_id": self.subject_principal_id,
        }

    @classmethod
    def create(
        cls,
        *,
        effect: PolicyEffect,
        scope: DesktopApprovalScope,
        source: RuleSource,
        description: str = "",
        subject_principal_id: str = "legacy",
    ) -> DesktopApprovalRule:
        return cls(
            rule_id=f"desktop-rule-{uuid.uuid4().hex}",
            effect=effect,
            scope=scope,
            source=source,
            created_at=datetime.now(UTC).isoformat(),
            description=description.strip(),
            subject_principal_id=subject_principal_id.strip() or "legacy",
        )

    @classmethod
    def from_dict(cls, raw: object) -> DesktopApprovalRule:
        if not isinstance(raw, dict):
            raise ValueError("rule должен быть JSON-объектом")
        rule_id = raw.get("rule_id")
        effect = raw.get("effect")
        source = raw.get("source")
        created_at = raw.get("created_at")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError("rule_id должен быть непустой строкой")
        if effect not in {"allow", "ask", "deny"}:
            raise ValueError("effect должен быть allow|ask|deny")
        if source not in {"builtin", "once", "session", "persistent"}:
            raise ValueError("source имеет неизвестное значение")
        if not isinstance(created_at, str) or not created_at.strip():
            raise ValueError("created_at должен быть непустой строкой")
        description_raw = raw.get("description")
        subject_raw = raw.get("subject_principal_id")
        return cls(
            rule_id=rule_id.strip(),
            effect=effect,
            scope=DesktopApprovalScope.from_dict(raw.get("scope")),
            source=source,
            created_at=created_at.strip(),
            description=description_raw.strip() if isinstance(description_raw, str) else "",
            subject_principal_id=(
                subject_raw.strip()
                if isinstance(subject_raw, str) and subject_raw.strip()
                else "legacy"
            ),
        )


@dataclass(frozen=True, slots=True)
class DesktopPolicyResolution:
    effect: PolicyEffect
    reason: str
    rule_id: str | None = None
    consumed_rule_id: str | None = None


class DesktopPolicyRuntime:
    """Immutable policy snapshot with single-use rule consumption for one agent run."""

    def __init__(self, rules: list[DesktopApprovalRule] | None = None) -> None:
        self._rules = list(rules or [])
        self._consumed: list[str] = []
        self._lock = threading.Lock()

    def resolve(
        self,
        action: DesktopAction,
        *,
        default_effect: PolicyEffect,
        default_reason: str,
        consume_once: bool = True,
    ) -> DesktopPolicyResolution:
        with self._lock:
            matching = [rule for rule in self._rules if rule.scope.matches(action)]
            denies = [rule for rule in matching if rule.effect == "deny"]
            if denies:
                selected = max(denies, key=lambda item: item.scope.specificity)
                return DesktopPolicyResolution("deny", "explicit_deny", selected.rule_id)
            allows = [rule for rule in matching if rule.effect == "allow"]
            if allows:
                selected = max(allows, key=lambda item: item.scope.specificity)
                consumed: str | None = None
                if selected.source == "once" and consume_once:
                    self._rules = [item for item in self._rules if item.rule_id != selected.rule_id]
                    self._consumed.append(selected.rule_id)
                    consumed = selected.rule_id
                return DesktopPolicyResolution(
                    "allow",
                    "matching_allow",
                    selected.rule_id,
                    consumed,
                )
            asks = [rule for rule in matching if rule.effect == "ask"]
            if asks:
                selected = max(asks, key=lambda item: item.scope.specificity)
                return DesktopPolicyResolution("ask", "explicit_ask", selected.rule_id)
            return DesktopPolicyResolution(default_effect, default_reason)

    def consume_once_rule_ids(self, rule_ids: set[str]) -> None:
        if not rule_ids:
            return
        with self._lock:
            consumed = [
                rule.rule_id
                for rule in self._rules
                if rule.rule_id in rule_ids and rule.source == "once"
            ]
            if not consumed:
                return
            consumed_ids = set(consumed)
            self._rules = [rule for rule in self._rules if rule.rule_id not in consumed_ids]
            self._consumed.extend(consumed)

    def drain_consumed_rule_ids(self) -> list[str]:
        with self._lock:
            consumed = list(self._consumed)
            self._consumed.clear()
            return consumed


class DesktopPolicyStore:
    def __init__(
        self,
        path: Path | None = None,
        *,
        legacy_subject_principal_id: str = "legacy",
    ) -> None:
        self.path = (path or DEFAULT_DESKTOP_POLICY_PATH).expanduser().resolve()
        normalized_subject = legacy_subject_principal_id.strip()
        self._legacy_subject_principal_id = normalized_subject or "legacy"
        self._lock = threading.Lock()
        self._load_errors: list[str] = []

    def list_rules(
        self,
        *,
        subject_principal_id: str | None = None,
    ) -> list[DesktopApprovalRule]:
        with self._lock:
            rules = self._load_locked()
            if subject_principal_id is None:
                return rules
            normalized_subject = subject_principal_id.strip()
            return [rule for rule in rules if rule.subject_principal_id == normalized_subject]

    def list_load_errors(self) -> list[str]:
        with self._lock:
            return list(self._load_errors)

    def add_rule(self, rule: DesktopApprovalRule) -> DesktopApprovalRule:
        if rule.source != "persistent":
            raise ValueError("DesktopPolicyStore принимает только persistent rules")
        _validate_persistent_rule(rule)
        with self._lock:
            rules = self._load_locked()
            if any(item.rule_id == rule.rule_id for item in rules):
                raise ValueError("rule_id уже существует")
            rules.append(rule)
            self._write_locked(rules)
        return rule

    def update_rule(
        self,
        rule_id: str,
        *,
        effect: PolicyEffect | None = None,
        scope: DesktopApprovalScope | None = None,
        description: str | None = None,
    ) -> DesktopApprovalRule | None:
        with self._lock:
            rules = self._load_locked()
            updated: DesktopApprovalRule | None = None
            next_rules: list[DesktopApprovalRule] = []
            for rule in rules:
                if rule.rule_id != rule_id:
                    next_rules.append(rule)
                    continue
                updated = replace(
                    rule,
                    effect=effect if effect is not None else rule.effect,
                    scope=scope if scope is not None else rule.scope,
                    description=(
                        description.strip() if description is not None else rule.description
                    ),
                )
                _validate_persistent_rule(updated)
                next_rules.append(updated)
            if updated is not None:
                self._write_locked(next_rules)
            return updated

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            rules = self._load_locked()
            next_rules = [rule for rule in rules if rule.rule_id != rule_id]
            if len(next_rules) == len(rules):
                return False
            self._write_locked(next_rules)
            return True

    def _load_locked(self) -> list[DesktopApprovalRule]:
        self._load_errors = []
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Desktop approval store должен содержать JSON-массив")
        rules: list[DesktopApprovalRule] = []
        migrated = False
        for index, item in enumerate(raw):
            try:
                rule = DesktopApprovalRule.from_dict(item)
                if rule.subject_principal_id == "legacy":
                    rule = replace(
                        rule,
                        subject_principal_id=self._legacy_subject_principal_id,
                    )
                    migrated = True
                rules.append(rule)
            except ValueError as exc:
                message = f"rule[{index}] invalid: {exc}"
                self._load_errors.append(message)
                logger.warning(
                    "Desktop approval store rejected due to invalid rule",
                    extra={"path": str(self.path), "index": index, "error": str(exc)},
                )
        if self._load_errors:
            raise ValueError(
                "Desktop approval store contains invalid rules: " + "; ".join(self._load_errors)
            )
        if migrated and not self._load_errors:
            self._write_locked(rules)
        return rules

    def _write_locked(self, rules: list[DesktopApprovalRule]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        payload = [rule.to_dict() for rule in rules]
        try:
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def exact_scope_for_action(action: DesktopAction) -> DesktopApprovalScope:
    return DesktopApprovalScope(
        tool=action.tool,
        action=action.action,
        target_pattern=action.target,
        command_class=action.command_class,
        command_exact=action.command,
        risk_class=action.risk_class,
        execution_target=action.execution_target,
    )


def desktop_actions_from_request(request: ToolRequest) -> list[DesktopAction]:
    args = request.args
    tool = request.name
    target: str | None = None
    action = ""
    command_class: str | None = None
    command: str | None = None
    actions: list[DesktopAction] = []
    risk_class: str | None = None
    if tool == "desktop_file_search":
        action, target = "search", _optional_string(args.get("root"))
    elif tool == "desktop_file_read":
        action, target = "read", _optional_string(args.get("path"))
    elif tool == "desktop_file_write":
        action, target = "write", _optional_string(args.get("path"))
        risk_class = "overwrite" if args.get("overwrite") is True else "create"
    elif tool == "desktop_file_transfer":
        action = str(args.get("operation") or "move").strip().lower()
        target = _optional_string(args.get("destination"))
        risk_class = "overwrite" if args.get("overwrite") is True else "write"
        source = _optional_string(args.get("source"))
        if source is not None:
            source_action = "read_source" if action == "copy" else "move_source"
            actions.append(
                DesktopAction(
                    tool=tool,
                    action=source_action,
                    target=source,
                    risk_class="read" if source_action == "read_source" else "write",
                )
            )
    elif tool == "desktop_file_delete":
        action, target, risk_class = "delete", _optional_string(args.get("path")), "destructive"
    elif tool == "desktop_archive_extract":
        action = "archive_extract"
        target = _optional_string(args.get("destination"))
        risk_class = "write"
        archive = _optional_string(args.get("archive"))
        if archive is not None:
            actions.append(
                DesktopAction(
                    tool=tool,
                    action="read_archive",
                    target=archive,
                    risk_class="read",
                )
            )
    elif tool == "desktop_shell":
        action = "execute"
        command_class = classify_command(args.get("argv"))
        command = _command_string(args.get("argv"))
        target = _optional_string(args.get("cwd"))
        risk_class = command_risk_class(command_class)
    elif tool == "desktop_launch":
        action = "launch"
        command_class = classify_command(args.get("argv"))
        command = _command_string(args.get("argv"))
        target = _optional_string(args.get("cwd"))
        risk_class = command_risk_class(command_class)
    elif tool == "desktop_clipboard":
        action = str(args.get("operation") or "read").strip().lower()
        target = "clipboard"
        risk_class = "sensitive_read" if action == "read" else "external_side_effect"
    elif tool == "desktop_system_info":
        action = str(args.get("operation") or "summary").strip().lower()
        target = _optional_string(args.get("query"))
        risk_class = "read"
    elif tool == "desktop_process":
        action = str(args.get("operation") or "list").strip().lower()
        if action == "launch":
            command_class = classify_command(args.get("argv"))
            command = _command_string(args.get("argv"))
            target = _optional_string(args.get("cwd"))
            risk_class = command_risk_class(command_class)
        else:
            pid = args.get("pid")
            expected = args.get("expected_create_time")
            if isinstance(pid, int) and not isinstance(pid, bool):
                target = f"process:{pid}:{expected}" if expected is not None else f"process:{pid}"
            else:
                target = _optional_string(args.get("query"))
            risk_class = (
                "destructive"
                if action == "kill"
                else "system_impact"
                if action == "terminate"
                else "read"
            )
    elif tool == "desktop_cleanup_unverified_launches":
        action = "rollback"
        target = "current-desktop-run"
        risk_class = "rollback"
    elif tool == "desktop_systemd":
        action = str(args.get("operation") or "status").strip().lower()
        scope = str(args.get("scope") or "system").strip().lower()
        unit = _optional_string(args.get("unit"))
        target = f"{scope}:{unit}" if unit is not None else scope
        risk_class = "read" if action in {"status", "logs"} else "system_impact"
    elif tool == "desktop_package":
        action = str(args.get("operation") or "query").strip().lower()
        target = _optional_string(args.get("package")) or "apt-metadata"
        command_class = (
            "package_management" if action in {"install", "remove", "update_metadata"} else None
        )
        risk_class = (
            "destructive"
            if action == "remove"
            else "install"
            if action in {"install", "update_metadata"}
            else "read"
        )
    elif tool == "desktop_session":
        action = str(args.get("operation") or "capabilities").strip().lower()
        target = _optional_string(args.get("title")) or "desktop-session"
        risk_class = "read" if action == "capabilities" else "system_impact"
    elif tool == "desktop_open":
        action, target, risk_class = (
            "open",
            _optional_string(args.get("target")),
            "external_side_effect",
        )
    elif tool == "desktop_browser":
        action = str(args.get("operation") or "snapshot").strip().lower()
        target = (
            _optional_string(args.get("url"))
            or _optional_string(args.get("page_id"))
            or _optional_string(args.get("selector"))
        )
        if action == "download":
            destination = _optional_string(args.get("destination"))
            if destination is not None:
                actions.append(
                    DesktopAction(
                        tool=tool,
                        action="download_destination",
                        target=destination,
                        risk_class=("overwrite" if args.get("overwrite") is True else "write"),
                    )
                )
        risk_class = (
            "network"
            if action in {"open", "new_tab", "navigate"}
            else "external_side_effect"
            if action in {"click", "input", "select", "submit", "download", "close_tab"}
            else "read"
        )
    elif tool == "desktop_gui":
        action = str(args.get("operation") or "observe").strip().lower()
        target = _optional_string(args.get("accessible_path")) or _optional_string(
            args.get("window_id")
        )
        if target is None and action == "click":
            target = f"screen:{args.get('x')}:{args.get('y')}"
        risk_class = (
            "sensitive_read"
            if action in {"windows", "active_window", "observe", "screenshot"}
            else "ui_interaction"
        )
    elif tool == "desktop_verify":
        action, target, risk_class = "verify", _optional_string(args.get("path")), "read"
    else:
        return []
    actions.append(
        DesktopAction(
            tool=tool,
            action=action,
            target=target,
            command_class=command_class,
            command=command,
            risk_class=risk_class,
        )
    )
    return actions


def desktop_action_from_request(request: ToolRequest) -> DesktopAction | None:
    actions = desktop_actions_from_request(request)
    return actions[-1] if actions else None


def classify_command(argv_raw: object) -> str:
    if not isinstance(argv_raw, list) or not argv_raw:
        return "malformed"
    first = argv_raw[0]
    if not isinstance(first, str) or not first.strip():
        return "malformed"
    executable = Path(first.strip()).name.lower()
    if executable in {"sudo", "su", "pkexec", "doas"}:
        return "privilege_escalation"
    if executable in {"mkfs", "fdisk", "parted", "dd", "grub-install", "update-grub"}:
        return "disk_boot"
    if executable in {"sh", "bash", "dash", "zsh", "fish"}:
        return "shell_indirection"
    if executable in {
        "env",
        "xargs",
        "nohup",
        "command",
        "timeout",
        "nice",
        "ionice",
        "chrt",
        "setsid",
        "stdbuf",
        "busybox",
        "systemd-run",
    }:
        return "shell_indirection"
    if executable in {"python", "python3"} and any(item in {"-c", "-m"} for item in argv_raw[1:]):
        return "shell_indirection"
    if executable in {"node", "perl", "ruby"} and any(
        item in {"-e", "--eval"} for item in argv_raw[1:]
    ):
        return "shell_indirection"
    if executable == "find" and any(item in {"-exec", "-execdir"} for item in argv_raw[1:]):
        return "shell_indirection"
    if executable in {"apt", "apt-get", "dnf", "yum", "pacman", "snap", "flatpak"}:
        return "package_management"
    if executable in {"systemctl", "service", "journalctl"}:
        return "service_management"
    if executable in {"kill", "pkill", "killall"}:
        return "process_management"
    if executable in {"curl", "wget", "ssh", "scp", "rsync"}:
        return "network"
    if executable == "git":
        subcommand = next(
            (str(item).lower() for item in argv_raw[1:] if isinstance(item, str) and item),
            "",
        )
        if subcommand in {"status", "log", "diff", "show", "branch", "rev-parse"}:
            return "read_only"
        return "filesystem_mutation"
    if executable in {"unzip"}:
        return "filesystem_mutation"
    if executable == "tar" and any(
        isinstance(item, str) and ("x" in item.lstrip("-") or item == "--extract")
        for item in argv_raw[1:]
    ):
        return "filesystem_mutation"
    if executable == "sed" and any(
        isinstance(item, str) and (item == "-i" or item.startswith("--in-place"))
        for item in argv_raw[1:]
    ):
        return "filesystem_mutation"
    if executable in {
        "rm",
        "mv",
        "cp",
        "install",
        "chmod",
        "chown",
        "ln",
        "truncate",
        "tee",
        "touch",
        "mkdir",
        "rmdir",
    }:
        return "filesystem_mutation"
    if executable in {"python", "python3", "pytest", "node", "npm", "npx", "make"}:
        return "project_execution"
    if executable in {
        "pwd",
        "ls",
        "find",
        "stat",
        "file",
        "du",
        "df",
        "grep",
        "rg",
        "head",
        "tail",
        "wc",
        "uname",
        "whoami",
        "id",
        "ps",
        "which",
        "realpath",
    }:
        return "read_only"
    return "unknown"


def command_risk_class(command_class: str) -> str:
    mapping = {
        "privilege_escalation": "privileged",
        "disk_boot": "destructive",
        "shell_indirection": "indirect",
        "package_management": "install",
        "service_management": "system",
        "process_management": "system",
        "network": "network",
        "project_execution": "arbitrary_code",
        "read_only": "read",
        "malformed": "malformed",
        "unknown": "unknown",
        "filesystem_mutation": "write",
    }
    return mapping.get(command_class, "unknown")


def _command_string(argv_raw: object) -> str | None:
    if not isinstance(argv_raw, list) or not argv_raw:
        return None
    if not all(isinstance(item, str) and item.strip() for item in argv_raw):
        return None
    return shlex.join([str(item) for item in argv_raw])


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _validate_persistent_rule(rule: DesktopApprovalRule) -> None:
    if rule.scope.tool == "desktop_cleanup_unverified_launches":
        raise ValueError("Runtime cleanup policy недоступна для persistent Desktop rules")
    if rule.effect == "allow" and not rule.scope.supports_persistent_allow:
        raise ValueError("Persistent allow запрещён для unrestricted or sensitive Desktop scopes")
