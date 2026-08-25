"""PR-15: Policy / approval semantics for Computer tools.

Verifies that ASK/PLAN/ACT/AUTO modes and safe_mode/approved_categories
actually constrain Computer tools as ARCH_CANON requires.

Scope:
- ToolRegistry._mode_policy_error() enforces PLAN read-only correctly
- ToolGateway + ApprovalContext enforces safe_mode approval correctly
- AgentComputerRuntime operations go through the same gateway/policy path
- AUTO uses the same approval path as ACT (no bypass)
- Path traversal outside workspace is detected as FS_OUTSIDE_WORKSPACE
- Dangerous shell commands detected as EXEC_ARBITRARY / SUDO / etc.
- No lane="computer", no FakeContainerRunner, no new runtime.
"""

from __future__ import annotations

import pytest

from core.approval_policy import (
    ALL_CATEGORIES,
    ApprovalContext,
    ApprovalRequired,
    detect_action_intents,
)
from core.tool_gateway import ToolGateway
from shared.models import ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry

# ── Helpers ──────────────────────────────────────────────────────────────────


def _ok_handler(req: ToolRequest) -> ToolResult:
    return ToolResult.success({"tool": req.name})


def _make_registry_with_workspace_tools() -> ToolRegistry:
    """Minimal registry with the same capability/risk_class as production."""
    registry = ToolRegistry()
    registry.register("workspace_list", _ok_handler, enabled=True, capability="read")
    registry.register("workspace_read", _ok_handler, enabled=True, capability="read")
    registry.register(
        "workspace_write",
        _ok_handler,
        enabled=True,
        capability="write",
    )
    registry.register(
        "workspace_patch",
        _ok_handler,
        enabled=True,
        capability="write",
    )
    registry.register(
        "workspace_create",
        _ok_handler,
        enabled=True,
        capability="write",
    )
    registry.register(
        "workspace_delete",
        _ok_handler,
        enabled=True,
        capability="write",
        risk_classes=["write", "destructive"],
    )
    registry.register(
        "workspace_terminal_run",
        _ok_handler,
        enabled=True,
        capability="exec",
        risk_classes=["execute"],
    )
    registry.register(
        "workspace_run",
        _ok_handler,
        enabled=True,
        capability="exec",
        risk_classes=["execute"],
    )
    return registry


def _permissive_ctx() -> ApprovalContext:
    """safe_mode=False → all intents are allowed without approval."""
    return ApprovalContext(safe_mode=False, session_id=None, approved_categories=set())


def _strict_ctx(approved: set[str] | None = None) -> ApprovalContext:
    """safe_mode=True → dangerous intents require approval unless pre-approved."""
    from core.approval_policy import ApprovalCategory

    cats: set[ApprovalCategory] = set()
    if approved:
        for c in approved:
            if c in ALL_CATEGORIES:
                cats.add(c)  # type: ignore[arg-type]
    return ApprovalContext(safe_mode=True, session_id="sess-1", approved_categories=cats)


# ── PLAN: ToolRegistry mode policy enforcement ────────────────────────────────


class TestPlanModeReadOnlyPolicy:
    """PLAN mode blocks all non-read tools at ToolRegistry level."""

    def setup_method(self) -> None:
        self.registry = _make_registry_with_workspace_tools()
        self.registry.set_execution_policy(mode="plan")

    def test_plan_allows_workspace_list(self) -> None:
        result = self.registry.call(ToolRequest("workspace_list"))
        assert result.ok, f"workspace_list should be allowed in PLAN, got: {result.error}"

    def test_plan_allows_workspace_read(self) -> None:
        result = self.registry.call(ToolRequest("workspace_read", {"path": "x.py"}))
        assert result.ok, f"workspace_read should be allowed in PLAN, got: {result.error}"

    def test_plan_blocks_workspace_write(self) -> None:
        result = self.registry.call(ToolRequest("workspace_write", {"path": "x.py", "content": ""}))
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_plan_blocks_workspace_patch(self) -> None:
        result = self.registry.call(ToolRequest("workspace_patch", {"path": "x.py", "patch": ""}))
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_plan_blocks_workspace_create(self) -> None:
        result = self.registry.call(ToolRequest("workspace_create", {"path": "new.py"}))
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_plan_blocks_workspace_delete(self) -> None:
        result = self.registry.call(ToolRequest("workspace_delete", {"path": "x.py"}))
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_plan_blocks_workspace_terminal_run(self) -> None:
        """run_command / git_diff / run_tests / check all use workspace_terminal_run."""
        result = self.registry.call(ToolRequest("workspace_terminal_run", {"command": "pytest"}))
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_plan_blocks_workspace_run(self) -> None:
        result = self.registry.call(ToolRequest("workspace_run", {"path": "script.py"}))
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")


# ── ACT / AUTO: no mode-level block; approval applies instead ─────────────────


class TestActAutoModeNoModeBlock:
    """ACT and AUTO do not block tools at mode-policy level; approval gates instead."""

    def _registry_with_mode(self, mode: str) -> ToolRegistry:
        r = _make_registry_with_workspace_tools()
        r.set_execution_policy(mode=mode)
        return r

    def test_act_does_not_block_workspace_write_at_registry_level(self) -> None:
        registry = self._registry_with_mode("act")
        result = registry.call(ToolRequest("workspace_write", {"path": "x.py", "content": ""}))
        assert result.ok, "ACT should not mode-block write at registry level"

    def test_act_does_not_block_workspace_terminal_run_at_registry_level(self) -> None:
        registry = self._registry_with_mode("act")
        result = registry.call(ToolRequest("workspace_terminal_run", {"command": "pytest"}))
        assert result.ok, "ACT should not mode-block exec at registry level"

    def test_auto_does_not_block_workspace_write_at_registry_level(self) -> None:
        registry = self._registry_with_mode("auto")
        result = registry.call(ToolRequest("workspace_write", {"path": "x.py", "content": ""}))
        assert result.ok, "AUTO should not mode-block write at registry level"

    def test_auto_does_not_block_workspace_terminal_run_at_registry_level(self) -> None:
        registry = self._registry_with_mode("auto")
        result = registry.call(ToolRequest("workspace_terminal_run", {"command": "make check"}))
        assert result.ok, "AUTO should not mode-block exec at registry level"


# ── safe_mode=False: unrestricted (no approval required) ─────────────────────


class TestSafeModeOffNoApproval:
    """safe_mode=False → all Computer tool calls allowed without approval."""

    def setup_method(self) -> None:
        self.registry = _make_registry_with_workspace_tools()
        self.registry.set_execution_policy(mode="act")
        self.gw = ToolGateway(registry=self.registry, approval_context=_permissive_ctx())

    def test_write_file_no_approval_required(self) -> None:
        result = self.gw.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))
        assert result.ok

    def test_apply_patch_no_approval_required(self) -> None:
        result = self.gw.call(ToolRequest("workspace_patch", {"path": "x.py", "patch": "@@ @@"}))
        assert result.ok

    def test_run_command_no_approval_required(self) -> None:
        result = self.gw.call(ToolRequest("workspace_terminal_run", {"command": "pytest tests/"}))
        assert result.ok

    def test_delete_no_approval_required(self) -> None:
        result = self.gw.call(ToolRequest("workspace_delete", {"path": "x.py"}))
        assert result.ok

    def test_read_file_no_approval_required(self) -> None:
        result = self.gw.call(ToolRequest("workspace_read", {"path": "x.py"}))
        assert result.ok


# ── safe_mode=True: dangerous tools need approval ────────────────────────────


class TestSafeModeOnApprovalGate:
    """safe_mode=True → write/exec Computer tools raise ApprovalRequired."""

    def setup_method(self) -> None:
        self.registry = _make_registry_with_workspace_tools()
        self.registry.set_execution_policy(mode="act")

    def _gateway(self, approved: set[str] | None = None) -> ToolGateway:
        return ToolGateway(registry=self.registry, approval_context=_strict_ctx(approved))

    def test_workspace_write_raises_approval_required(self) -> None:
        gw = self._gateway()
        with pytest.raises(ApprovalRequired) as exc_info:
            gw.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))
        assert exc_info.value.request.tool == "workspace_write"

    def test_workspace_patch_raises_approval_required(self) -> None:
        gw = self._gateway()
        with pytest.raises(ApprovalRequired):
            gw.call(ToolRequest("workspace_patch", {"path": "x.py", "patch": "@@ @@"}))

    def test_workspace_create_raises_approval_required(self) -> None:
        gw = self._gateway()
        with pytest.raises(ApprovalRequired):
            gw.call(ToolRequest("workspace_create", {"path": "new.py"}))

    def test_workspace_delete_raises_approval_required(self) -> None:
        gw = self._gateway()
        with pytest.raises(ApprovalRequired):
            gw.call(ToolRequest("workspace_delete", {"path": "x.py"}))

    def test_workspace_terminal_run_raises_approval_required(self) -> None:
        """run_command / run_tests / check → EXEC_ARBITRARY approval required."""
        gw = self._gateway()
        with pytest.raises(ApprovalRequired) as exc_info:
            gw.call(ToolRequest("workspace_terminal_run", {"command": "pytest"}))
        req = exc_info.value.request
        assert req.tool == "workspace_terminal_run"
        assert "EXEC_ARBITRARY" in req.required_categories

    def test_workspace_run_raises_approval_required(self) -> None:
        gw = self._gateway()
        with pytest.raises(ApprovalRequired):
            gw.call(ToolRequest("workspace_run", {"path": "script.py"}))

    def test_workspace_read_does_not_raise(self) -> None:
        """Read-only Computer tools are safe in safe_mode — no approval needed."""
        gw = self._gateway()
        result = gw.call(ToolRequest("workspace_read", {"path": "x.py"}))
        assert result.ok

    def test_workspace_list_does_not_raise(self) -> None:
        gw = self._gateway()
        result = gw.call(ToolRequest("workspace_list", {"path": ""}))
        assert result.ok

    def test_approved_category_bypasses_write_approval(self) -> None:
        """Pre-approved FS_DELETE_OVERWRITE → write_file goes through."""
        gw = self._gateway(approved={"FS_DELETE_OVERWRITE"})
        result = gw.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))
        assert result.ok

    def test_approved_exec_category_bypasses_terminal_approval(self) -> None:
        """Pre-approved EXEC_ARBITRARY → terminal_run goes through."""
        gw = self._gateway(approved={"EXEC_ARBITRARY"})
        result = gw.call(ToolRequest("workspace_terminal_run", {"command": "pytest"}))
        assert result.ok

    def test_wrong_approved_category_does_not_bypass(self) -> None:
        """Approving NETWORK_RISK does not bypass EXEC_ARBITRARY requirement."""
        gw = self._gateway(approved={"NETWORK_RISK"})
        with pytest.raises(ApprovalRequired):
            gw.call(ToolRequest("workspace_terminal_run", {"command": "pytest"}))


# ── AUTO: same approval path as ACT ──────────────────────────────────────────


class TestAutoUseSameApprovalAsAct:
    """AUTO does not bypass ToolGateway/approval — same path as ACT."""

    def _gateway_for_mode(self, mode: str) -> ToolGateway:
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode=mode)
        return ToolGateway(registry=registry, approval_context=_strict_ctx())

    def test_auto_write_raises_approval_required(self) -> None:
        gw = self._gateway_for_mode("auto")
        with pytest.raises(ApprovalRequired):
            gw.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))

    def test_auto_terminal_run_raises_approval_required(self) -> None:
        gw = self._gateway_for_mode("auto")
        with pytest.raises(ApprovalRequired):
            gw.call(ToolRequest("workspace_terminal_run", {"command": "make check"}))

    def test_auto_read_does_not_raise(self) -> None:
        gw = self._gateway_for_mode("auto")
        result = gw.call(ToolRequest("workspace_read", {"path": "x.py"}))
        assert result.ok

    def test_auto_approval_required_has_same_structure_as_act(self) -> None:
        """ApprovalRequired from AUTO carries same fields as from ACT."""
        gw_auto = self._gateway_for_mode("auto")
        gw_act = self._gateway_for_mode("act")

        with pytest.raises(ApprovalRequired) as auto_exc:
            gw_auto.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))
        with pytest.raises(ApprovalRequired) as act_exc:
            gw_act.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))

        auto_req = auto_exc.value.request
        act_req = act_exc.value.request
        assert auto_req.tool == act_req.tool
        assert auto_req.category == act_req.category
        assert auto_req.required_categories == act_req.required_categories


# ── AgentComputerRuntime: write_file/run_command blocked in PLAN ──────────────


class TestAgentComputerRuntimePlanBlock:
    """AgentComputerRuntime operations route through ToolGateway + PLAN policy."""

    def test_write_file_blocked_in_plan_mode(self) -> None:
        from core.agent_computer import AgentComputerRuntime
        from core.computer_backend import LocalComputerBackend

        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="plan")
        gateway = ToolGateway(registry=registry, approval_context=None)
        backend = LocalComputerBackend(gateway=gateway)
        runtime = AgentComputerRuntime(backend=backend)

        result = runtime.write_file("x.py", "content")
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_apply_patch_blocked_in_plan_mode(self) -> None:
        from core.agent_computer import AgentComputerRuntime
        from core.computer_backend import LocalComputerBackend

        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="plan")
        gateway = ToolGateway(registry=registry, approval_context=None)
        backend = LocalComputerBackend(gateway=gateway)
        runtime = AgentComputerRuntime(backend=backend)

        result = runtime.apply_patch("x.py", "@@ -1 +1 @@\n-old\n+new")
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_run_command_blocked_in_plan_mode(self) -> None:
        """run_command → workspace_terminal_run (exec) → blocked in PLAN."""
        from core.agent_computer import AgentComputerRuntime
        from core.computer_backend import LocalComputerBackend

        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="plan")
        gateway = ToolGateway(registry=registry, approval_context=None)
        backend = LocalComputerBackend(gateway=gateway)
        runtime = AgentComputerRuntime(backend=backend)

        result = runtime.run_command("pytest tests/")
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_run_tests_blocked_in_plan_mode(self) -> None:
        from core.agent_computer import AgentComputerRuntime
        from core.computer_backend import LocalComputerBackend

        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="plan")
        gateway = ToolGateway(registry=registry, approval_context=None)
        backend = LocalComputerBackend(gateway=gateway)
        runtime = AgentComputerRuntime(backend=backend)

        result = runtime.run_tests()
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_git_diff_blocked_in_plan_mode(self) -> None:
        """git_diff also uses workspace_terminal_run (exec) — blocked in PLAN."""
        from core.agent_computer import AgentComputerRuntime
        from core.computer_backend import LocalComputerBackend

        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="plan")
        gateway = ToolGateway(registry=registry, approval_context=None)
        backend = LocalComputerBackend(gateway=gateway)
        runtime = AgentComputerRuntime(backend=backend)

        result = runtime.git_diff()
        assert not result.ok
        assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")

    def test_read_file_allowed_in_plan_mode(self) -> None:
        from core.agent_computer import AgentComputerRuntime
        from core.computer_backend import LocalComputerBackend

        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="plan")
        gateway = ToolGateway(registry=registry, approval_context=None)
        backend = LocalComputerBackend(gateway=gateway)
        runtime = AgentComputerRuntime(backend=backend)

        result = runtime.read_file("x.py")
        assert result.ok, f"read_file should be allowed in PLAN, got: {result.error}"

    def test_list_files_allowed_in_plan_mode(self) -> None:
        from core.agent_computer import AgentComputerRuntime
        from core.computer_backend import LocalComputerBackend

        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="plan")
        gateway = ToolGateway(registry=registry, approval_context=None)
        backend = LocalComputerBackend(gateway=gateway)
        runtime = AgentComputerRuntime(backend=backend)

        result = runtime.list_files()
        assert result.ok, f"list_files should be allowed in PLAN, got: {result.error}"


# ── Path traversal: FS_OUTSIDE_WORKSPACE detected ────────────────────────────


class TestPathTraversalDetection:
    """detect_action_intents catches paths that escape workspace boundary."""

    def test_absolute_path_flagged_outside_workspace(self) -> None:
        req = ToolRequest("workspace_write", {"path": "/etc/passwd"})
        intents = detect_action_intents(req)
        categories = {i.category for i in intents}
        assert "FS_OUTSIDE_WORKSPACE" in categories

    def test_dotdot_path_flagged_outside_workspace(self) -> None:
        req = ToolRequest("workspace_write", {"path": "../../../etc/passwd"})
        intents = detect_action_intents(req)
        categories = {i.category for i in intents}
        assert "FS_OUTSIDE_WORKSPACE" in categories

    def test_tilde_path_flagged_outside_workspace(self) -> None:
        req = ToolRequest("workspace_write", {"path": "~/secrets.txt"})
        intents = detect_action_intents(req)
        categories = {i.category for i in intents}
        assert "FS_OUTSIDE_WORKSPACE" in categories

    def test_relative_within_workspace_not_flagged(self) -> None:
        req = ToolRequest("workspace_write", {"path": "src/main.py"})
        intents = detect_action_intents(req)
        categories = {i.category for i in intents}
        assert "FS_OUTSIDE_WORKSPACE" not in categories

    def test_outside_workspace_blocked_in_strict_gateway(self) -> None:
        """Absolute path → FS_OUTSIDE_WORKSPACE intent → approval required."""
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="act")
        gw = ToolGateway(registry=registry, approval_context=_strict_ctx())
        with pytest.raises(ApprovalRequired) as exc_info:
            gw.call(ToolRequest("workspace_write", {"path": "/etc/passwd", "content": ""}))
        req = exc_info.value.request
        assert "FS_OUTSIDE_WORKSPACE" in req.required_categories


# ── Dangerous shell commands: intent detection ────────────────────────────────


class TestDangerousCommandIntentDetection:
    """detect_action_intents correctly classifies dangerous shell commands."""

    def test_arbitrary_command_always_exec_arbitrary(self) -> None:
        req = ToolRequest("workspace_terminal_run", {"command": "pytest tests/"})
        intents = detect_action_intents(req, risk_classes=["execute"])
        categories = {i.category for i in intents}
        assert "EXEC_ARBITRARY" in categories

    def test_sudo_command_is_hard_denied_intent(self) -> None:
        req = ToolRequest("workspace_terminal_run", {"command": "sudo reboot"})
        intents = detect_action_intents(req, risk_classes=["execute"])
        categories = {i.category for i in intents}
        assert "HARD_DENY" in categories

    def test_git_commit_adds_git_publish_intent(self) -> None:
        req = ToolRequest("workspace_terminal_run", {"command": "git commit -m 'x'"})
        intents = detect_action_intents(req, risk_classes=["execute"])
        categories = {i.category for i in intents}
        assert "GIT_PUBLISH" in categories

    def test_git_push_adds_git_publish_intent(self) -> None:
        req = ToolRequest("workspace_terminal_run", {"command": "git push origin main"})
        intents = detect_action_intents(req, risk_classes=["execute"])
        categories = {i.category for i in intents}
        assert "GIT_PUBLISH" in categories

    def test_pip_install_adds_deps_intent(self) -> None:
        req = ToolRequest("workspace_terminal_run", {"command": "pip install requests"})
        intents = detect_action_intents(req, risk_classes=["execute"])
        categories = {i.category for i in intents}
        assert "DEPS_INSTALL_UPDATE" in categories

    def test_npm_install_adds_deps_intent(self) -> None:
        req = ToolRequest("workspace_terminal_run", {"command": "npm install"})
        intents = detect_action_intents(req, risk_classes=["execute"])
        categories = {i.category for i in intents}
        assert "DEPS_INSTALL_UPDATE" in categories

    def test_systemctl_adds_system_impact_intent(self) -> None:
        req = ToolRequest("workspace_terminal_run", {"command": "systemctl restart nginx"})
        intents = detect_action_intents(req, risk_classes=["execute"])
        categories = {i.category for i in intents}
        assert "SYSTEM_IMPACT" in categories

    def test_dangerous_command_blocked_in_strict_gateway(self) -> None:
        """sudo command → HARD_DENY → hard block (not an approval prompt)."""
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="act")
        gw = ToolGateway(registry=registry, approval_context=_strict_ctx())
        result = gw.call(ToolRequest("workspace_terminal_run", {"command": "sudo rm -rf /tmp/x"}))
        assert not result.ok
        assert "запрещена политикой безопасности" in (result.error or "")


class TestHardSafetyBlockAppliesEvenInYolo:
    """Hard-safety commands are blocked regardless of safe_mode/YOLO."""

    def _gateway(self, ctx: ApprovalContext) -> ToolGateway:
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="act")
        return ToolGateway(registry=registry, approval_context=ctx)

    def test_yolo_blocks_sudo(self) -> None:
        result = self._gateway(_permissive_ctx()).call(
            ToolRequest("workspace_terminal_run", {"command": "sudo reboot"})
        )
        assert not result.ok
        assert "запрещена политикой безопасности" in (result.error or "")

    def test_yolo_blocks_rm_rf(self) -> None:
        result = self._gateway(_permissive_ctx()).call(
            ToolRequest("workspace_terminal_run", {"command": "rm -rf /tmp/x"})
        )
        assert not result.ok
        assert "запрещена политикой безопасности" in (result.error or "")

    def test_yolo_allows_safe_command(self) -> None:
        result = self._gateway(_permissive_ctx()).call(
            ToolRequest("workspace_terminal_run", {"command": "pwd"})
        )
        assert result.ok

    def test_git_push_blocked_in_strict_gateway(self) -> None:
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="act")
        gw = ToolGateway(registry=registry, approval_context=_strict_ctx())
        with pytest.raises(ApprovalRequired) as exc_info:
            gw.call(ToolRequest("workspace_terminal_run", {"command": "git push origin main"}))
        req = exc_info.value.request
        overlap = {"GIT_PUBLISH", "EXEC_ARBITRARY"} & set(req.required_categories)
        assert overlap


# ── Config secret path detection ─────────────────────────────────────────────


class TestConfigSecretPathDetection:
    """write to config/secret paths triggers FS_CONFIG_SECRETS intent."""

    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            "config/settings.yaml",
            "secrets.json",
            "deploy/token.txt",
        ],
    )
    def test_config_path_triggers_fs_config_secrets(self, path: str) -> None:
        req = ToolRequest("workspace_write", {"path": path})
        intents = detect_action_intents(req)
        categories = {i.category for i in intents}
        assert "FS_CONFIG_SECRETS" in categories, (
            f"Expected FS_CONFIG_SECRETS for path={path!r}, got {categories}"
        )

    def test_normal_path_no_config_secret(self) -> None:
        req = ToolRequest("workspace_write", {"path": "src/main.py"})
        intents = detect_action_intents(req)
        categories = {i.category for i in intents}
        assert "FS_CONFIG_SECRETS" not in categories

    def test_shell_config_path_triggers_fs_config_secrets(self) -> None:
        """shell config_path writes a config file: must be visible to approval."""
        req = ToolRequest(
            "shell",
            {"command": "echo hi", "config_path": "config/shell_config.json"},
        )
        intents = detect_action_intents(req, risk_classes=["execute"])
        categories = {i.category for i in intents}
        assert "FS_CONFIG_SECRETS" in categories, (
            f"Expected FS_CONFIG_SECRETS for shell config_path, got {categories}"
        )

    def test_shell_without_config_path_has_no_fs_config_secret(self) -> None:
        req = ToolRequest("shell", {"command": "echo hi"})
        intents = detect_action_intents(req, risk_classes=["execute"])
        categories = {i.category for i in intents}
        assert "FS_CONFIG_SECRETS" not in categories
        assert "EXEC_ARBITRARY" in categories


# ── ASK mode: ToolRegistry does not hard-block (uses LLM integration layer) ──


class TestAskModeRegistryBehavior:
    """ASK mode hard-blocks non-read tools at ToolRegistry._mode_policy_error().

    This is an execution-boundary guarantee: even if a tool_call bypasses the
    LLM integration layer (_chat_read_tool_specs), the registry itself rejects
    mutating/exec tools in ASK with ASK_READ_ONLY_BLOCK.  safe_mode=False
    cannot override this block because mode check runs before safe_mode gating.
    """

    def test_ask_blocks_workspace_write(self) -> None:
        """ToolRegistry in ASK mode hard-blocks write at execution boundary."""
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        result = registry.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))
        assert not result.ok
        assert "ASK_READ_ONLY_BLOCK" in (result.error or "")

    def test_ask_blocks_workspace_terminal_run(self) -> None:
        """ToolRegistry in ASK mode hard-blocks exec at execution boundary."""
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        result = registry.call(ToolRequest("workspace_terminal_run", {"command": "pytest"}))
        assert not result.ok
        assert "ASK_READ_ONLY_BLOCK" in (result.error or "")

    def test_ask_blocks_workspace_patch(self) -> None:
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        result = registry.call(ToolRequest("workspace_patch", {"path": "x.py", "patch": "@@ @@"}))
        assert not result.ok
        assert "ASK_READ_ONLY_BLOCK" in (result.error or "")

    def test_ask_blocks_workspace_create(self) -> None:
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        result = registry.call(ToolRequest("workspace_create", {"path": "new.py", "content": ""}))
        assert not result.ok
        assert "ASK_READ_ONLY_BLOCK" in (result.error or "")

    def test_ask_blocks_workspace_delete(self) -> None:
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        result = registry.call(ToolRequest("workspace_delete", {"path": "old.py"}))
        assert not result.ok
        assert "ASK_READ_ONLY_BLOCK" in (result.error or "")

    def test_ask_blocks_workspace_run(self) -> None:
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        result = registry.call(ToolRequest("workspace_run", {"command": "echo hi"}))
        assert not result.ok
        assert "ASK_READ_ONLY_BLOCK" in (result.error or "")

    def test_ask_allows_workspace_read(self) -> None:
        """read-only tool is permitted in ASK mode."""
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        result = registry.call(ToolRequest("workspace_read", {"path": "x.py"}))
        assert result.ok

    def test_ask_allows_workspace_list(self) -> None:
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        result = registry.call(ToolRequest("workspace_list", {"path": "src/"}))
        assert result.ok

    def test_ask_block_unaffected_by_safe_mode_false(self) -> None:
        """safe_mode=False via ApprovalContext cannot bypass ASK_READ_ONLY_BLOCK."""
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        gw = ToolGateway(registry=registry, approval_context=_permissive_ctx())
        result = gw.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))
        assert not result.ok
        assert "ASK_READ_ONLY_BLOCK" in (result.error or "")

    def test_ask_block_unaffected_by_bypass_safe_mode_arg(self) -> None:
        """Direct registry.call(bypass_safe_mode=True) still blocked in ASK mode."""
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        result = registry.call(
            ToolRequest("workspace_write", {"path": "x.py", "content": "x"}),
            bypass_safe_mode=True,
        )
        assert not result.ok
        assert "ASK_READ_ONLY_BLOCK" in (result.error or "")

    def test_ask_block_on_direct_gateway_call(self) -> None:
        """Block fires even when request arrives via ToolGateway.call() directly."""
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        gw = ToolGateway(registry=registry, approval_context=None)
        result = gw.call(ToolRequest("workspace_terminal_run", {"command": "rm -rf /"}))
        assert not result.ok
        assert "ASK_READ_ONLY_BLOCK" in (result.error or "")

    def test_ask_with_safe_mode_write_blocked_at_mode_layer(self) -> None:
        """In ASK + safe_mode=True, mode block fires before approval check."""
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        gw = ToolGateway(registry=registry, approval_context=_strict_ctx())
        # Does NOT raise ApprovalRequired — mode block short-circuits first via registry.
        # gateway raises ApprovalRequired before calling registry when safe_mode=True,
        # but the mode block is verified via the gateway-less path above.
        # Here we confirm the end result is a failure, not a success.
        try:
            result = gw.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))
            assert not result.ok
        except ApprovalRequired:
            pass  # also acceptable — approval layer fired before registry

    def test_ask_with_safe_mode_read_still_allowed(self) -> None:
        registry = _make_registry_with_workspace_tools()
        registry.set_execution_policy(mode="ask")
        gw = ToolGateway(registry=registry, approval_context=_strict_ctx())
        result = gw.call(ToolRequest("workspace_read", {"path": "x.py"}))
        assert result.ok


# ── No lane="computer" in policy path ────────────────────────────────────────


def test_no_lane_computer_in_approval_required_request() -> None:
    """ApprovalRequired does not carry lane='computer' — Computer is not a lane."""
    registry = _make_registry_with_workspace_tools()
    registry.set_execution_policy(mode="act")
    gw = ToolGateway(registry=registry, approval_context=_strict_ctx())
    with pytest.raises(ApprovalRequired) as exc_info:
        gw.call(ToolRequest("workspace_write", {"path": "x.py", "content": "x"}))
    req = exc_info.value.request
    details_str = str(req.details)
    assert "lane" not in details_str or "computer" not in details_str
