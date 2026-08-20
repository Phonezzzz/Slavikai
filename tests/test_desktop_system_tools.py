from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import tools.desktop_system_tools as system_tools_module
from core.desktop_security import DesktopPathSecurity
from shared.models import ToolRequest
from tools.desktop_system_tools import (
    DesktopClipboardTool,
    DesktopPackageTool,
    DesktopProcessTool,
    DesktopSessionTool,
    DesktopSystemdTool,
    DesktopSystemInfoTool,
)
from tools.desktop_tools import DesktopLaunchTool, DesktopShellTool


class MemoryClipboard:
    name = "memory-test"

    def __init__(self) -> None:
        self.text = ""

    def read(self) -> str:
        return self.text

    def write(self, text: str) -> None:
        self.text = text

    def clear(self) -> None:
        self.text = ""


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop-approvals.json",
    )


def test_clipboard_write_read_clear_are_verified() -> None:
    backend = MemoryClipboard()
    tool = DesktopClipboardTool(backend)

    written = tool.handle(
        ToolRequest("desktop_clipboard", {"operation": "write", "text": "desktop text"})
    )
    read = tool.handle(ToolRequest("desktop_clipboard", {"operation": "read"}))
    cleared = tool.handle(ToolRequest("desktop_clipboard", {"operation": "clear"}))

    assert written.ok and written.data["verified"] is True
    assert read.ok and read.data["text"] == "desktop text"
    assert cleared.ok and cleared.data["verified"] is True
    assert backend.text == ""


def test_real_system_information_is_structured_and_paginated(monkeypatch) -> None:
    monkeypatch.setattr(
        system_tools_module.psutil,
        "net_if_stats",
        lambda: {"test0": SimpleNamespace(isup=True, speed=1000)},
    )
    monkeypatch.setattr(
        system_tools_module.psutil,
        "net_if_addrs",
        lambda: {
            "test0": [
                SimpleNamespace(
                    family="AF_INET",
                    address="192.0.2.10",
                    netmask="255.255.255.0",
                )
            ]
        },
    )
    tool = DesktopSystemInfoTool()

    summary = tool.handle(ToolRequest("desktop_system_info", {"operation": "summary"}))
    os_info = tool.handle(ToolRequest("desktop_system_info", {"operation": "os"}))
    cpu = tool.handle(ToolRequest("desktop_system_info", {"operation": "cpu"}))
    memory = tool.handle(ToolRequest("desktop_system_info", {"operation": "memory"}))
    disks = tool.handle(ToolRequest("desktop_system_info", {"operation": "disks"}))
    network = tool.handle(ToolRequest("desktop_system_info", {"operation": "network"}))
    processes = tool.handle(
        ToolRequest("desktop_system_info", {"operation": "processes", "limit": 1})
    )
    session = tool.handle(ToolRequest("desktop_system_info", {"operation": "session"}))
    unknown = tool.handle(ToolRequest("desktop_system_info", {"operation": "unknown"}))

    assert summary.ok
    assert isinstance(summary.data["os"], dict)
    assert isinstance(summary.data["cpu"], dict)
    assert isinstance(summary.data["memory"], dict)
    assert os_info.ok and isinstance(os_info.data["kernel"], str)
    assert cpu.ok and isinstance(cpu.data["logical_cores"], int)
    assert memory.ok and isinstance(memory.data["total_bytes"], int)
    assert disks.ok and isinstance(disks.data["items"], list)
    assert network.ok and isinstance(network.data["items"], list)
    assert processes.ok and len(processes.data["items"]) == 1
    assert isinstance(processes.data["truncated"], bool)
    assert processes.data["next_offset"] in {None, 1}
    assert session.ok and isinstance(session.data["environment"], dict)
    assert not unknown.ok


def test_real_process_launch_inspect_verify_and_terminate(tmp_path: Path) -> None:
    security = _security(tmp_path)
    tool = DesktopProcessTool(
        security,
        launcher=DesktopLaunchTool(security),
    )
    launched = tool.handle(
        ToolRequest(
            "desktop_process",
            {"operation": "launch", "argv": ["/usr/bin/sleep", "30"], "cwd": str(tmp_path)},
        )
    )
    assert launched.ok
    pid = launched.data["pid"]
    create_time = launched.data["create_time"]
    assert isinstance(pid, int)
    assert isinstance(create_time, float)
    try:
        inspected = tool.handle(
            ToolRequest("desktop_process", {"operation": "inspect", "pid": pid})
        )
        verified = tool.handle(
            ToolRequest(
                "desktop_process",
                {
                    "operation": "status",
                    "pid": pid,
                    "expected_create_time": create_time,
                },
            )
        )
        stale = tool.handle(
            ToolRequest(
                "desktop_process",
                {
                    "operation": "terminate",
                    "pid": pid,
                    "expected_create_time": create_time + 100,
                },
            )
        )
        assert inspected.ok and inspected.data["running"] is True
        assert verified.ok
        running_after_stale = tool.handle(
            ToolRequest("desktop_process", {"operation": "status", "pid": pid})
        )
        assert not stale.ok
        assert running_after_stale.ok and running_after_stale.data["running"] is True

        terminated = tool.handle(
            ToolRequest(
                "desktop_process",
                {
                    "operation": "terminate",
                    "pid": pid,
                    "expected_create_time": create_time,
                },
            )
        )
        assert terminated.ok and terminated.data["verified"] is True
        status = tool.handle(
            ToolRequest(
                "desktop_process",
                {
                    "operation": "status",
                    "pid": pid,
                    "expected_create_time": create_time,
                },
            )
        )
        assert status.ok and status.data["running"] is False
    finally:
        final_status = tool.handle(
            ToolRequest("desktop_process", {"operation": "status", "pid": pid})
        )
        if final_status.data.get("running") is True:
            tool.handle(
                ToolRequest(
                    "desktop_process",
                    {
                        "operation": "kill",
                        "pid": pid,
                        "expected_create_time": create_time,
                    },
                )
            )


def test_generic_shell_refuses_typed_service_package_and_process_actions(tmp_path: Path) -> None:
    tool = DesktopShellTool(_security(tmp_path))

    for argv in (["systemctl", "status", "ssh"], ["apt-get", "install", "x"], ["kill", "1"]):
        result = tool.handle(ToolRequest("desktop_shell", {"argv": argv, "cwd": str(tmp_path)}))
        assert not result.ok and "typed capability" in (result.error or "")


def test_systemd_state_change_uses_fixed_argv_and_verifies() -> None:
    active = False
    calls: list[list[str]] = []

    def runner(
        argv: Sequence[str],
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal active
        del timeout, env, input_text
        args = list(argv)
        calls.append(args)
        if "start" in args:
            active = True
            return subprocess.CompletedProcess(args, 0, "", "")
        if "show" in args:
            state = "active" if active else "inactive"
            output = (
                "Id=demo.service\nLoadState=loaded\n"
                f"ActiveState={state}\nSubState={'running' if active else 'dead'}\n"
                "UnitFileState=enabled\nMainPID=123\nResult=success\n"
            )
            return subprocess.CompletedProcess(args, 0, output, "")
        return subprocess.CompletedProcess(args, 1, "", "unexpected")

    result = DesktopSystemdTool(runner).handle(
        ToolRequest(
            "desktop_systemd",
            {"operation": "start", "scope": "user", "unit": "demo.service"},
        )
    )

    assert result.ok and result.data["verified"] is True
    assert calls[0] == ["systemctl", "--user", "start", "--", "demo.service"]
    assert calls[1][0:3] == ["systemctl", "--user", "show"]


def test_package_install_uses_typed_argv_and_verifies_installed_state() -> None:
    installed = False
    calls: list[list[str]] = []

    def runner(
        argv: Sequence[str],
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal installed
        del timeout, input_text
        args = list(argv)
        calls.append(args)
        if args[:3] == ["apt-get", "-y", "--no-install-recommends"]:
            assert env is not None and env["DEBIAN_FRONTEND"] == "noninteractive"
            installed = True
            return subprocess.CompletedProcess(args, 0, "installed", "")
        if args[0] == "dpkg-query":
            output = "ii \t1.0\n" if installed else ""
            return subprocess.CompletedProcess(args, 0 if installed else 1, output, "")
        if args[:2] == ["apt-cache", "policy"]:
            return subprocess.CompletedProcess(
                args,
                0,
                "Installed: 1.0\nCandidate: 1.0\n",
                "",
            )
        return subprocess.CompletedProcess(args, 1, "", "unexpected")

    result = DesktopPackageTool(runner).handle(
        ToolRequest("desktop_package", {"operation": "install", "package": "demo-package"})
    )

    assert result.ok and result.data["installed"] is True
    assert result.data["verified"] is True
    assert calls[0] == [
        "apt-get",
        "-y",
        "--no-install-recommends",
        "install",
        "demo-package",
    ]


def test_real_package_query_and_unavailable_desktop_session_are_honest() -> None:
    package = DesktopPackageTool().handle(
        ToolRequest("desktop_package", {"operation": "query", "package": "python3"})
    )
    capabilities = DesktopSessionTool().handle(
        ToolRequest("desktop_session", {"operation": "capabilities"})
    )

    assert package.ok and package.data["package"] == "python3"
    assert capabilities.ok
    assert isinstance(capabilities.data["display_available"], bool)
    assert capabilities.data["clipboard_backend"] is None or isinstance(
        capabilities.data["clipboard_backend"], str
    )
