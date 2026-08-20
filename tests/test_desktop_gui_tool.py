from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import tools.desktop_gui_tool as gui_module
from shared.models import ToolRequest
from tools.desktop_gui_tool import DesktopGuiTool


def test_headless_gui_capabilities_and_observation_fail_honestly(tmp_path: Path) -> None:
    tool = DesktopGuiTool(environ={}, artifact_root=tmp_path)

    capabilities = tool.handle(ToolRequest("desktop_gui", {"operation": "capabilities"}))
    observation = tool.handle(ToolRequest("desktop_gui", {"operation": "observe"}))

    assert capabilities.ok
    assert capabilities.data["display_available"] is False
    assert capabilities.data["screenshot_backend"] is None
    assert not observation.ok and "No screenshot backend" in (observation.error or "")


def test_visual_gui_action_observes_and_verifies_expected_post_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def available(command: str) -> str | None:
        return f"/usr/bin/{command}" if command in {"xdotool", "import", "tesseract"} else None

    def runner(argv: Sequence[str], timeout: int) -> subprocess.CompletedProcess[str]:
        del timeout
        args = list(argv)
        calls.append(args)
        if args[0] == "import":
            Path(args[-1]).write_bytes(b"not-empty-test-image")
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "tesseract":
            tsv = (
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\t"
                "height\tconf\ttext\n1\t1\t1\t1\t1\t1\t10\t10\t50\t20\t99\tSaved\n"
            )
            return subprocess.CompletedProcess(args, 0, tsv, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(gui_module.shutil, "which", available)
    tool = DesktopGuiTool(
        environ={"DISPLAY": ":99", "XDG_SESSION_TYPE": "x11"},
        artifact_root=tmp_path,
        runner=runner,
    )

    result = tool.handle(
        ToolRequest(
            "desktop_gui",
            {
                "operation": "click",
                "x": 25,
                "y": 35,
                "expected_text": "Saved",
            },
        )
    )

    assert result.ok and result.data["verified"] is True
    assert result.data["mechanism"] == "x11-xdotool"
    assert calls[0] == ["xdotool", "mousemove", "--sync", "25", "35", "click", "1"]
    screenshot = result.data["post_observation"]
    assert isinstance(screenshot, dict)
    assert Path(str(screenshot["screenshot_path"])).is_file()


def test_gui_action_does_not_claim_verification_without_expected_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def available(command: str) -> str | None:
        return f"/usr/bin/{command}" if command in {"xdotool", "import"} else None

    def runner(argv: Sequence[str], timeout: int) -> subprocess.CompletedProcess[str]:
        del timeout
        args = list(argv)
        if args[0] == "import":
            Path(args[-1]).write_bytes(b"not-empty-test-image")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(gui_module.shutil, "which", available)
    result = DesktopGuiTool(
        environ={"DISPLAY": ":99"},
        artifact_root=tmp_path,
        runner=runner,
    ).handle(ToolRequest("desktop_gui", {"operation": "click", "x": 1, "y": 2}))

    assert result.ok
    assert result.data.get("verified") is not True
    assert result.data["requires_followup_observation"] is True
