from __future__ import annotations

import shutil
from pathlib import Path

from shared.models import ToolRequest
from tools.workspace_tools import (
    WORKSPACE_ROOT,
    ApplyPatchTool,
    ListFilesTool,
    ReadFileTool,
    RunCodeTool,
    WriteFileTool,
)


def _make_request(name: str, args: dict[str, object]) -> ToolRequest:
    return ToolRequest(name=name, args=args)


def test_read_write_and_list_files() -> None:
    (WORKSPACE_ROOT / "nested").mkdir(parents=True, exist_ok=True)
    write = WriteFileTool()
    target_path = "nested/sample.txt"
    content = "line1\nline2\n"
    write_result = write.handle(
        _make_request("workspace_write", {"path": target_path, "content": content})
    )
    assert write_result.ok

    read = ReadFileTool().handle(_make_request("workspace_read", {"path": target_path}))
    assert read.ok
    assert read.data["output"] == content

    tree_result = ListFilesTool().handle(_make_request("workspace_list", {}))
    assert tree_result.ok
    paths = _flatten_paths(tree_result.data["tree"])
    assert target_path in paths


def test_apply_patch_dry_run_and_apply() -> None:
    file_path = "patch_me.txt"
    WriteFileTool().handle(
        _make_request("workspace_write", {"path": file_path, "content": "hello\nworld\n"})
    )
    patch_text = "@@ -1,2 +1,2 @@\n-hello\n+HELLO\n world\n"
    tool = ApplyPatchTool()

    dry = tool.handle(
        _make_request(
            "workspace_patch",
            {"path": file_path, "patch": patch_text, "dry_run": True},
        )
    )
    assert dry.ok
    assert dry.data["dry_run"] is True
    # Файл не меняется после dry-run
    still_original = ReadFileTool().handle(_make_request("workspace_read", {"path": file_path}))
    assert still_original.data["output"].startswith("hello")

    applied = tool.handle(
        _make_request(
            "workspace_patch",
            {"path": file_path, "patch": patch_text, "dry_run": False},
        )
    )
    assert applied.ok
    updated = ReadFileTool().handle(_make_request("workspace_read", {"path": file_path}))
    assert updated.data["output"].startswith("HELLO")


def test_apply_patch_rejects_escape() -> None:
    tool = ApplyPatchTool()
    result = tool.handle(
        _make_request(
            "workspace_patch",
            {"path": "../etc/passwd", "patch": "@@ -1,1 +1,1 @@\n-a\n+b\n"},
        )
    )
    assert not result.ok
    assert "рабочей" in (result.error or "").lower()


def test_apply_patch_rejects_multifile_contract() -> None:
    file_path = "single_file.txt"
    WriteFileTool().handle(
        _make_request("workspace_write", {"path": file_path, "content": "one\ntwo\n"})
    )
    multi_file_patch = (
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+A\n"
        "diff --git a/b.txt b/b.txt\n"
        "--- a/b.txt\n"
        "+++ b/b.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-b\n"
        "+B\n"
    )
    result = ApplyPatchTool().handle(
        _make_request(
            "workspace_patch",
            {"path": file_path, "patch": multi_file_patch},
        )
    )
    assert not result.ok
    assert "single-file" in (result.error or "").lower()


def test_run_code_success_and_timeout(tmp_path: Path) -> None:
    scripts_dir = WORKSPACE_ROOT / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    good_script = scripts_dir / "hello.py"
    good_script.write_text("print('hi')\n", encoding="utf-8")

    result_ok = RunCodeTool(timeout=2).handle(
        _make_request("workspace_run", {"path": str(good_script.relative_to(WORKSPACE_ROOT))})
    )
    assert result_ok.ok
    assert "hi" in str(result_ok.data.get("output"))

    slow_script = scripts_dir / "sleepy.py"
    slow_script.write_text("import time\n\ntime.sleep(5)\n", encoding="utf-8")
    result_timeout = RunCodeTool(timeout=1).handle(
        _make_request("workspace_run", {"path": str(slow_script.relative_to(WORKSPACE_ROOT))})
    )
    assert not result_timeout.ok
    assert "превышено" in (result_timeout.error or "").lower()

    shutil.rmtree(scripts_dir, ignore_errors=True)


def _flatten_paths(tree: list[dict[str, object]]) -> set[str]:
    collected: set[str] = set()
    for node in tree:
        if node.get("type") == "file":
            path = str(node.get("path") or "")
            collected.add(path)
        for child in node.get("children", []) or []:
            collected.update(_flatten_paths([child]))
    return collected
