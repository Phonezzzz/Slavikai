from __future__ import annotations

import shutil
from pathlib import Path

import tools.workspace_tools as workspace_tools_module
from config.shell_config import ShellConfig
from shared.models import ToolRequest
from tools.workspace_tools import (
    MAX_TREE_ENTRIES,
    WORKSPACE_ROOT,
    ApplyPatchTool,
    CreateFileTool,
    DeleteFileTool,
    ListFilesTool,
    MoveFileTool,
    ReadFileTool,
    RenameFileTool,
    RunCodeTool,
    WorkspaceTerminalRunTool,
    WriteFileTool,
    workspace_root_context,
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

    tree_result = ListFilesTool().handle(_make_request("workspace_list", {"recursive": True}))
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


def test_workspace_write_rejects_sibling_prefix_escape() -> None:
    sibling = WORKSPACE_ROOT.parent / f"{WORKSPACE_ROOT.name}2"
    sibling.mkdir(parents=True, exist_ok=True)
    target = sibling / "escape.txt"
    if target.exists():
        target.unlink()
    result = WriteFileTool().handle(
        _make_request("workspace_write", {"path": f"../{sibling.name}/escape.txt", "content": "x"})
    )
    assert not result.ok
    assert not target.exists()


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


def test_workspace_create_rename_move_delete_file() -> None:
    create_result = CreateFileTool().handle(
        _make_request(
            "workspace_create",
            {"path": "ops/new_file.txt", "content": "hello", "overwrite": False},
        )
    )
    assert create_result.ok

    rename_result = RenameFileTool().handle(
        _make_request(
            "workspace_rename",
            {"old_path": "ops/new_file.txt", "new_path": "ops/renamed_file.txt"},
        )
    )
    assert rename_result.ok

    move_result = MoveFileTool().handle(
        _make_request(
            "workspace_move",
            {"from_path": "ops/renamed_file.txt", "to_path": "ops/moved/file.txt"},
        )
    )
    assert move_result.ok

    moved_read = ReadFileTool().handle(
        _make_request("workspace_read", {"path": "ops/moved/file.txt"})
    )
    assert moved_read.ok
    assert moved_read.data.get("output") == "hello"

    delete_result = DeleteFileTool().handle(
        _make_request("workspace_delete", {"path": "ops/moved/file.txt", "recursive": False})
    )
    assert delete_result.ok

    read_deleted = ReadFileTool().handle(
        _make_request("workspace_read", {"path": "ops/moved/file.txt"})
    )
    assert not read_deleted.ok


def test_workspace_delete_rejects_root_and_allows_recursive_dir_delete() -> None:
    nested_dir = WORKSPACE_ROOT / "ops" / "to_remove"
    shutil.rmtree(nested_dir, ignore_errors=True)
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "tmp.txt").write_text("x", encoding="utf-8")

    root_delete = DeleteFileTool().handle(
        _make_request("workspace_delete", {"path": ".", "recursive": True})
    )
    assert not root_delete.ok
    assert "корня workspace" in (root_delete.error or "")
    assert WORKSPACE_ROOT.exists()

    dir_delete = DeleteFileTool().handle(
        _make_request("workspace_delete", {"path": "ops/to_remove", "recursive": True})
    )
    assert dir_delete.ok
    assert not nested_dir.exists()


def test_workspace_terminal_run_uses_workspace_root_cwd(monkeypatch) -> None:
    monkeypatch.setattr(
        workspace_tools_module,
        "load_shell_config",
        lambda path: ShellConfig(
            allowed_commands=["echo"],
            timeout_seconds=5,
            max_output_chars=2000,
            sandbox_root="sandbox",
        ),
    )
    command_result = WorkspaceTerminalRunTool().handle(
        _make_request("workspace_terminal_run", {"command": "echo ok", "cwd_mode": "session_root"})
    )
    assert command_result.ok
    cwd_value = str(command_result.data.get("cwd") or "")
    assert cwd_value == str(WORKSPACE_ROOT)


def test_workspace_list_returns_tree_meta_and_truncates_by_max_entries(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(workspace_tools_module, "WORKSPACE_ROOT", tmp_path)
    workspace_tools_module.set_workspace_root(None)
    monkeypatch.setattr(workspace_tools_module, "MAX_TREE_ENTRIES", 5)
    monkeypatch.setattr(workspace_tools_module, "MAX_CHILDREN_PER_DIR", 100)
    for idx in range(10):
        (tmp_path / f"f{idx}.txt").write_text("x", encoding="utf-8")

    result = ListFilesTool().handle(_make_request("workspace_list", {"recursive": False}))
    assert result.ok
    tree = result.data.get("tree")
    assert isinstance(tree, list)
    assert len(tree) == 5
    tree_meta = result.data.get("tree_meta")
    assert isinstance(tree_meta, dict)
    assert tree_meta.get("truncated") is True
    reasons = tree_meta.get("truncated_reasons")
    assert isinstance(reasons, list)
    assert "max_entries" in reasons
    assert tree_meta.get("returned_entries") == 5
    assert tree_meta.get("max_entries") == 5


def test_workspace_list_marks_children_truncated_per_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(workspace_tools_module, "WORKSPACE_ROOT", tmp_path)
    workspace_tools_module.set_workspace_root(None)
    monkeypatch.setattr(workspace_tools_module, "MAX_TREE_ENTRIES", MAX_TREE_ENTRIES)
    monkeypatch.setattr(workspace_tools_module, "MAX_CHILDREN_PER_DIR", 2)

    nested = tmp_path / "many"
    nested.mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        (nested / f"child{idx}.txt").write_text("x", encoding="utf-8")

    result = ListFilesTool().handle(_make_request("workspace_list", {"recursive": True}))
    assert result.ok
    tree = result.data.get("tree")
    assert isinstance(tree, list)
    many_node = next(
        (item for item in tree if isinstance(item, dict) and item.get("name") == "many"),
        None,
    )
    assert isinstance(many_node, dict)
    assert many_node.get("children_truncated") is True
    children = many_node.get("children")
    assert isinstance(children, list)
    assert len(children) == 2
    tree_meta = result.data.get("tree_meta")
    assert isinstance(tree_meta, dict)
    assert tree_meta.get("truncated") is True
    reasons = tree_meta.get("truncated_reasons")
    assert isinstance(reasons, list)
    assert "max_children_per_dir" in reasons
    assert tree_meta.get("max_children_per_dir") == 2


def test_workspace_list_respects_max_depth() -> None:
    base = WORKSPACE_ROOT / "depth_test"
    shutil.rmtree(base, ignore_errors=True)
    nested = base / "a" / "b"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "leaf.txt").write_text("leaf", encoding="utf-8")

    result = ListFilesTool().handle(
        _make_request("workspace_list", {"path": "depth_test", "recursive": True, "max_depth": 0})
    )
    assert result.ok
    tree = result.data.get("tree")
    assert isinstance(tree, list)
    a_node = next(
        (item for item in tree if isinstance(item, dict) and item.get("name") == "a"),
        None,
    )
    assert isinstance(a_node, dict)
    children = a_node.get("children")
    assert children is None or children == []
    tree_meta = result.data.get("tree_meta")
    assert isinstance(tree_meta, dict)
    assert tree_meta.get("max_depth_applied") == 0
    shutil.rmtree(base, ignore_errors=True)


def test_workspace_root_context_isolation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(workspace_tools_module, "WORKSPACE_ROOT", tmp_path / "default")
    default_root = workspace_tools_module.WORKSPACE_ROOT
    default_root.mkdir(parents=True, exist_ok=True)
    isolated = tmp_path / "isolated"
    isolated.mkdir(parents=True, exist_ok=True)

    assert workspace_tools_module.get_workspace_root() == default_root
    with workspace_root_context(isolated):
        assert workspace_tools_module.get_workspace_root() == isolated
    assert workspace_tools_module.get_workspace_root() == default_root


def _flatten_paths(tree: list[dict[str, object]]) -> set[str]:
    collected: set[str] = set()
    for node in tree:
        if node.get("type") == "file":
            path = str(node.get("path") or "")
            collected.add(path)
        for child in node.get("children", []) or []:
            collected.update(_flatten_paths([child]))
    return collected
