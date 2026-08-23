from __future__ import annotations

from pathlib import Path

import pytest

from core.tool_gateway import ToolGateway
from server.http.common.workspace_runtime import _run_plan_readonly_audit
from shared.models import ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import ListFilesTool, ReadFileTool, workspace_root_context


def _audit(root: Path) -> tuple[list[dict], dict[str, int]]:
    registry = ToolRegistry()
    registry.register("workspace_list", ListFilesTool(), capability="read")
    registry.register("workspace_read", ReadFileTool(), capability="read")
    registry.set_execution_policy(mode="plan")
    gateway = ToolGateway(registry)
    with workspace_root_context(root):
        return _run_plan_readonly_audit(
            call_tool=gateway.call,
            plan_audit_timeout_seconds=5,
            plan_audit_max_total_bytes=100_000,
            plan_audit_max_read_files=100,
        )


def test_plan_audit_reads_allowed_files(tmp_path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("print('x')", encoding="utf-8")
    (root / "doc.md").write_text("# doc", encoding="utf-8")
    (root / "blob.bin").write_bytes(b"\x00\x01\x02")

    entries, usage = _audit(root)
    paths = [entry["path"] for entry in entries]
    assert "a.py" in paths
    assert "doc.md" in paths
    assert not any(path.endswith("blob.bin") for path in paths)
    assert usage["read_files"] == 2


def test_plan_audit_skips_symlink_escape(tmp_path, monkeypatch) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "ok.py").write_text("print('ok')", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.md"
    secret.write_text("TOP SECRET", encoding="utf-8")
    link = root / "leak.md"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("Symlink недоступен в этом окружении.")

    entries, _ = _audit(root)
    paths = [entry["path"] for entry in entries]
    assert not any(path.endswith("leak.md") for path in paths), paths
    assert any(path.endswith("ok.py") for path in paths)


def test_plan_audit_uses_gateway_tool_requests_only() -> None:
    requests: list[ToolRequest] = []

    def call_tool(request: ToolRequest):
        requests.append(request)
        if request.name == "workspace_list":
            return ToolResult.success(
                {
                    "tree": [
                        {"name": "a.py", "type": "file", "path": "a.py"},
                        {
                            "name": "src",
                            "type": "dir",
                            "path": "src",
                            "children": [
                                {
                                    "name": "b.py",
                                    "type": "file",
                                    "path": "src/b.py",
                                }
                            ],
                        },
                    ]
                }
            )
        return ToolResult.success({"output": f"content:{request.args['path']}"})

    entries, usage = _run_plan_readonly_audit(
        call_tool=call_tool,
        plan_audit_timeout_seconds=5,
        plan_audit_max_total_bytes=100_000,
        plan_audit_max_read_files=100,
    )

    assert [request.name for request in requests] == [
        "workspace_list",
        "workspace_read",
        "workspace_read",
    ]
    assert [entry["path"] for entry in entries] == ["a.py", "src/b.py"]
    assert usage == {"read_files": 2, "total_bytes": 28, "search_calls": 1}
