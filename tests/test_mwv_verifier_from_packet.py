from __future__ import annotations

import sys
from pathlib import Path

from core.mwv.models import RunContext, TaskPacket, VerificationStatus
from core.mwv.verifier_runtime import VerifierRuntime


def test_verifier_runtime_uses_packet_command(tmp_path: Path) -> None:
    task = TaskPacket(
        task_id="task-1",
        session_id="session-1",
        trace_id="trace-1",
        goal="run verifier",
        verifier={
            "command": [sys.executable, "-c", "print('packet-verifier-ok')"],
            "cwd": ".",
            "timeout_seconds": 5,
        },
    )
    context = RunContext(
        session_id="session-1",
        trace_id="trace-1",
        workspace_root=str(tmp_path),
        safe_mode=True,
    )
    runtime = VerifierRuntime(project_root=tmp_path)

    result = runtime.run(task, context)
    assert result.status == VerificationStatus.PASSED
    assert result.command[0] == sys.executable
    assert "packet-verifier-ok" in result.stdout


def test_verifier_runtime_blocks_cwd_outside_workspace(tmp_path: Path) -> None:
    outside = (tmp_path / "..").resolve()
    task = TaskPacket(
        task_id="task-1",
        session_id="session-1",
        trace_id="trace-1",
        goal="run verifier",
        verifier={
            "command": [sys.executable, "-c", "print('x')"],
            "cwd": str(outside),
        },
    )
    context = RunContext(
        session_id="session-1",
        trace_id="trace-1",
        workspace_root=str(tmp_path),
        safe_mode=True,
    )
    runtime = VerifierRuntime(project_root=tmp_path)

    result = runtime.run(task, context)
    assert result.status == VerificationStatus.ERROR
    assert result.error is not None
    assert "invalid_verifier_config" in result.error
