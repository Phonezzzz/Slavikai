import pytest

from core.mwv.models import RunContext, TaskPacket, VerificationStatus
from core.mwv.verifier_runtime import VerifierRuntime, is_repo_workspace


@pytest.mark.parametrize(
    "verifier",
    [{}, {"command": ["make", "check"]}, {"command": "python untrusted.py", "timeout_seconds": 1}],
)
def test_verifier_explicit_and_fallback_refuse_without_calling_runner(
    tmp_path, monkeypatch, verifier
):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: calls.append(a))

    class Runner:
        def run(self):
            calls.append("runner")
            raise AssertionError("unqualified fallback runner")

    task = TaskPacket(task_id="t", session_id="s", trace_id="x", goal="check", verifier=verifier)
    context = RunContext(
        session_id="s", trace_id="x", workspace_root=str(tmp_path), safe_mode=False
    )
    result = VerifierRuntime(runner=Runner()).run(task, context)
    assert calls == []
    assert result.status == VerificationStatus.ERROR
    assert result.error == "sandbox_execution_unavailable"
    assert result.exit_code is None


@pytest.mark.parametrize("command", [[], [1], 1])
def test_verifier_keeps_invalid_configuration_error(tmp_path, command):
    task = TaskPacket(
        task_id="t", session_id="s", trace_id="x", goal="check", verifier={"command": command}
    )
    context = RunContext(
        session_id="s", trace_id="x", workspace_root=str(tmp_path), safe_mode=False
    )
    result = VerifierRuntime().run(task, context)
    assert result.status == VerificationStatus.ERROR
    assert result.error.startswith("invalid_verifier_config")


def test_preexisting_cancellation_precedes_unavailable(tmp_path):
    task = TaskPacket(task_id="t", session_id="s", trace_id="x", goal="check")
    context = RunContext(
        session_id="s", trace_id="x", workspace_root=str(tmp_path), safe_mode=False
    )
    result = VerifierRuntime().run(task, context, cancelled=lambda: True)
    assert result.error == "verifier_cancelled"


@pytest.mark.parametrize("marker", ["directory", "worktree_file"])
def test_repository_detection_never_executes_git_and_keeps_root_scope(
    tmp_path, monkeypatch, marker
):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    if marker == "directory":
        (tmp_path / ".git").mkdir()
    else:
        (tmp_path / ".git").write_text("gitdir: /untrusted/path")
    child = tmp_path / "nested"
    child.mkdir()
    assert is_repo_workspace(tmp_path)
    assert not is_repo_workspace(child)
    assert calls == []
