from __future__ import annotations

from pathlib import Path

from core.mwv.coding_task import CodingTaskRuntime
from core.mwv.models import VerificationStatus, WorkStatus
from core.mwv.verifier import VerifierRunner


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    return repo


def test_single_attempt_coding_pass(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    script_path = repo / "scripts" / "check.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "if ! grep -q \"mwv change\" sample.py; then echo 'missing' 1>&2; exit 1; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    target = repo / "sample.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    runner = VerifierRunner(script_path=script_path)
    runtime = CodingTaskRuntime(workspace_root=repo, verifier=runner)
    result = runtime.run("make a small change in file sample.py")

    assert result.attempt == 1
    assert result.work_result.status == WorkStatus.SUCCESS
    assert result.verification_result.status == VerificationStatus.PASSED
    assert str(target) in result.report.changed_files
    assert "Changed files:" in result.report_text
    assert "Verifier: PASS" in result.report_text
    assert "Next steps:" in result.report_text
    assert 1 <= len(result.report.next_steps) <= 3


def test_single_attempt_coding_fail_with_diagnostics(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    script_path = repo / "scripts" / "check.sh"
    script_path.write_text("#!/usr/bin/env bash\necho fail 1>&2\nexit 1\n", encoding="utf-8")
    target = repo / "sample.py"
    target.write_text("print('hi')\n# mwv change\n", encoding="utf-8")

    runner = VerifierRunner(script_path=script_path)
    runtime = CodingTaskRuntime(workspace_root=repo, verifier=runner)
    result = runtime.run("make a small change in file sample.py")

    assert result.attempt == 1
    assert result.work_result.status == WorkStatus.FAILURE
    assert result.verification_result.status == VerificationStatus.FAILED
    assert result.report.diagnostics is not None
    assert result.report.diagnostics.command[-1].endswith("check.sh")
    assert result.report.diagnostics.summary
    assert "Verifier: FAIL" in result.report_text
    assert "Diagnostics:" in result.report_text
    assert "Next steps:" in result.report_text
