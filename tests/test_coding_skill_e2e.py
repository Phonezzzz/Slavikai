from __future__ import annotations

from pathlib import Path

from core.mwv.coding_skill import CodingSkill
from core.mwv.models import VerificationStatus, WorkStatus
from core.mwv.verifier import VerifierRunner


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    return repo


def test_coding_skill_happy_path(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    script_path = repo / "scripts" / "check.sh"
    script_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    target = repo / "sample.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    runner = VerifierRunner(script_path=script_path)
    skill = CodingSkill(workspace_root=repo, verifier=runner, max_retries=1)
    result = skill.run("внеси маленькое изменение в файл sample.py")

    assert result.run_result.work_result.status == WorkStatus.SUCCESS
    assert result.run_result.verification_result.status == VerificationStatus.PASSED
    assert "mwv" in target.read_text(encoding="utf-8")


def test_coding_skill_fail_then_fix(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    script_path = repo / "scripts" / "check.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "if grep -q BAD sample.py; then echo 'tests failed' 1>&2; exit 1; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    target = repo / "sample.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    runner = VerifierRunner(script_path=script_path)
    skill = CodingSkill(
        workspace_root=repo,
        verifier=runner,
        max_retries=1,
        change_text="BAD",
        retry_text="GOOD",
    )
    result = skill.run("внеси маленькое изменение в файл sample.py")

    assert result.run_result.attempt == 2
    assert result.run_result.verification_result.status == VerificationStatus.PASSED
    content = target.read_text(encoding="utf-8")
    assert "BAD" not in content
    assert "GOOD" in content


def test_coding_skill_hard_fail(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    script_path = repo / "scripts" / "check.sh"
    script_path.write_text("#!/usr/bin/env bash\necho fail 1>&2\nexit 1\n", encoding="utf-8")
    target = repo / "sample.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    runner = VerifierRunner(script_path=script_path)
    skill = CodingSkill(workspace_root=repo, verifier=runner, max_retries=1)
    result = skill.run("внеси маленькое изменение в файл sample.py")

    assert result.run_result.verification_result.status == VerificationStatus.FAILED
    assert result.run_result.retry_decision is not None
    assert result.run_result.retry_decision.allow_retry is False
    assert "Verifier failed" in result.report
