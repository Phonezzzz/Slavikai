from __future__ import annotations

from pathlib import Path

from core.mwv.coding_skill import CodingSkill
from core.mwv.models import VerificationStatus, WorkStatus
from core.mwv.verifier import VerifierRunner
from shared.models import ToolRequest


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "scripts").mkdir()
    return repo


def test_coding_skill_preserves_write_before_verifier_unavailable(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    script_path = repo / "scripts" / "check.sh"
    script_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    target = repo / "sample.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    runner = VerifierRunner(script_path=script_path)
    skill = CodingSkill(
        workspace_root=repo,
        verifier=runner,
        request_builder=lambda _task, _context: [
            ToolRequest(
                name="workspace_write",
                args={"path": "sample.py", "content": "print('hi')\nprint('mwv')\n"},
            )
        ],
    )
    result = skill.run("внеси маленькое изменение в файл sample.py")

    assert result.run_result.work_result.status == WorkStatus.SUCCESS
    assert result.run_result.verification_result.status == VerificationStatus.ERROR
    assert "mwv" in target.read_text(encoding="utf-8")


def test_coding_skill_does_not_retry_when_verifier_unavailable(tmp_path: Path) -> None:
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
        request_builder=lambda _task, context: [
            ToolRequest(
                name="workspace_write",
                args={
                    "path": "sample.py",
                    "content": "print('hi')\nBAD\n"
                    if context.attempt <= 1
                    else "print('hi')\nGOOD\n",
                },
            )
        ],
    )
    result = skill.run("внеси маленькое изменение в файл sample.py")

    assert result.run_result.attempt == 1
    assert result.run_result.verification_result.status == VerificationStatus.ERROR
    content = target.read_text(encoding="utf-8")
    assert "BAD" in content
    assert "GOOD" not in content
    assert result.run_result.verification_result.error == "sandbox_execution_unavailable"


def test_coding_skill_verifier_unavailable_is_not_test_failure(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    script_path = repo / "scripts" / "check.sh"
    script_path.write_text("#!/usr/bin/env bash\necho fail 1>&2\nexit 1\n", encoding="utf-8")
    target = repo / "sample.py"
    target.write_text("print('hi')\n", encoding="utf-8")

    runner = VerifierRunner(script_path=script_path)
    skill = CodingSkill(
        workspace_root=repo,
        verifier=runner,
        request_builder=lambda _task, _context: [
            ToolRequest(
                name="workspace_write",
                args={"path": "sample.py", "content": "print('hi')\nprint('mwv')\n"},
            )
        ],
    )
    result = skill.run("внеси маленькое изменение в файл sample.py")

    assert result.run_result.attempt == 1
    assert result.run_result.verification_result.status == VerificationStatus.ERROR
    assert result.run_result.retry_decision is not None
    assert result.run_result.retry_decision.allow_retry is False
    assert "sandbox_execution_unavailable" == result.run_result.verification_result.error
