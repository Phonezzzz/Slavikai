from __future__ import annotations

from core.mwv.models import VerificationResult, VerificationStatus
from core.mwv.verifier_summary import extract_verifier_excerpt, summarize_verifier_failure


def test_verifier_excerpt_prefers_multiline_stderr() -> None:
    result = VerificationResult(
        status=VerificationStatus.FAILED,
        command=["pytest"],
        exit_code=1,
        stdout="",
        stderr="\n===\nline one\n\nline two\nline three\nline four",
        duration_seconds=0.2,
    )

    excerpt = extract_verifier_excerpt(result, max_lines=3, max_chars=200)

    assert excerpt == "line one\nline two\nline three"
    assert summarize_verifier_failure(result) == "line one\nline two"


def test_verifier_excerpt_falls_back_to_stdout_and_exit_code() -> None:
    stdout_result = VerificationResult(
        status=VerificationStatus.FAILED,
        command=["ruff"],
        exit_code=1,
        stdout="stdout one\nstdout two",
        stderr="",
        duration_seconds=0.1,
    )
    assert extract_verifier_excerpt(stdout_result) == "stdout one\nstdout two"

    exit_code_result = VerificationResult(
        status=VerificationStatus.ERROR,
        command=["check"],
        exit_code=2,
        stdout="",
        stderr="",
        duration_seconds=0.1,
    )
    assert extract_verifier_excerpt(exit_code_result) == "exit_code=2"
