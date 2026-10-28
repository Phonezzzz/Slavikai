from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.mwv.models import VerificationResult, VerificationStatus
from shared.sandbox_execution import SANDBOX_EXECUTION_MESSAGE, SandboxExecutionReason

DEFAULT_TIMEOUT_SECONDS = 60 * 30
DEFAULT_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check.sh"
MAX_OUTPUT_CHARS = 8000
TRUNCATION_SUFFIX = "\n[truncated]"


@dataclass(frozen=True)
class VerifierRunner:
    script_path: Path = DEFAULT_SCRIPT_PATH
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS

    def run(self) -> VerificationResult:
        command = ["bash", str(self.script_path)]
        if not self.script_path.exists():
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=0.0,
                error=f"Verifier script not found: {self.script_path}",
            )

        return sandbox_verification_unavailable(command)


def sandbox_verification_unavailable(command: list[str]) -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.ERROR,
        command=command,
        exit_code=None,
        stdout="",
        stderr=SANDBOX_EXECUTION_MESSAGE,
        duration_seconds=0.0,
        error=SandboxExecutionReason.UNAVAILABLE.value,
        fail_type="execution_unavailable",
        excerpt=SANDBOX_EXECUTION_MESSAGE,
        verifier_profile="unavailable",
    )
