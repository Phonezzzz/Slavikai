from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from core.mwv.models import VerificationResult, VerificationStatus

DEFAULT_TIMEOUT_SECONDS = 60 * 30
DEFAULT_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check.sh"


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

        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=self._project_root(),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=command,
                exit_code=None,
                stdout=_coerce_output(exc.stdout),
                stderr=_coerce_output(exc.stderr),
                duration_seconds=duration,
                error="verifier_timeout",
            )
        except OSError as exc:
            duration = time.monotonic() - start
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=duration,
                error=str(exc),
            )

        duration = time.monotonic() - start
        status = (
            VerificationStatus.PASSED if completed.returncode == 0 else VerificationStatus.FAILED
        )
        return VerificationResult(
            status=status,
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_seconds=duration,
            error=None,
        )

    def _project_root(self) -> Path:
        return self.script_path.resolve().parents[1]


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
