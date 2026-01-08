from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.mwv.models import RunContext, VerificationResult
from core.mwv.verifier import VerifierRunner


class VerifierRunnerProtocol(Protocol):
    def run(self) -> VerificationResult: ...


def _default_runner() -> VerifierRunnerProtocol:
    return VerifierRunner()


@dataclass(frozen=True)
class VerifierRuntime:
    runner: VerifierRunnerProtocol = field(default_factory=_default_runner)

    def run(self, context: RunContext) -> VerificationResult:
        _ = context
        return self.runner.run()
