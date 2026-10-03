from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.agent_response import AgentResponse
from core.mwv.models import RunContext, TaskPacket, VerificationResult, WorkResult
from shared.models import LLMMessage


@runtime_checkable
class AgentFacade(Protocol):
    def respond(self, messages: list[LLMMessage]) -> AgentResponse: ...


@runtime_checkable
class WorkerFacade(Protocol):
    def run(self, task: TaskPacket, context: RunContext) -> WorkResult: ...


@runtime_checkable
class VerifierFacade(Protocol):
    def run(self, task: TaskPacket, context: RunContext) -> VerificationResult: ...
