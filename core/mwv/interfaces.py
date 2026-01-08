from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.mwv.models import RunContext, TaskPacket, VerificationResult, WorkResult
from shared.models import LLMMessage, TaskPlan


@runtime_checkable
class AgentFacade(Protocol):
    def respond(self, messages: list[LLMMessage]) -> str: ...


@runtime_checkable
class PlannerFacade(Protocol):
    def build_plan(self, goal: str) -> TaskPlan: ...

    def execute_plan(self, plan: TaskPlan) -> TaskPlan: ...


@runtime_checkable
class WorkerFacade(Protocol):
    def run(self, task: TaskPacket, context: RunContext) -> WorkResult: ...


@runtime_checkable
class VerifierFacade(Protocol):
    def run(self, context: RunContext) -> VerificationResult: ...
