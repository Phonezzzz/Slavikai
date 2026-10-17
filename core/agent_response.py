from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.auto_runtime import AutoRunOutcome
from core.desktop_runtime import DesktopRunOutcome
from core.mwv.manager import MWVRunResult
from core.tool_loop import AgentToolLoopResult
from llm.stream_model import StreamEvent
from llm.types import LLMResult


@dataclass(frozen=True, slots=True)
class LocalResponseOutcome:
    """Наблюдаемый локальный ответ/decision, без утверждения о tool execution или acceptance."""

    reason: Literal["memory", "skill_blocked", "decision_required"]
    execution_started: Literal[False] = False


RuntimeResult = (
    AutoRunOutcome
    | AgentToolLoopResult
    | LLMResult
    | MWVRunResult
    | DesktopRunOutcome
    | LocalResponseOutcome
)


@dataclass(frozen=True, slots=True)
class ResponseFailure:
    """Ошибка получения/projection ответа; не outcome уже выполненных tools."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Ответ runtime до HTTP/UI projection; не terminal acceptance или execution evidence."""

    text: str
    runtime_result: RuntimeResult | None = None
    failure: ResponseFailure | None = None


@dataclass(frozen=True, slots=True)
class ResponseProduced(StreamEvent):
    """Request-local результат Agent; не provider event и не lifecycle transition."""

    response: AgentResponse
