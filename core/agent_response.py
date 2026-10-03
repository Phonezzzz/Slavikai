from __future__ import annotations

from dataclasses import dataclass

from core.auto_runtime import AutoRunOutcome
from llm.stream_model import StreamEvent


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Ответ runtime до HTTP/UI projection; не terminal acceptance или execution evidence."""

    text: str
    auto_outcome: AutoRunOutcome | None = None


@dataclass(frozen=True, slots=True)
class ResponseProduced(StreamEvent):
    """Request-local результат Agent; не provider event и не lifecycle transition."""

    response: AgentResponse
