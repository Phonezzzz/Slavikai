from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

JSONPrimitive = str | bytes | int | float | bool | None
JSONValue = JSONPrimitive | Sequence["JSONValue"] | Mapping[str, "JSONValue"]


class ToolResultStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class ToolRequest:
    name: str
    args: dict[str, JSONValue] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, JSONValue] = field(default_factory=dict)
    error: str | None = None
    meta: dict[str, JSONValue] | None = None

    @classmethod
    def success(
        cls,
        data: dict[str, JSONValue] | None = None,
        meta: dict[str, JSONValue] | None = None,
    ) -> ToolResult:
        return cls(ok=True, data=data or {}, meta=meta)

    @classmethod
    def failure(cls, error: str, meta: dict[str, JSONValue] | None = None) -> ToolResult:
        return cls(ok=False, data={}, error=error, meta=meta)


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"


@dataclass
class PlanStep:
    description: str
    status: PlanStepStatus = PlanStepStatus.PENDING
    operation: str | None = None
    result: str | None = None


@dataclass
class TaskPlan:
    goal: str
    steps: list[PlanStep]


@dataclass
class MemoryItem:
    id: str
    content: str
    tags: list[str]
    timestamp: str


class MemoryKind(str, Enum):
    NOTE = "note"
    USER_PREF = "user_pref"
    PROJECT_FACT = "project_fact"


@dataclass
class MemoryRecord:
    id: str
    kind: MemoryKind
    content: str
    tags: list[str]
    timestamp: str
    meta: dict[str, JSONValue] | None = None


@dataclass
class UserPreference:
    id: str
    key: str
    value: JSONValue
    timestamp: str
    source: str = "user"
    tags: list[str] = field(default_factory=list)
    meta: dict[str, JSONValue] | None = None


@dataclass
class ProjectFact:
    id: str
    project: str
    content: str
    timestamp: str
    tags: list[str] = field(default_factory=list)
    meta: dict[str, JSONValue] | None = None


class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass(frozen=True)
class VectorSearchResult:
    path: str
    snippet: str
    score: float
    meta: dict[str, JSONValue] | None = None


@dataclass(frozen=True)
class ToolCallRecord:
    timestamp: str
    tool: str
    ok: bool
    error: str | None = None
    meta: dict[str, JSONValue] | None = None
    args: dict[str, JSONValue] | None = None


@dataclass(frozen=True)
class WorkspaceDiffEntry:
    path: str
    added: int
    removed: int
    diff: str
