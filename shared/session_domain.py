from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from shared.models import JSONValue

MessageLane = Literal["chat", "workspace"]
SessionMode = Literal["ask", "plan", "act", "auto"]
PolicyProfile = Literal["sandbox", "index", "yolo"]


@dataclass(frozen=True)
class SessionMessage:
    role: str
    content: str
    created_at: str | None = None
    trace_id: str | None = None
    parent_user_message_id: str | None = None
    attachments: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ChatThread:
    thread_id: str
    model_provider: str | None
    model_id: str | None
    messages: list[SessionMessage]
    workspace_session_id: str | None = None


@dataclass(frozen=True)
class WorkspaceRun:
    run_id: str
    status: str
    summary: str
    trace_id: str | None = None


@dataclass(frozen=True)
class AgentRun:
    run_id: str
    goal: str
    status: str
    trace_id: str | None = None


@dataclass(frozen=True)
class WorkspaceSession:
    session_id: str
    root: str | None
    policy_profile: PolicyProfile
    mode: SessionMode
    messages: list[SessionMessage]
    runs: list[WorkspaceRun] = field(default_factory=list)
    current_run: AgentRun | None = None
    active_plan: dict[str, JSONValue] | None = None
    active_task: dict[str, JSONValue] | None = None
    auto_state: dict[str, JSONValue] | None = None


def normalize_message_lane(value: object, *, default: MessageLane = "chat") -> MessageLane:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "workspace":
            return "workspace"
        if normalized == "chat":
            return "chat"
    return default


def normalize_policy_profile(value: object) -> PolicyProfile:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "index":
            return "index"
        if normalized == "yolo":
            return "yolo"
    return "sandbox"


def normalize_session_mode(value: object) -> SessionMode:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "plan":
            return "plan"
        if normalized == "act":
            return "act"
        if normalized == "auto":
            return "auto"
    return "ask"


def session_message_from_legacy(message: dict[str, JSONValue]) -> SessionMessage:
    role_raw = message.get("role")
    content_raw = message.get("content")
    created_at_raw = message.get("created_at")
    trace_id_raw = message.get("trace_id")
    parent_raw = message.get("parent_user_message_id")
    attachments_raw = message.get("attachments")
    attachments: list[dict[str, str]] = []
    if isinstance(attachments_raw, list):
        for item in attachments_raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            mime = item.get("mime")
            content = item.get("content")
            if isinstance(name, str) and isinstance(mime, str) and isinstance(content, str):
                attachments.append({"name": name, "mime": mime, "content": content})
    return SessionMessage(
        role=role_raw if isinstance(role_raw, str) else "assistant",
        content=content_raw if isinstance(content_raw, str) else "",
        created_at=created_at_raw if isinstance(created_at_raw, str) else None,
        trace_id=trace_id_raw if isinstance(trace_id_raw, str) else None,
        parent_user_message_id=parent_raw if isinstance(parent_raw, str) else None,
        attachments=attachments,
    )


def legacy_messages_for_lane(
    messages: list[dict[str, JSONValue]],
    *,
    lane: MessageLane,
) -> list[SessionMessage]:
    return [
        session_message_from_legacy(message)
        for message in messages
        if normalize_message_lane(message.get("lane"), default="chat") == lane
    ]
