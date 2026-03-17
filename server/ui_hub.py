from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, TypedDict, cast

from server.ui_session_storage import (
    ChatRecord,
    InMemoryUISessionStorage,
    MessageRecord,
    PersistedSession,
    RuntimeStateRecord,
    UISessionStorage,
    WorkspaceActivityRecord,
    WorkspaceRecord,
)
from shared.auto_models import normalize_auto_state
from shared.models import JSONValue


def _utc_iso_now() -> str:
    return datetime.now(UTC).isoformat()


SessionStatus = Literal["ok", "busy", "error"]
PolicyProfile = Literal["sandbox", "index", "yolo"]
SessionMode = Literal["ask", "plan", "act", "auto"]
EntityAccess = Literal["owned", "forbidden", "missing"]
MessageLane = Literal["chat", "workspace"]
_UNSET: object = object()

DEFAULT_EVENT_BUFFER_SIZE = 512
DEFAULT_SUBSCRIBER_QUEUE_MAXSIZE = 256
SubscriberDropPolicy = Literal["drop_oldest", "coalesce_deltas"]
DEFAULT_SUBSCRIBER_DROP_POLICY: SubscriberDropPolicy = "drop_oldest"
DEFAULT_LEGACY_PRINCIPAL_ID = "legacy"
DEFAULT_WORKSPACE_TITLE = "New Workspace"
DEFAULT_CHAT_TITLE = "New Chat"
_MESSAGE_ROLES = {"user", "assistant", "system"}
_CONTROL_EVENT_TYPES = {
    "decision.packet",
    "session.workflow",
    "session.resync_required",
    "status",
}


def _normalize_message_lane(value: object, *, default: MessageLane = "chat") -> MessageLane:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "workspace":
            return "workspace"
        if normalized == "chat":
            return "chat"
    return default


class WorkspaceListItem(TypedDict):
    workspace_id: str
    title: str
    root_path: str | None
    created_at: str
    updated_at: str
    chat_count: int


class ChatListItem(TypedDict):
    chat_id: str
    workspace_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class SessionListItem(TypedDict):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    chat_message_count: int
    workspace_message_count: int
    last_message_lane: MessageLane | None
    title_override: str | None
    folder_id: str | None


SessionAccess = EntityAccess


@dataclass
class _EventRecord:
    event_id: str
    recorded_at: datetime
    event: dict[str, JSONValue]


@dataclass
class _SubscriberState:
    queue: asyncio.Queue[dict[str, JSONValue]]
    pending_resync_reason: str | None = None


@dataclass
class _WorkspaceState:
    principal_id: str
    title: str = DEFAULT_WORKSPACE_TITLE
    root_path: str | None = None
    policy_profile: PolicyProfile = "sandbox"
    yolo_armed: bool = False
    yolo_armed_at: str | None = None
    tools_state: dict[str, bool] | None = None
    pending_decision: dict[str, JSONValue] | None = None
    created_at: str = field(default_factory=_utc_iso_now)
    updated_at: str = field(default_factory=_utc_iso_now)


@dataclass
class _RuntimeState:
    mode: SessionMode = "ask"
    active_plan: dict[str, JSONValue] | None = None
    active_task: dict[str, JSONValue] | None = None
    auto_state: dict[str, JSONValue] | None = None
    decision: dict[str, JSONValue] | None = None
    status_state: SessionStatus = "ok"
    updated_at: str = field(default_factory=_utc_iso_now)


@dataclass
class _ChatState:
    workspace_id: str
    title: str = DEFAULT_CHAT_TITLE
    messages: list[dict[str, JSONValue]] = field(default_factory=list)
    artifacts: list[dict[str, JSONValue]] = field(default_factory=list)
    output_text: str | None = None
    output_updated_at: str | None = None
    files: list[str] = field(default_factory=list)
    model_provider: str | None = None
    model_id: str | None = None
    runtime: _RuntimeState = field(default_factory=_RuntimeState)
    subscribers: dict[asyncio.Queue[dict[str, JSONValue]], _SubscriberState] = field(
        default_factory=dict
    )
    event_buffer: deque[_EventRecord] = field(default_factory=deque)
    created_at: str = field(default_factory=_utc_iso_now)
    updated_at: str = field(default_factory=_utc_iso_now)


def _normalize_message_attachments(value: JSONValue) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("attachments must be array")
    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("attachments item must be object")
        name_raw = item.get("name")
        mime_raw = item.get("mime")
        content_raw = item.get("content")
        if not isinstance(name_raw, str) or not name_raw.strip():
            raise ValueError("attachments.name required")
        if not isinstance(mime_raw, str) or not mime_raw.strip():
            raise ValueError("attachments.mime required")
        if not isinstance(content_raw, str):
            raise ValueError("attachments.content required")
        normalized.append(
            {
                "name": name_raw.strip(),
                "mime": mime_raw.strip(),
                "content": content_raw,
            }
        )
    return normalized


def _build_message(
    *,
    role: str,
    content: str,
    lane: str = "chat",
    trace_id: str | None = None,
    parent_user_message_id: str | None = None,
    attachments: list[dict[str, str]] | None = None,
    message_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, JSONValue]:
    normalized_role = role.strip()
    if normalized_role not in _MESSAGE_ROLES:
        raise ValueError(f"unsupported message role: {role}")
    normalized_lane = _normalize_message_lane(lane, default="chat")
    normalized_attachments = list(attachments or []) if normalized_role == "user" else []
    return {
        "message_id": (message_id or uuid.uuid4().hex).strip(),
        "role": normalized_role,
        "lane": normalized_lane,
        "content": content,
        "created_at": (created_at or _utc_iso_now()).strip(),
        "trace_id": trace_id.strip() if isinstance(trace_id, str) and trace_id.strip() else None,
        "parent_user_message_id": (
            parent_user_message_id.strip()
            if isinstance(parent_user_message_id, str) and parent_user_message_id.strip()
            else None
        ),
        "attachments": normalized_attachments,
    }


def _normalize_message_payload(message: dict[str, JSONValue]) -> dict[str, JSONValue]:
    message_id = message.get("message_id")
    role = message.get("role")
    lane = message.get("lane")
    content = message.get("content")
    created_at = message.get("created_at")
    trace_id = message.get("trace_id")
    parent_user_message_id = message.get("parent_user_message_id")
    attachments = message.get("attachments")
    if not isinstance(message_id, str) or not message_id.strip():
        raise ValueError("message_id required")
    if not isinstance(role, str) or role.strip() not in _MESSAGE_ROLES:
        raise ValueError("role required")
    if not isinstance(content, str):
        raise ValueError("content required")
    if lane is not None and not isinstance(lane, str):
        raise ValueError("lane must be string or null")
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("created_at required")
    if trace_id is not None and (not isinstance(trace_id, str) or not trace_id.strip()):
        raise ValueError("trace_id must be string or null")
    if parent_user_message_id is not None and (
        not isinstance(parent_user_message_id, str) or not parent_user_message_id.strip()
    ):
        raise ValueError("parent_user_message_id must be string or null")
    normalized_role = role.strip()
    normalized_lane = _normalize_message_lane(lane, default="chat")
    normalized_trace = trace_id.strip() if isinstance(trace_id, str) and trace_id.strip() else None
    normalized_parent = (
        parent_user_message_id.strip()
        if isinstance(parent_user_message_id, str) and parent_user_message_id.strip()
        else None
    )
    normalized_attachments = _normalize_message_attachments(attachments)
    if normalized_role != "assistant":
        normalized_trace = None
        normalized_parent = None
    if normalized_role != "user":
        normalized_attachments = []
    return {
        "message_id": message_id.strip(),
        "role": normalized_role,
        "lane": normalized_lane,
        "content": content,
        "created_at": created_at.strip(),
        "trace_id": normalized_trace,
        "parent_user_message_id": normalized_parent,
        "attachments": normalized_attachments,
    }


class UIHub:
    def __init__(
        self,
        *,
        storage: UISessionStorage | None = None,
        subscriber_queue_maxsize: int = DEFAULT_SUBSCRIBER_QUEUE_MAXSIZE,
        event_buffer_size: int = DEFAULT_EVENT_BUFFER_SIZE,
        subscriber_drop_policy: SubscriberDropPolicy = DEFAULT_SUBSCRIBER_DROP_POLICY,
    ) -> None:
        self._storage: UISessionStorage = storage or InMemoryUISessionStorage()
        self._workspaces: dict[str, _WorkspaceState] = {}
        self._chats: dict[str, _ChatState] = {}
        self._workspace_activity: dict[str, list[WorkspaceActivityRecord]] = {}
        self._subscriber_queue_maxsize = max(1, subscriber_queue_maxsize)
        self._event_buffer_size = max(1, event_buffer_size)
        self._subscriber_drop_policy: SubscriberDropPolicy = (
            subscriber_drop_policy
            if subscriber_drop_policy in {"drop_oldest", "coalesce_deltas"}
            else DEFAULT_SUBSCRIBER_DROP_POLICY
        )
        self._lock = asyncio.Lock()
        self._restore_state()

    async def create_workspace(self, principal_id: str, *, title: str | None = None) -> str:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            workspace_id = uuid.uuid4().hex
            now = _utc_iso_now()
            self._workspaces[workspace_id] = _WorkspaceState(
                principal_id=normalized_principal,
                title=(
                    title.strip()
                    if isinstance(title, str) and title.strip()
                    else DEFAULT_WORKSPACE_TITLE
                ),
                created_at=now,
                updated_at=now,
            )
            self._persist_workspace_locked(workspace_id)
            return workspace_id

    async def list_workspaces(self, principal_id: str) -> list[WorkspaceListItem]:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            items: list[WorkspaceListItem] = []
            for workspace_id, state in self._workspaces.items():
                if state.principal_id != normalized_principal:
                    continue
                chat_count = sum(
                    1 for chat in self._chats.values() if chat.workspace_id == workspace_id
                )
                items.append(
                    {
                        "workspace_id": workspace_id,
                        "title": state.title,
                        "root_path": state.root_path,
                        "created_at": state.created_at,
                        "updated_at": state.updated_at,
                        "chat_count": chat_count,
                    }
                )
            items.sort(key=lambda item: self._parse_timestamp(item["updated_at"]), reverse=True)
            return items

    async def get_workspace(self, workspace_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._workspaces.get(workspace_id)
            if state is None:
                return None
            return self._workspace_payload(workspace_id, state)

    async def get_workspace_access(self, workspace_id: str, principal_id: str) -> EntityAccess:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            state = self._workspaces.get(workspace_id)
            if state is None:
                return "missing"
            if state.principal_id != normalized_principal:
                return "forbidden"
            return "owned"

    async def delete_workspace(self, workspace_id: str) -> bool:
        async with self._lock:
            if workspace_id not in self._workspaces:
                return False
            chat_ids = [
                chat_id
                for chat_id, chat in self._chats.items()
                if chat.workspace_id == workspace_id
            ]
            for chat_id in chat_ids:
                self._drop_chat_locked(chat_id)
            self._workspaces.pop(workspace_id, None)
            self._storage.delete_workspace_activity([workspace_id])
            self._storage.delete_workspaces([workspace_id])
            return True

    async def create_chat(
        self,
        workspace_id: str,
        *,
        title: str | None = None,
    ) -> str:
        async with self._lock:
            if workspace_id not in self._workspaces:
                raise KeyError("workspace not found")
            chat_id = uuid.uuid4().hex
            now = _utc_iso_now()
            self._chats[chat_id] = _ChatState(
                workspace_id=workspace_id,
                title=(
                    title.strip()
                    if isinstance(title, str) and title.strip()
                    else DEFAULT_CHAT_TITLE
                ),
                runtime=_RuntimeState(updated_at=now),
                created_at=now,
                updated_at=now,
            )
            self._persist_chat_locked(chat_id)
            self._persist_runtime_locked(chat_id)
            self._touch_workspace_locked(workspace_id)
            return chat_id

    async def get_or_create_session(
        self,
        session_id: str | None,
        *,
        principal_id: str,
    ) -> str:
        normalized_principal = self._normalize_principal_id(principal_id)
        normalized_session = (
            session_id.strip() if isinstance(session_id, str) and session_id.strip() else None
        )
        async with self._lock:
            if normalized_session:
                access = self._chat_access_locked(normalized_session, normalized_principal)
                if access == "owned":
                    return normalized_session
                if access == "forbidden":
                    raise PermissionError("chat access forbidden")
                workspace_access = self._workspace_access_locked(
                    normalized_session,
                    normalized_principal,
                )
                if workspace_access == "owned":
                    return normalized_session
                if workspace_access == "forbidden":
                    raise PermissionError("workspace access forbidden")
            return self._create_session_locked(
                normalized_principal,
                session_id=normalized_session,
            )

    async def create_session(self, principal_id: str) -> str:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            return self._create_session_locked(normalized_principal)

    async def list_sessions(self, principal_id: str) -> list[SessionListItem]:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            items: list[SessionListItem] = []
            for chat_id, state in self._chats.items():
                workspace = self._workspaces.get(state.workspace_id)
                if workspace is None or workspace.principal_id != normalized_principal:
                    continue
                chat_count = sum(
                    1
                    for message in state.messages
                    if _normalize_message_lane(message.get("lane"), default="chat") == "chat"
                )
                workspace_count = sum(
                    1
                    for message in state.messages
                    if _normalize_message_lane(message.get("lane"), default="chat") == "workspace"
                )
                last_lane: MessageLane | None = None
                if state.messages:
                    lane_raw = state.messages[-1].get("lane")
                    last_lane = _normalize_message_lane(lane_raw, default="chat")
                items.append(
                    {
                        "session_id": chat_id,
                        "title": state.title,
                        "created_at": state.created_at,
                        "updated_at": state.updated_at,
                        "message_count": chat_count,
                        "chat_message_count": chat_count,
                        "workspace_message_count": workspace_count,
                        "last_message_lane": last_lane,
                        "title_override": state.title,
                        "folder_id": None,
                    }
                )
            items.sort(key=lambda item: self._parse_timestamp(item["updated_at"]), reverse=True)
            return items

    async def export_sessions(self, principal_id: str) -> list[PersistedSession]:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            session_ids = [
                chat_id
                for chat_id, state in self._chats.items()
                if (
                    (workspace := self._workspaces.get(state.workspace_id)) is not None
                    and workspace.principal_id == normalized_principal
                )
            ]
            session_ids.sort(
                key=lambda chat_id: self._parse_timestamp(
                    self._require_chat_locked(chat_id).updated_at
                ),
                reverse=True,
            )
            return [self._session_record_locked(chat_id) for chat_id in session_ids]

    async def import_sessions(
        self,
        sessions: list[PersistedSession],
        *,
        principal_id: str,
        mode: Literal["replace", "merge"],
    ) -> int:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            if mode == "replace":
                owned_chat_ids = [
                    chat_id
                    for chat_id, state in self._chats.items()
                    if (
                        (workspace := self._workspaces.get(state.workspace_id)) is not None
                        and workspace.principal_id == normalized_principal
                    )
                ]
                owned_workspace_ids = [
                    workspace_id
                    for workspace_id, workspace in self._workspaces.items()
                    if workspace.principal_id == normalized_principal
                ]
                self._storage.delete_sessions(owned_chat_ids)
                self._storage.delete_workspace_activity(owned_workspace_ids)
                self._storage.delete_workspaces(owned_workspace_ids)
            for session in sessions:
                normalized_session = PersistedSession(
                    session_id=session.session_id,
                    principal_id=normalized_principal,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    status=session.status,
                    decision=(
                        dict(session.decision) if isinstance(session.decision, dict) else None
                    ),
                    messages=[dict(item) for item in session.messages],
                    model_provider=session.model_provider,
                    model_id=session.model_id,
                    title_override=session.title_override,
                    folder_id=session.folder_id,
                    output_text=session.output_text,
                    output_updated_at=session.output_updated_at,
                    files=list(session.files),
                    artifacts=[dict(item) for item in session.artifacts],
                    workspace_root=session.workspace_root,
                    policy_profile=session.policy_profile,
                    yolo_armed=session.yolo_armed,
                    yolo_armed_at=session.yolo_armed_at,
                    tools_state=(
                        dict(session.tools_state) if isinstance(session.tools_state, dict) else None
                    ),
                    mode=session.mode,
                    active_plan=(
                        dict(session.active_plan) if isinstance(session.active_plan, dict) else None
                    ),
                    active_task=(
                        dict(session.active_task) if isinstance(session.active_task, dict) else None
                    ),
                    auto_state=(
                        dict(session.auto_state) if isinstance(session.auto_state, dict) else None
                    ),
                )
                self._storage.save_session(normalized_session)
            self._workspaces = {}
            self._chats = {}
            self._workspace_activity = {}
            self._restore_state()
            return len(sessions)

    async def get_session_access(self, session_id: str, principal_id: str) -> SessionAccess:
        access = await self.get_chat_access(session_id, principal_id)
        return access

    async def get_session(self, session_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._chats.get(session_id)
            if state is None:
                return None
            workspace = self._workspaces.get(state.workspace_id)
            chat_messages = [
                dict(message)
                for message in state.messages
                if _normalize_message_lane(message.get("lane"), default="chat") == "chat"
            ]
            workspace_messages = [
                dict(message)
                for message in state.messages
                if _normalize_message_lane(message.get("lane"), default="chat") == "workspace"
            ]
            last_lane: MessageLane | None = None
            if state.messages:
                last_lane = _normalize_message_lane(
                    state.messages[-1].get("lane"),
                    default="chat",
                )
            selected_model: dict[str, JSONValue] | None = None
            if state.model_provider and state.model_id:
                selected_model = {
                    "provider": state.model_provider,
                    "model": state.model_id,
                }
            return {
                "session_id": session_id,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "status": state.runtime.status_state,
                "messages": chat_messages,
                "workspace_messages": workspace_messages,
                "lane_stats": {
                    "chat_message_count": len(chat_messages),
                    "workspace_message_count": len(workspace_messages),
                    "last_message_lane": last_lane,
                },
                "output": {
                    "content": state.output_text,
                    "updated_at": state.output_updated_at,
                },
                "files": list(state.files),
                "artifacts": [dict(item) for item in state.artifacts],
                "decision": (
                    dict(state.runtime.decision)
                    if isinstance(state.runtime.decision, dict)
                    else None
                ),
                "selected_model": selected_model,
                "title_override": state.title,
                "folder_id": None,
                "workspace_root": workspace.root_path if workspace is not None else None,
                "policy": {
                    "profile": workspace.policy_profile if workspace is not None else "sandbox",
                    "yolo_armed": workspace.yolo_armed if workspace is not None else False,
                    "yolo_armed_at": workspace.yolo_armed_at if workspace is not None else None,
                },
                "tools_state": (
                    dict(workspace.tools_state)
                    if workspace is not None and isinstance(workspace.tools_state, dict)
                    else None
                ),
                "mode": state.runtime.mode,
                "active_plan": (
                    dict(state.runtime.active_plan)
                    if isinstance(state.runtime.active_plan, dict)
                    else None
                ),
                "active_task": (
                    dict(state.runtime.active_task)
                    if isinstance(state.runtime.active_task, dict)
                    else None
                ),
                "auto_state": (
                    dict(state.runtime.auto_state)
                    if isinstance(state.runtime.auto_state, dict)
                    else None
                ),
            }

    def _session_record_locked(self, chat_id: str) -> PersistedSession:
        state = self._require_chat_locked(chat_id)
        workspace = self._workspaces.get(state.workspace_id)
        return PersistedSession(
            session_id=chat_id,
            principal_id=workspace.principal_id if workspace is not None else None,
            created_at=state.created_at,
            updated_at=state.updated_at,
            status=state.runtime.status_state,
            decision=(
                dict(state.runtime.decision) if isinstance(state.runtime.decision, dict) else None
            ),
            messages=[dict(message) for message in state.messages],
            model_provider=state.model_provider,
            model_id=state.model_id,
            title_override=state.title,
            folder_id=None,
            output_text=state.output_text,
            output_updated_at=state.output_updated_at,
            files=list(state.files),
            artifacts=[dict(item) for item in state.artifacts],
            workspace_root=workspace.root_path if workspace is not None else None,
            policy_profile=workspace.policy_profile if workspace is not None else None,
            yolo_armed=workspace.yolo_armed if workspace is not None else False,
            yolo_armed_at=workspace.yolo_armed_at if workspace is not None else None,
            tools_state=(
                dict(workspace.tools_state)
                if workspace is not None and isinstance(workspace.tools_state, dict)
                else None
            ),
            mode=state.runtime.mode,
            active_plan=(
                dict(state.runtime.active_plan)
                if isinstance(state.runtime.active_plan, dict)
                else None
            ),
            active_task=(
                dict(state.runtime.active_task)
                if isinstance(state.runtime.active_task, dict)
                else None
            ),
            auto_state=(
                dict(state.runtime.auto_state)
                if isinstance(state.runtime.auto_state, dict)
                else None
            ),
        )

    async def delete_session(self, session_id: str) -> bool:
        return await self.delete_chat(session_id)

    async def set_session_title(self, session_id: str, title: str) -> dict[str, JSONValue]:
        return await self.set_chat_title(session_id, title)

    async def list_folders(self) -> list[dict[str, str]]:
        return []

    async def create_folder(self, name: str) -> dict[str, str]:
        normalized = name.strip()
        if not normalized:
            raise ValueError("name required")
        return {"folder_id": uuid.uuid4().hex, "name": normalized}

    async def set_session_folder(self, session_id: str, folder_id: str | None) -> None:
        del session_id, folder_id

    async def list_chats(self, workspace_id: str) -> list[ChatListItem]:
        async with self._lock:
            items: list[ChatListItem] = []
            for chat_id, state in self._chats.items():
                if state.workspace_id != workspace_id:
                    continue
                items.append(
                    {
                        "chat_id": chat_id,
                        "workspace_id": workspace_id,
                        "title": state.title,
                        "created_at": state.created_at,
                        "updated_at": state.updated_at,
                        "message_count": len(state.messages),
                    }
                )
            items.sort(key=lambda item: self._parse_timestamp(item["updated_at"]), reverse=True)
            return items

    async def get_chat(self, chat_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None:
                return None
            return self._chat_payload(chat_id, state)

    async def get_chat_access(self, chat_id: str, principal_id: str) -> EntityAccess:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None:
                return "missing"
            workspace = self._workspaces.get(state.workspace_id)
            if workspace is None:
                return "missing"
            if workspace.principal_id != normalized_principal:
                return "forbidden"
            return "owned"

    async def delete_chat(self, chat_id: str) -> bool:
        async with self._lock:
            if chat_id not in self._chats:
                return False
            workspace_id = self._chats[chat_id].workspace_id
            self._drop_chat_locked(chat_id)
            self._touch_workspace_locked(workspace_id)
            return True

    async def set_chat_title(self, chat_id: str, title: str) -> dict[str, JSONValue]:
        normalized = title.strip()
        if not normalized:
            raise ValueError("title required")
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            state.title = normalized
            state.updated_at = _utc_iso_now()
            self._persist_chat_locked(chat_id)
            self._touch_workspace_locked(state.workspace_id)
            return {"chat_id": chat_id, "title": normalized}

    async def get_messages(self, chat_id: str, *, lane: str = "chat") -> list[dict[str, JSONValue]]:
        return await self.get_session_history(chat_id, lane=lane) or []

    async def get_chat_messages(self, chat_id: str) -> list[dict[str, JSONValue]]:
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None:
                return []
            return [dict(message) for message in state.messages]

    async def get_session_history(
        self,
        session_id: str,
        *,
        lane: str = "chat",
    ) -> list[dict[str, JSONValue]] | None:
        normalized_lane = _normalize_message_lane(lane, default="chat")
        async with self._lock:
            state = self._chats.get(session_id)
            if state is None:
                return None
            return [
                dict(message)
                for message in state.messages
                if _normalize_message_lane(message.get("lane"), default="chat") == normalized_lane
            ]

    async def append_message(
        self,
        chat_id: str,
        message: dict[str, JSONValue],
        *,
        lane: str = "chat",
    ) -> dict[str, JSONValue]:
        normalized = _normalize_message_payload({**message, "lane": lane})
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            state.messages.append(normalized)
            if (
                state.title == DEFAULT_CHAT_TITLE
                and normalized.get("role") == "user"
                and _normalize_message_lane(normalized.get("lane"), default="chat") == "chat"
            ):
                content = str(normalized.get("content") or "").strip()
                if content:
                    state.title = content[:120]
            state.updated_at = _utc_iso_now()
            self._persist_chat_locked(chat_id)
            self._persist_messages_locked(chat_id)
            self._touch_workspace_locked(state.workspace_id)
            return dict(normalized)

    async def get_session_output(self, chat_id: str) -> dict[str, str | None] | None:
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None:
                return None
            return {"content": state.output_text, "updated_at": state.output_updated_at}

    async def set_session_output(self, chat_id: str, content: str | None) -> None:
        normalized = content.strip() if isinstance(content, str) else ""
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            state.output_text = normalized or None
            state.output_updated_at = _utc_iso_now() if normalized else None
            state.updated_at = _utc_iso_now()
            self._persist_chat_locked(chat_id)
            self._touch_workspace_locked(state.workspace_id)

    async def get_session_files(self, chat_id: str) -> list[str] | None:
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None:
                return None
            return list(state.files)

    async def merge_session_files(self, chat_id: str, paths: list[str]) -> list[str]:
        normalized = [item.strip() for item in paths if isinstance(item, str) and item.strip()]
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            existing = list(state.files)
            seen = set(existing)
            for path in normalized:
                if path not in seen:
                    existing.append(path)
                    seen.add(path)
            state.files = existing
            state.updated_at = _utc_iso_now()
            self._persist_chat_locked(chat_id)
            self._touch_workspace_locked(state.workspace_id)
            return list(existing)

    async def get_session_artifacts(self, chat_id: str) -> list[dict[str, JSONValue]] | None:
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None:
                return None
            return [dict(item) for item in state.artifacts]

    async def append_session_artifact(
        self,
        chat_id: str,
        artifact: dict[str, JSONValue],
    ) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            normalized = dict(artifact)
            state.artifacts.append(normalized)
            state.updated_at = _utc_iso_now()
            self._persist_chat_locked(chat_id)
            self._touch_workspace_locked(state.workspace_id)
            return normalized

    async def get_workspace_activity(self, workspace_id: str) -> list[dict[str, JSONValue]]:
        async with self._lock:
            return [
                {
                    "activity_id": item.activity_id,
                    "workspace_id": item.workspace_id,
                    "kind": item.kind,
                    "summary": item.summary,
                    "payload": dict(item.payload),
                    "created_at": item.created_at,
                }
                for item in self._activity_for_workspace_locked(workspace_id)
            ]

    async def append_workspace_activity(
        self,
        workspace_id: str,
        *,
        kind: str,
        summary: str,
        payload: dict[str, JSONValue] | None = None,
    ) -> dict[str, JSONValue]:
        async with self._lock:
            self._require_workspace_locked(workspace_id)
            record = WorkspaceActivityRecord(
                activity_id=uuid.uuid4().hex,
                workspace_id=workspace_id,
                kind=kind.strip(),
                summary=summary.strip(),
                payload=dict(payload or {}),
                created_at=_utc_iso_now(),
            )
            self._storage.append_workspace_activity(record)
            self._workspace_activity.setdefault(workspace_id, []).append(record)
            self._touch_workspace_locked(workspace_id)
            return {
                "activity_id": record.activity_id,
                "workspace_id": record.workspace_id,
                "kind": record.kind,
                "summary": record.summary,
                "payload": dict(record.payload),
                "created_at": record.created_at,
            }

    async def get_workspace_decision(self, workspace_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._workspaces.get(workspace_id)
            if state is None or state.pending_decision is None:
                return None
            return dict(state.pending_decision)

    async def set_workspace_decision(
        self,
        workspace_id: str,
        decision: dict[str, JSONValue] | None,
    ) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._require_workspace_locked(workspace_id)
            state.pending_decision = dict(decision) if isinstance(decision, dict) else None
            state.updated_at = _utc_iso_now()
            self._persist_workspace_locked(workspace_id)
            return dict(state.pending_decision) if state.pending_decision is not None else None

    async def transition_workspace_decision(
        self,
        workspace_id: str,
        *,
        expected_id: str,
        expected_status: str,
        next_decision: dict[str, JSONValue] | None,
    ) -> tuple[bool, dict[str, JSONValue] | None]:
        async with self._lock:
            state = self._require_workspace_locked(workspace_id)
            current = state.pending_decision
            if not isinstance(current, dict):
                return False, None
            current_id = current.get("id")
            current_status = current.get("status")
            if current_id != expected_id or current_status != expected_status:
                return False, dict(current)
            state.pending_decision = (
                dict(next_decision) if isinstance(next_decision, dict) else None
            )
            state.updated_at = _utc_iso_now()
            self._persist_workspace_locked(workspace_id)
            return True, (
                dict(state.pending_decision) if state.pending_decision is not None else None
            )

    async def get_workspace_root(self, ref_id: str) -> str | None:
        async with self._lock:
            workspace = self._workspace_for_ref_locked(ref_id)
            return workspace.root_path if workspace is not None else None

    async def set_workspace_root(self, ref_id: str, root_path: str | None) -> None:
        normalized = root_path.strip() if isinstance(root_path, str) and root_path.strip() else None
        async with self._lock:
            workspace_id, workspace = self._require_workspace_from_ref_locked(ref_id)
            workspace.root_path = normalized
            workspace.updated_at = _utc_iso_now()
            self._persist_workspace_locked(workspace_id)

    async def get_session_policy(self, ref_id: str) -> dict[str, JSONValue]:
        async with self._lock:
            _, state = self._require_workspace_from_ref_locked(ref_id)
            return {
                "profile": state.policy_profile,
                "yolo_armed": state.yolo_armed,
                "yolo_armed_at": state.yolo_armed_at,
            }

    async def set_session_policy(
        self,
        ref_id: str,
        *,
        profile: str | None = None,
        yolo_armed: bool | None = None,
        yolo_armed_at: str | None | object = _UNSET,
    ) -> dict[str, JSONValue]:
        async with self._lock:
            workspace_id, state = self._require_workspace_from_ref_locked(ref_id)
            if profile is not None:
                state.policy_profile = self._normalize_policy_profile(profile)
            if yolo_armed is not None:
                state.yolo_armed = bool(yolo_armed)
            if yolo_armed_at is not _UNSET:
                state.yolo_armed_at = cast("str | None", yolo_armed_at)
            if not state.yolo_armed:
                state.yolo_armed_at = None
            state.updated_at = _utc_iso_now()
            self._persist_workspace_locked(workspace_id)
            return {
                "profile": state.policy_profile,
                "yolo_armed": state.yolo_armed,
                "yolo_armed_at": state.yolo_armed_at,
            }

    async def consume_yolo_once(self, ref_id: str) -> bool:
        async with self._lock:
            workspace_id, state = self._require_workspace_from_ref_locked(ref_id)
            if not state.yolo_armed:
                return False
            state.yolo_armed = False
            state.yolo_armed_at = None
            state.updated_at = _utc_iso_now()
            self._persist_workspace_locked(workspace_id)
            return True

    async def get_session_tools_state(self, ref_id: str) -> dict[str, bool] | None:
        async with self._lock:
            _, state = self._require_workspace_from_ref_locked(ref_id)
            return dict(state.tools_state) if isinstance(state.tools_state, dict) else None

    async def set_session_tools_state(
        self,
        ref_id: str,
        tools_state: dict[str, bool] | None,
        *,
        merge: bool = False,
    ) -> dict[str, bool] | None:
        async with self._lock:
            workspace_id, state = self._require_workspace_from_ref_locked(ref_id)
            next_tools = dict(tools_state) if isinstance(tools_state, dict) else None
            if merge and isinstance(state.tools_state, dict) and isinstance(next_tools, dict):
                merged = dict(state.tools_state)
                merged.update(next_tools)
                state.tools_state = merged
            else:
                state.tools_state = next_tools
            state.updated_at = _utc_iso_now()
            self._persist_workspace_locked(workspace_id)
            return dict(state.tools_state) if isinstance(state.tools_state, dict) else None

    async def set_session_model(self, chat_id: str, provider: str, model_id: str) -> None:
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            state.model_provider = provider.strip()
            state.model_id = model_id.strip()
            state.updated_at = _utc_iso_now()
            self._persist_chat_locked(chat_id)
            self._touch_workspace_locked(state.workspace_id)

    async def get_session_model(self, chat_id: str) -> dict[str, str] | None:
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None or state.model_provider is None or state.model_id is None:
                return None
            return {"provider": state.model_provider, "model": state.model_id}

    async def set_session_status(self, chat_id: str, status_state: str) -> None:
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            state.runtime.status_state = self._normalize_status(status_state)
            state.runtime.updated_at = _utc_iso_now()
            self._persist_runtime_locked(chat_id)
            self._publish_locked(
                chat_id,
                self._build_event(
                    "status",
                    {
                        "session_id": chat_id,
                        "chat_id": chat_id,
                        "state": state.runtime.status_state,
                        "status": state.runtime.status_state,
                        "ok": state.runtime.status_state != "error",
                    },
                ),
            )

    async def get_session_status_event(self, chat_id: str) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._require_or_create_chat_locked(chat_id)
            return self._build_event(
                "status",
                {
                    "session_id": chat_id,
                    "chat_id": chat_id,
                    "state": state.runtime.status_state,
                    "status": state.runtime.status_state,
                    "ok": state.runtime.status_state != "error",
                },
            )

    async def get_session_workflow_event(self, chat_id: str) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._require_or_create_chat_locked(chat_id)
            runtime = state.runtime
            return self._build_event(
                "session.workflow",
                {
                    "session_id": chat_id,
                    "chat_id": chat_id,
                    "mode": runtime.mode,
                    "active_plan": runtime.active_plan,
                    "active_task": runtime.active_task,
                    "auto_state": runtime.auto_state,
                    "decision": runtime.decision,
                },
            )

    async def get_events_since(
        self,
        chat_id: str,
        after_event_id: str | None,
    ) -> list[dict[str, JSONValue]]:
        async with self._lock:
            state = self._require_or_create_chat_locked(chat_id)
            records = list(state.event_buffer)
            if not after_event_id:
                return [dict(item.event) for item in records]
            matched = False
            result: list[dict[str, JSONValue]] = []
            for item in records:
                if matched:
                    result.append(dict(item.event))
                elif item.event_id == after_event_id:
                    matched = True
            if matched:
                return result
            return [self._resync_event(chat_id, reason="last_event_id_stale")]

    async def get_session_workflow(self, chat_id: str) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None:
                return {"mode": "ask", "active_plan": None, "active_task": None, "auto_state": None}
            runtime = state.runtime
            return {
                "mode": runtime.mode,
                "active_plan": (
                    dict(runtime.active_plan) if runtime.active_plan is not None else None
                ),
                "active_task": (
                    dict(runtime.active_task) if runtime.active_task is not None else None
                ),
                "auto_state": (
                    dict(runtime.auto_state) if isinstance(runtime.auto_state, dict) else None
                ),
            }

    async def start_plan_task_if_possible(
        self,
        chat_id: str,
        *,
        expected_mode: str,
        expected_plan_id: str,
        expected_plan_hash: str,
        expected_plan_revision: int,
        running_plan: dict[str, JSONValue],
        task_payload: dict[str, JSONValue],
        next_mode: str,
    ) -> tuple[bool, dict[str, JSONValue] | None, str | None]:
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            runtime = state.runtime
            if runtime.mode != expected_mode:
                return False, None, "mode_mismatch"
            plan = runtime.active_plan
            if not isinstance(plan, dict):
                return False, None, "plan_mismatch"
            if (
                plan.get("plan_id") != expected_plan_id
                or plan.get("plan_hash") != expected_plan_hash
                or plan.get("plan_revision") != expected_plan_revision
            ):
                return False, None, "plan_mismatch"
            runtime.mode = self._normalize_mode(next_mode)
            runtime.active_plan = dict(running_plan)
            runtime.active_task = dict(task_payload)
            runtime.updated_at = _utc_iso_now()
            state.updated_at = runtime.updated_at
            self._persist_chat_locked(chat_id)
            self._persist_runtime_locked(chat_id)
            return True, dict(runtime.active_task), None

    async def set_session_workflow(
        self,
        chat_id: str,
        *,
        mode: str | None = None,
        active_plan: dict[str, JSONValue] | None | object = _UNSET,
        active_task: dict[str, JSONValue] | None | object = _UNSET,
        auto_state: dict[str, JSONValue] | None | object = _UNSET,
    ) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            runtime = state.runtime
            if mode is not None:
                runtime.mode = self._normalize_mode(mode)
            if active_plan is not _UNSET:
                runtime.active_plan = (
                    dict(cast("dict[str, JSONValue]", active_plan))
                    if isinstance(active_plan, dict)
                    else None
                )
            if active_task is not _UNSET:
                runtime.active_task = (
                    dict(cast("dict[str, JSONValue]", active_task))
                    if isinstance(active_task, dict)
                    else None
                )
            if auto_state is not _UNSET:
                runtime.auto_state = (
                    normalize_auto_state(cast("dict[str, JSONValue] | None", auto_state))
                    if isinstance(auto_state, dict)
                    else None
                )
            runtime.updated_at = _utc_iso_now()
            state.updated_at = runtime.updated_at
            self._persist_chat_locked(chat_id)
            self._persist_runtime_locked(chat_id)
            self._publish_locked(
                chat_id,
                self._build_event(
                    "session.workflow",
                    {
                        "session_id": chat_id,
                        "chat_id": chat_id,
                        "mode": runtime.mode,
                        "active_plan": runtime.active_plan,
                        "active_task": runtime.active_task,
                        "auto_state": runtime.auto_state,
                        "decision": runtime.decision,
                    },
                ),
            )
            return {
                "mode": runtime.mode,
                "active_plan": runtime.active_plan,
                "active_task": runtime.active_task,
                "auto_state": runtime.auto_state,
            }

    async def get_session_decision(self, chat_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None or state.runtime.decision is None:
                return None
            return dict(state.runtime.decision)

    async def set_session_decision(
        self,
        chat_id: str,
        decision: dict[str, JSONValue] | None,
    ) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            state.runtime.decision = dict(decision) if isinstance(decision, dict) else None
            state.runtime.updated_at = _utc_iso_now()
            state.updated_at = state.runtime.updated_at
            self._persist_chat_locked(chat_id)
            self._persist_runtime_locked(chat_id)
            self._publish_locked(
                chat_id,
                self._build_event(
                    "decision.packet",
                    {
                        "session_id": chat_id,
                        "chat_id": chat_id,
                        "decision": (
                            dict(state.runtime.decision)
                            if isinstance(state.runtime.decision, dict)
                            else None
                        ),
                    },
                ),
            )
            return dict(state.runtime.decision) if state.runtime.decision is not None else None

    async def transition_session_decision(
        self,
        chat_id: str,
        *,
        expected_id: str,
        expected_status: str,
        next_decision: dict[str, JSONValue] | None,
    ) -> tuple[bool, dict[str, JSONValue] | None]:
        async with self._lock:
            state = self._require_chat_locked(chat_id)
            current = state.runtime.decision
            if not isinstance(current, dict):
                return False, None
            if current.get("id") != expected_id or current.get("status") != expected_status:
                return False, dict(current)
            state.runtime.decision = (
                dict(next_decision) if isinstance(next_decision, dict) else None
            )
            state.runtime.updated_at = _utc_iso_now()
            state.updated_at = state.runtime.updated_at
            self._persist_chat_locked(chat_id)
            self._persist_runtime_locked(chat_id)
            self._publish_locked(
                chat_id,
                self._build_event(
                    "decision.packet",
                    {
                        "session_id": chat_id,
                        "chat_id": chat_id,
                        "decision": (
                            dict(state.runtime.decision)
                            if isinstance(state.runtime.decision, dict)
                            else None
                        ),
                    },
                ),
            )
            return True, (
                dict(state.runtime.decision) if state.runtime.decision is not None else None
            )

    async def subscribe(self, chat_id: str) -> asyncio.Queue[dict[str, JSONValue]]:
        async with self._lock:
            state = self._require_or_create_chat_locked(chat_id)
            queue: asyncio.Queue[dict[str, JSONValue]] = asyncio.Queue(
                maxsize=self._subscriber_queue_maxsize
            )
            state.subscribers[queue] = _SubscriberState(queue=queue)
            return queue

    async def unsubscribe(self, chat_id: str, queue: asyncio.Queue[dict[str, JSONValue]]) -> None:
        async with self._lock:
            state = self._chats.get(chat_id)
            if state is None:
                return
            state.subscribers.pop(queue, None)

    async def publish(self, chat_id: str, event: dict[str, JSONValue]) -> None:
        async with self._lock:
            self._require_or_create_chat_locked(chat_id)
            normalized = self._ensure_event_fields(event)
            self._publish_locked(chat_id, normalized)

    def create_message(
        self,
        *,
        role: str,
        content: str,
        lane: str = "chat",
        trace_id: str | None = None,
        parent_user_message_id: str | None = None,
        attachments: list[dict[str, str]] | None = None,
        message_id: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, JSONValue]:
        return _build_message(
            role=role,
            content=content,
            lane=lane,
            trace_id=trace_id,
            parent_user_message_id=parent_user_message_id,
            attachments=attachments,
            message_id=message_id,
            created_at=created_at,
        )

    def _restore_state(self) -> None:
        for workspace in self._storage.load_workspaces():
            policy = dict(workspace.policy)
            self._workspaces[workspace.workspace_id] = _WorkspaceState(
                principal_id=self._normalize_principal_id(workspace.principal_id),
                title=workspace.title,
                root_path=workspace.root_path,
                policy_profile=self._normalize_policy_profile(policy.get("profile")),
                yolo_armed=policy.get("yolo_armed") is True,
                yolo_armed_at=cast("str | None", policy.get("yolo_armed_at"))
                if isinstance(policy.get("yolo_armed_at"), str)
                else None,
                tools_state=(
                    dict(workspace.tools_state) if isinstance(workspace.tools_state, dict) else None
                ),
                pending_decision=dict(workspace.pending_decision)
                if isinstance(workspace.pending_decision, dict)
                else None,
                created_at=workspace.created_at,
                updated_at=workspace.updated_at,
            )
        for chat in self._storage.load_chats():
            self._chats[chat.chat_id] = _ChatState(
                workspace_id=chat.workspace_id,
                title=chat.title,
                artifacts=[dict(item) for item in chat.artifacts],
                output_text=chat.output_text,
                output_updated_at=chat.output_updated_at,
                files=list(chat.files),
                model_provider=chat.selected_model_provider,
                model_id=chat.selected_model_id,
                created_at=chat.created_at,
                updated_at=chat.updated_at,
            )
        for runtime in self._storage.load_runtime_states():
            chat_state = self._chats.get(runtime.chat_id)
            if chat_state is None:
                continue
            chat_state.runtime = _RuntimeState(
                mode=self._normalize_mode(runtime.mode),
                active_plan=(
                    dict(runtime.active_plan) if isinstance(runtime.active_plan, dict) else None
                ),
                active_task=(
                    dict(runtime.active_task) if isinstance(runtime.active_task, dict) else None
                ),
                auto_state=normalize_auto_state(runtime.auto_state),
                decision=dict(runtime.decision) if isinstance(runtime.decision, dict) else None,
                status_state=self._normalize_status(runtime.status),
                updated_at=runtime.updated_at,
            )
        for message in self._storage.load_messages():
            chat_state = self._chats.get(message.chat_id)
            if chat_state is None:
                continue
            chat_state.messages.append(
                {
                    "message_id": message.message_id,
                    "role": message.role,
                    "lane": message.lane,
                    "content": message.content,
                    "created_at": message.created_at,
                    "trace_id": message.trace_id,
                    "parent_user_message_id": message.parent_user_message_id,
                    "attachments": [dict(item) for item in message.attachments],
                }
            )
        for chat_state in self._chats.values():
            chat_state.messages.sort(key=lambda item: str(item.get("created_at") or ""))
        self._workspace_activity = {}
        for activity in self._storage.load_workspace_activity():
            self._workspace_activity.setdefault(activity.workspace_id, []).append(activity)

    def _normalize_tools_state_payload(self, value: object) -> dict[str, bool]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, bool] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.strip() and isinstance(item, bool):
                normalized[key.strip()] = item
        return normalized

    def _normalize_plan_payload(self, value: object) -> dict[str, JSONValue] | None:
        return dict(value) if isinstance(value, dict) else None

    def _normalize_task_payload(self, value: object) -> dict[str, JSONValue] | None:
        return dict(value) if isinstance(value, dict) else None

    def _normalize_auto_state_payload(self, value: object) -> dict[str, JSONValue] | None:
        return normalize_auto_state(cast("dict[str, JSONValue] | None", value))

    def _persist_workspace_locked(self, workspace_id: str) -> None:
        state = self._require_workspace_locked(workspace_id)
        self._storage.save_workspace(
            WorkspaceRecord(
                workspace_id=workspace_id,
                principal_id=state.principal_id,
                title=state.title,
                root_path=state.root_path,
                policy={
                    "profile": state.policy_profile,
                    "yolo_armed": state.yolo_armed,
                    "yolo_armed_at": state.yolo_armed_at,
                },
                tools_state=(
                    dict(state.tools_state) if isinstance(state.tools_state, dict) else None
                ),
                pending_decision=(
                    dict(state.pending_decision)
                    if isinstance(state.pending_decision, dict)
                    else None
                ),
                created_at=state.created_at,
                updated_at=state.updated_at,
            )
        )

    def _persist_chat_locked(self, chat_id: str) -> None:
        state = self._require_chat_locked(chat_id)
        self._storage.save_chat(
            ChatRecord(
                chat_id=chat_id,
                workspace_id=state.workspace_id,
                title=state.title,
                selected_model_provider=state.model_provider,
                selected_model_id=state.model_id,
                artifacts=[dict(item) for item in state.artifacts],
                output_text=state.output_text,
                output_updated_at=state.output_updated_at,
                files=list(state.files),
                created_at=state.created_at,
                updated_at=state.updated_at,
            )
        )

    def _persist_messages_locked(self, chat_id: str) -> None:
        state = self._require_chat_locked(chat_id)
        self._storage.replace_chat_messages(
            chat_id,
            [
                MessageRecord(
                    message_id=str(message["message_id"]),
                    chat_id=chat_id,
                    lane=_normalize_message_lane(message.get("lane"), default="chat"),
                    role=str(message["role"]),
                    content=str(message["content"]),
                    created_at=str(message["created_at"]),
                    trace_id=cast("str | None", message.get("trace_id")),
                    parent_user_message_id=cast(
                        "str | None",
                        message.get("parent_user_message_id"),
                    ),
                    attachments=[
                        dict(item)
                        for item in cast(
                            "list[dict[str, str]]",
                            message.get("attachments") or [],
                        )
                    ],
                )
                for message in state.messages
            ],
        )

    def _persist_runtime_locked(self, chat_id: str) -> None:
        state = self._require_chat_locked(chat_id)
        runtime = state.runtime
        self._storage.save_runtime_state(
            RuntimeStateRecord(
                chat_id=chat_id,
                mode=runtime.mode,
                active_plan=(
                    dict(runtime.active_plan) if isinstance(runtime.active_plan, dict) else None
                ),
                active_task=(
                    dict(runtime.active_task) if isinstance(runtime.active_task, dict) else None
                ),
                auto_state=(
                    dict(runtime.auto_state) if isinstance(runtime.auto_state, dict) else None
                ),
                decision=(dict(runtime.decision) if isinstance(runtime.decision, dict) else None),
                status=runtime.status_state,
                updated_at=runtime.updated_at,
            )
        )

    def _require_or_create_chat_locked(self, chat_id: str) -> _ChatState:
        state = self._chats.get(chat_id)
        if state is not None:
            return state
        self._create_session_locked(DEFAULT_LEGACY_PRINCIPAL_ID, session_id=chat_id)
        return self._require_chat_locked(chat_id)

    def _resync_event(self, chat_id: str, *, reason: str) -> dict[str, JSONValue]:
        return self._build_event(
            "session.resync_required",
            {
                "session_id": chat_id,
                "chat_id": chat_id,
                "resync_only": True,
                "reason": reason,
            },
        )

    def _publish_locked(self, chat_id: str, event: dict[str, JSONValue]) -> None:
        state = self._require_or_create_chat_locked(chat_id)
        normalized = self._ensure_event_fields(event)
        self._append_event_record_locked(state, normalized)
        for subscriber in list(state.subscribers.values()):
            self._enqueue_subscriber_event_locked(chat_id, subscriber, normalized)

    def _enqueue_subscriber_event_locked(
        self,
        chat_id: str,
        subscriber: _SubscriberState,
        event: dict[str, JSONValue],
    ) -> None:
        self._flush_pending_resync_locked(chat_id, subscriber)
        if self._try_put_nowait(subscriber.queue, event):
            return
        if self._subscriber_drop_policy == "coalesce_deltas" and self._coalesce_delta_locked(
            subscriber.queue, event
        ):
            return
        event_type = str(event.get("type") or "")
        if event_type in _CONTROL_EVENT_TYPES and self._evict_oldest_non_control_locked(
            subscriber.queue,
        ):
            self._try_put_nowait(subscriber.queue, event)
            return
        if event_type not in _CONTROL_EVENT_TYPES and self._evict_oldest_non_control_locked(
            subscriber.queue,
        ):
            self._try_put_nowait(subscriber.queue, event)
            return
        subscriber.pending_resync_reason = "queue_overflow"
        self._flush_pending_resync_locked(chat_id, subscriber)

    def _flush_pending_resync_locked(self, chat_id: str, subscriber: _SubscriberState) -> None:
        reason = subscriber.pending_resync_reason
        if reason is None:
            return
        resync_event = self._resync_event(chat_id, reason=reason)
        if self._try_put_nowait(subscriber.queue, resync_event):
            subscriber.pending_resync_reason = None
            return
        if self._evict_oldest_non_control_locked(subscriber.queue) and self._try_put_nowait(
            subscriber.queue,
            resync_event,
        ):
            subscriber.pending_resync_reason = None

    def _try_put_nowait(
        self,
        queue: asyncio.Queue[dict[str, JSONValue]],
        event: dict[str, JSONValue],
    ) -> bool:
        try:
            queue.put_nowait(dict(event))
            return True
        except asyncio.QueueFull:
            return False

    def _coalesce_delta_locked(
        self,
        queue: asyncio.Queue[dict[str, JSONValue]],
        event: dict[str, JSONValue],
    ) -> bool:
        if event.get("type") != "chat.stream.delta":
            return False
        queued = list(queue._queue)  # type: ignore[attr-defined]
        if not queued:
            return False
        last = queued[-1]
        if not isinstance(last, dict) or last.get("type") != "chat.stream.delta":
            return False
        last_payload = last.get("payload")
        next_payload = event.get("payload")
        if not isinstance(last_payload, dict) or not isinstance(next_payload, dict):
            return False
        for key in ("session_id", "stream_id", "mode", "lane"):
            if last_payload.get(key) != next_payload.get(key):
                return False
        last_delta = last_payload.get("delta")
        next_delta = next_payload.get("delta")
        if not isinstance(last_delta, str) or not isinstance(next_delta, str):
            return False
        merged_payload = dict(last_payload)
        merged_payload["delta"] = last_delta + next_delta
        queued[-1] = {**last, "payload": merged_payload}
        queue._queue.clear()  # type: ignore[attr-defined]
        for item in queued:
            queue._queue.append(item)  # type: ignore[attr-defined]
        return True

    def _evict_oldest_non_control_locked(
        self,
        queue: asyncio.Queue[dict[str, JSONValue]],
    ) -> bool:
        queued = list(queue._queue)  # type: ignore[attr-defined]
        for index, item in enumerate(queued):
            if not isinstance(item, dict):
                continue
            event_type = item.get("type")
            if not isinstance(event_type, str) or event_type not in _CONTROL_EVENT_TYPES:
                del queued[index]
                queue._queue.clear()  # type: ignore[attr-defined]
                for queued_item in queued:
                    queue._queue.append(queued_item)  # type: ignore[attr-defined]
                return True
        return False

    def _drop_chat_locked(self, chat_id: str) -> None:
        state = self._chats.pop(chat_id, None)
        if state is None:
            return
        self._storage.delete_messages_for_chats([chat_id])
        self._storage.delete_runtime_states([chat_id])
        self._storage.delete_chats([chat_id])

    def _touch_workspace_locked(self, workspace_id: str) -> None:
        workspace = self._require_workspace_locked(workspace_id)
        workspace.updated_at = _utc_iso_now()
        self._persist_workspace_locked(workspace_id)

    def _default_workspace_for_principal_locked(self, principal_id: str) -> str | None:
        candidates: list[tuple[str, _WorkspaceState]] = [
            (workspace_id, state)
            for workspace_id, state in self._workspaces.items()
            if state.principal_id == principal_id
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: self._parse_timestamp(item[1].updated_at),
            reverse=True,
        )
        return candidates[0][0]

    def _create_session_locked(self, principal_id: str, *, session_id: str | None = None) -> str:
        workspace_id = self._default_workspace_for_principal_locked(principal_id)
        if workspace_id is None:
            workspace_id = uuid.uuid4().hex
            now = _utc_iso_now()
            self._workspaces[workspace_id] = _WorkspaceState(
                principal_id=principal_id,
                title=DEFAULT_WORKSPACE_TITLE,
                created_at=now,
                updated_at=now,
            )
            self._persist_workspace_locked(workspace_id)
        chat_id = session_id or uuid.uuid4().hex
        now = _utc_iso_now()
        self._chats[chat_id] = _ChatState(
            workspace_id=workspace_id,
            title=DEFAULT_CHAT_TITLE,
            runtime=_RuntimeState(updated_at=now),
            created_at=now,
            updated_at=now,
        )
        self._persist_chat_locked(chat_id)
        self._persist_runtime_locked(chat_id)
        self._touch_workspace_locked(workspace_id)
        return chat_id

    def _chat_access_locked(self, chat_id: str, principal_id: str) -> EntityAccess:
        state = self._chats.get(chat_id)
        if state is None:
            return "missing"
        workspace = self._workspaces.get(state.workspace_id)
        if workspace is None:
            return "missing"
        if workspace.principal_id != principal_id:
            return "forbidden"
        return "owned"

    def _workspace_access_locked(self, workspace_id: str, principal_id: str) -> EntityAccess:
        workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            return "missing"
        if workspace.principal_id != principal_id:
            return "forbidden"
        return "owned"

    def _workspace_payload(
        self,
        workspace_id: str,
        state: _WorkspaceState,
    ) -> dict[str, JSONValue]:
        return {
            "workspace_id": workspace_id,
            "title": state.title,
            "root_path": state.root_path,
            "policy": {
                "profile": state.policy_profile,
                "yolo_armed": state.yolo_armed,
                "yolo_armed_at": state.yolo_armed_at,
            },
            "tools_state": (
                dict(state.tools_state) if isinstance(state.tools_state, dict) else None
            ),
            "pending_decision": (
                dict(state.pending_decision) if isinstance(state.pending_decision, dict) else None
            ),
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

    def _chat_payload(self, chat_id: str, state: _ChatState) -> dict[str, JSONValue]:
        return {
            "chat_id": chat_id,
            "workspace_id": state.workspace_id,
            "title": state.title,
            "selected_model": (
                {"provider": state.model_provider, "model": state.model_id}
                if state.model_provider is not None and state.model_id is not None
                else None
            ),
            "artifacts": [dict(item) for item in state.artifacts],
            "messages": [dict(item) for item in state.messages],
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

    def _workspace_for_ref_locked(self, ref_id: str) -> _WorkspaceState | None:
        if ref_id in self._workspaces:
            return self._workspaces.get(ref_id)
        chat = self._chats.get(ref_id)
        if chat is None:
            return None
        return self._workspaces.get(chat.workspace_id)

    def _require_workspace_from_ref_locked(self, ref_id: str) -> tuple[str, _WorkspaceState]:
        if ref_id in self._workspaces:
            return ref_id, self._require_workspace_locked(ref_id)
        chat = self._require_chat_locked(ref_id)
        return chat.workspace_id, self._require_workspace_locked(chat.workspace_id)

    def _require_workspace_locked(self, workspace_id: str) -> _WorkspaceState:
        state = self._workspaces.get(workspace_id)
        if state is None:
            raise KeyError("workspace not found")
        return state

    def _require_chat_locked(self, chat_id: str) -> _ChatState:
        state = self._chats.get(chat_id)
        if state is None:
            raise KeyError("chat not found")
        return state

    def _activity_for_workspace_locked(self, workspace_id: str) -> list[WorkspaceActivityRecord]:
        return list(self._workspace_activity.get(workspace_id, []))

    def _ensure_event_fields(self, event: dict[str, JSONValue]) -> dict[str, JSONValue]:
        normalized = dict(event)
        event_type = normalized.get("type")
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event.type required")
        normalized["type"] = event_type.strip()
        normalized.setdefault("recorded_at", _utc_iso_now())
        normalized.setdefault("event_id", uuid.uuid4().hex)
        normalized.setdefault("id", normalized["event_id"])
        return normalized

    def _build_event(self, event_type: str, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return self._ensure_event_fields({"type": event_type, "payload": payload})

    def _append_event_record_locked(
        self,
        state: _ChatState,
        event: dict[str, JSONValue],
    ) -> None:
        event_id = str(event["event_id"])
        recorded_at_raw = event.get("recorded_at")
        recorded_at = (
            self._parse_timestamp(recorded_at_raw)
            if isinstance(recorded_at_raw, str)
            else datetime.now(UTC)
        )
        state.event_buffer.append(
            _EventRecord(
                event_id=event_id,
                recorded_at=recorded_at,
                event=dict(event),
            )
        )
        while len(state.event_buffer) > self._event_buffer_size:
            state.event_buffer.popleft()

    def _normalize_status(self, status_state: str) -> SessionStatus:
        normalized = status_state.strip().lower()
        if normalized == "busy":
            return "busy"
        if normalized == "error":
            return "error"
        return "ok"

    def _normalize_policy_profile(self, value: object) -> PolicyProfile:
        normalized = value.strip().lower() if isinstance(value, str) else "sandbox"
        if normalized == "index":
            return "index"
        if normalized == "yolo":
            return "yolo"
        return "sandbox"

    def _normalize_principal_id(self, value: str | None) -> str:
        normalized = (
            value.strip()
            if isinstance(value, str) and value.strip()
            else DEFAULT_LEGACY_PRINCIPAL_ID
        )
        return normalized

    def _normalize_mode(self, value: object) -> SessionMode:
        normalized = value.strip().lower() if isinstance(value, str) else "ask"
        if normalized == "plan":
            return "plan"
        if normalized == "act":
            return "act"
        if normalized == "auto":
            return "auto"
        return "ask"

    def _parse_timestamp(self, value: str) -> datetime:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
