from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, TypedDict, cast

from server.ui_session_storage import (
    InMemoryUISessionStorage,
    PersistedFolder,
    PersistedSession,
    UISessionStorage,
)
from shared.auto_models import normalize_auto_state
from shared.models import JSONValue


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


SessionStatus = Literal["ok", "busy", "error"]
MessageLane = Literal["chat", "workspace"]
PolicyProfile = Literal["sandbox", "index", "yolo"]
SessionMode = Literal["ask", "plan", "act", "auto"]
SessionAccess = Literal["owned", "forbidden", "missing"]
_UNSET: object = object()
DEFAULT_LEGACY_PRINCIPAL_ID = "legacy"


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


class _SessionListSortableItem(TypedDict):
    session_id: str
    title: str
    created_at: str
    updated_at: datetime
    message_count: int
    chat_message_count: int
    workspace_message_count: int
    last_message_lane: MessageLane | None
    title_override: str | None
    folder_id: str | None


DEFAULT_SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_SESSIONS = 200
DEFAULT_MAX_MESSAGES_PER_SESSION = 500
DEFAULT_SUBSCRIBER_QUEUE_MAXSIZE = 256
SubscriberDropPolicy = Literal["drop_newest", "coalesce_deltas"]
DEFAULT_SUBSCRIBER_DROP_POLICY: SubscriberDropPolicy = "coalesce_deltas"
DEFAULT_EVENT_BUFFER_SIZE = 512
DEFAULT_EVENT_BUFFER_TTL_SECONDS = 10 * 60

_MESSAGE_ROLES = {"user", "assistant", "system"}
_MESSAGE_LANES = {"chat", "workspace"}
_DELTA_EVENT_TYPES = {"chat.stream.delta", "canvas.stream.delta"}
_CONTROL_EVENT_TYPES = {"decision.packet", "session.workflow", "status"}


def _normalize_message_lane(value: object, *, default: MessageLane = "chat") -> MessageLane:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "workspace":
            return "workspace"
        if normalized == "chat":
            return "chat"
    return default


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
    lane: MessageLane = "chat",
    trace_id: str | None = None,
    parent_user_message_id: str | None = None,
    attachments: list[dict[str, str]] | None = None,
    message_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, JSONValue]:
    normalized_role = role.strip()
    if normalized_role not in _MESSAGE_ROLES:
        raise ValueError(f"unsupported message role: {role}")
    normalized_lane = _normalize_message_lane(lane)
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
        "attachments": list(attachments or []),
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
    if lane is not None and (not isinstance(lane, str) or lane.strip() not in _MESSAGE_LANES):
        raise ValueError("lane must be chat|workspace")
    if not isinstance(content, str):
        raise ValueError("content required")
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
        "attachments": [
            {"name": item["name"], "mime": item["mime"], "content": item["content"]}
            for item in normalized_attachments
        ],
    }


@dataclass
class _EventRecord:
    event_id: str
    recorded_at: datetime
    event: dict[str, JSONValue]


@dataclass
class _SubscriberState:
    session_id: str
    queue: asyncio.Queue[dict[str, JSONValue]]
    out_of_sync: bool = False
    pending_resync: bool = False
    out_of_sync_count: int = 0
    resync_reason: str = "subscriber_overflow"


@dataclass
class _SessionState:
    principal_id: str = DEFAULT_LEGACY_PRINCIPAL_ID
    messages: list[dict[str, JSONValue]] = field(default_factory=list)
    output_text: str | None = None
    output_updated_at: str | None = None
    files: list[str] = field(default_factory=list)
    artifacts: list[dict[str, JSONValue]] = field(default_factory=list)
    subscribers: dict[
        asyncio.Queue[dict[str, JSONValue]],
        _SubscriberState,
    ] = field(default_factory=dict)
    event_buffer: deque[_EventRecord] = field(default_factory=deque)
    last_decision_id: str | None = None
    decision_packet: dict[str, JSONValue] | None = None
    status_state: SessionStatus = "ok"
    model_provider: str | None = None
    model_id: str | None = None
    title_override: str | None = None
    folder_id: str | None = None
    workspace_root: str | None = None
    policy_profile: PolicyProfile = "sandbox"
    yolo_armed: bool = False
    yolo_armed_at: str | None = None
    tools_state: dict[str, bool] | None = None
    mode: SessionMode = "ask"
    active_plan: dict[str, JSONValue] | None = None
    active_task: dict[str, JSONValue] | None = None
    auto_state: dict[str, JSONValue] | None = None
    created_at: str = field(default_factory=_utc_iso_now)
    updated_at: str = field(default_factory=_utc_iso_now)


@dataclass
class _FolderState:
    name: str
    created_at: str = field(default_factory=_utc_iso_now)
    updated_at: str = field(default_factory=_utc_iso_now)


class UIHub:
    def __init__(
        self,
        *,
        storage: UISessionStorage | None = None,
        session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_messages_per_session: int = DEFAULT_MAX_MESSAGES_PER_SESSION,
        subscriber_queue_maxsize: int = DEFAULT_SUBSCRIBER_QUEUE_MAXSIZE,
        subscriber_drop_policy: SubscriberDropPolicy = DEFAULT_SUBSCRIBER_DROP_POLICY,
        event_buffer_size: int = DEFAULT_EVENT_BUFFER_SIZE,
        event_buffer_ttl_seconds: int = DEFAULT_EVENT_BUFFER_TTL_SECONDS,
    ) -> None:
        self._storage: UISessionStorage = storage or InMemoryUISessionStorage()
        self._sessions: dict[str, _SessionState] = {}
        self._folders: dict[str, _FolderState] = {}
        self._session_ttl_seconds = session_ttl_seconds
        self._max_sessions = max_sessions
        self._max_messages_per_session = max_messages_per_session
        self._subscriber_queue_maxsize = (
            subscriber_queue_maxsize
            if subscriber_queue_maxsize > 0
            else DEFAULT_SUBSCRIBER_QUEUE_MAXSIZE
        )
        self._subscriber_drop_policy = (
            subscriber_drop_policy
            if subscriber_drop_policy in {"drop_newest", "coalesce_deltas"}
            else DEFAULT_SUBSCRIBER_DROP_POLICY
        )
        self._event_buffer_size = (
            event_buffer_size if event_buffer_size > 0 else DEFAULT_EVENT_BUFFER_SIZE
        )
        self._event_buffer_ttl_seconds = (
            event_buffer_ttl_seconds
            if event_buffer_ttl_seconds > 0
            else DEFAULT_EVENT_BUFFER_TTL_SECONDS
        )
        self._lock = asyncio.Lock()
        self._restore_sessions()
        self._restore_folders()

    async def get_or_create_session(self, session_id: str | None, principal_id: str) -> str:
        normalized = session_id.strip() if session_id else ""
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            self._prune_sessions_locked(keep_session_id=normalized if normalized else None)
            if normalized:
                state = self._sessions.get(normalized)
                if state is None:
                    self._sessions[normalized] = _SessionState(principal_id=normalized_principal)
                    self._persist_session_locked(normalized)
                elif state.principal_id != normalized_principal:
                    raise PermissionError("session belongs to another principal")
                return normalized
            new_id = uuid.uuid4().hex
            while new_id in self._sessions:
                new_id = uuid.uuid4().hex
            self._sessions[new_id] = _SessionState(principal_id=normalized_principal)
            self._persist_session_locked(new_id)
            self._prune_sessions_locked(keep_session_id=new_id)
            return new_id

    async def create_session(self, principal_id: str) -> str:
        return await self.get_or_create_session(None, principal_id)

    async def get_session_access(self, session_id: str, principal_id: str) -> SessionAccess:
        normalized_session = session_id.strip()
        normalized_principal = self._normalize_principal_id(principal_id)
        if not normalized_session:
            return "missing"
        async with self._lock:
            state = self._sessions.get(normalized_session)
            if state is None:
                return "missing"
            if state.principal_id != normalized_principal:
                return "forbidden"
            return "owned"

    async def delete_session(self, session_id: str) -> bool:
        normalized = session_id.strip()
        if not normalized:
            return False
        async with self._lock:
            if normalized not in self._sessions:
                return False
            self._drop_sessions_locked([normalized])
            return True

    def _messages_for_lane(
        self,
        messages: list[dict[str, JSONValue]],
        *,
        lane: MessageLane,
    ) -> list[dict[str, JSONValue]]:
        normalized_lane = _normalize_message_lane(lane)
        filtered: list[dict[str, JSONValue]] = []
        for message in messages:
            message_lane = _normalize_message_lane(message.get("lane"), default="chat")
            if message_lane != normalized_lane:
                continue
            filtered.append(dict(message))
        return filtered

    def _session_lane_stats(
        self,
        messages: list[dict[str, JSONValue]],
    ) -> tuple[int, int, MessageLane | None]:
        chat_count = 0
        workspace_count = 0
        last_lane: MessageLane | None = None
        for message in messages:
            message_lane = _normalize_message_lane(message.get("lane"), default="chat")
            if message_lane == "workspace":
                workspace_count += 1
                last_lane = "workspace"
            else:
                chat_count += 1
                last_lane = "chat"
        return chat_count, workspace_count, last_lane

    async def get_messages(
        self,
        session_id: str,
        *,
        lane: MessageLane = "chat",
    ) -> list[dict[str, JSONValue]]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            return self._messages_for_lane(state.messages, lane=lane)

    async def get_session_history(
        self,
        session_id: str,
        *,
        lane: MessageLane = "chat",
    ) -> list[dict[str, JSONValue]] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return self._messages_for_lane(state.messages, lane=lane)

    async def get_session_output(self, session_id: str) -> dict[str, str | None] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return {
                "content": state.output_text,
                "updated_at": state.output_updated_at,
            }

    async def set_session_output(self, session_id: str, content: str | None) -> None:
        normalized = content.strip() if isinstance(content, str) else ""
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            next_content: str | None = normalized or None
            if state.output_text == next_content:
                return
            state.output_text = next_content
            state.output_updated_at = _utc_iso_now() if next_content is not None else None
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)

    async def get_session_files(self, session_id: str) -> list[str] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return list(state.files)

    async def merge_session_files(self, session_id: str, paths: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in paths:
            path = item.strip()
            if path:
                normalized.append(path)
        if not normalized:
            async with self._lock:
                state = self._sessions.get(session_id)
                return list(state.files) if state is not None else []
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            existing = list(state.files)
            seen = set(existing)
            changed = False
            for path in normalized:
                if path in seen:
                    continue
                existing.append(path)
                seen.add(path)
                changed = True
            if not changed:
                return list(state.files)
            state.files = existing
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            return list(state.files)

    async def get_session_artifacts(self, session_id: str) -> list[dict[str, JSONValue]] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return [dict(item) for item in state.artifacts]

    async def append_session_artifact(
        self,
        session_id: str,
        artifact: dict[str, JSONValue],
    ) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            state.artifacts.append(dict(artifact))
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            return dict(artifact)

    async def list_sessions(self, principal_id: str) -> list[SessionListItem]:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            self._prune_sessions_locked()
            items: list[_SessionListSortableItem] = []
            for session_id, state in self._sessions.items():
                if state.principal_id != normalized_principal:
                    continue
                chat_count, workspace_count, last_lane = self._session_lane_stats(state.messages)
                chat_messages = self._messages_for_lane(state.messages, lane="chat")
                title = state.title_override or self._build_session_title(chat_messages)
                items.append(
                    {
                        "session_id": session_id,
                        "title": title,
                        "created_at": state.created_at,
                        "updated_at": datetime.fromisoformat(state.updated_at),
                        "message_count": chat_count,
                        "chat_message_count": chat_count,
                        "workspace_message_count": workspace_count,
                        "last_message_lane": last_lane,
                        "title_override": state.title_override,
                        "folder_id": state.folder_id,
                    },
                )
            items.sort(key=lambda item: item["updated_at"], reverse=True)
            return [
                {
                    "session_id": item["session_id"],
                    "title": item["title"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"].isoformat(),
                    "message_count": item["message_count"],
                    "chat_message_count": item["chat_message_count"],
                    "workspace_message_count": item["workspace_message_count"],
                    "last_message_lane": item["last_message_lane"],
                    "title_override": item["title_override"],
                    "folder_id": item["folder_id"],
                }
                for item in items
            ]

    async def list_folders(self) -> list[dict[str, JSONValue]]:
        async with self._lock:
            items: list[dict[str, JSONValue]] = []
            for folder_id, state in self._folders.items():
                items.append(
                    {
                        "folder_id": folder_id,
                        "name": state.name,
                        "created_at": state.created_at,
                        "updated_at": state.updated_at,
                    },
                )
            items.sort(key=lambda item: str(item["updated_at"]), reverse=True)
            return items

    async def create_folder(self, name: str) -> dict[str, JSONValue]:
        normalized = name.strip()
        if not normalized:
            raise ValueError("folder name required")
        async with self._lock:
            for folder_id, state in self._folders.items():
                if state.name.lower() == normalized.lower():
                    return {
                        "folder_id": folder_id,
                        "name": state.name,
                        "created_at": state.created_at,
                        "updated_at": state.updated_at,
                    }
            folder_id = uuid.uuid4().hex
            now = _utc_iso_now()
            self._folders[folder_id] = _FolderState(
                name=normalized,
                created_at=now,
                updated_at=now,
            )
            self._storage.save_folder(
                PersistedFolder(
                    folder_id=folder_id,
                    name=normalized,
                    created_at=now,
                    updated_at=now,
                ),
            )
            return {
                "folder_id": folder_id,
                "name": normalized,
                "created_at": now,
                "updated_at": now,
            }

    async def set_session_title(self, session_id: str, title: str) -> dict[str, JSONValue]:
        normalized = title.strip()
        if not normalized:
            raise ValueError("title required")
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError("session not found")
            if state.title_override == normalized:
                return {"session_id": session_id, "title": normalized}
            state.title_override = normalized
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            return {"session_id": session_id, "title": normalized}

    async def assign_session_folder(
        self,
        session_id: str,
        folder_id: str | None,
    ) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError("session not found")
            normalized_folder = folder_id.strip() if folder_id else None
            if normalized_folder:
                if normalized_folder not in self._folders:
                    raise KeyError("folder not found")
            if state.folder_id == normalized_folder:
                return {"session_id": session_id, "folder_id": normalized_folder}
            state.folder_id = normalized_folder
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            return {"session_id": session_id, "folder_id": normalized_folder}

    async def export_sessions(self, principal_id: str) -> list[PersistedSession]:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            self._prune_sessions_locked()
            items: list[PersistedSession] = []
            for session_id, state in self._sessions.items():
                if state.principal_id != normalized_principal:
                    continue
                items.append(
                    PersistedSession(
                        session_id=session_id,
                        principal_id=state.principal_id,
                        created_at=state.created_at,
                        updated_at=state.updated_at,
                        status=self._normalize_status(state.status_state),
                        decision=(
                            dict(state.decision_packet)
                            if state.decision_packet is not None
                            else None
                        ),
                        messages=[dict(message) for message in state.messages],
                        model_provider=state.model_provider,
                        model_id=state.model_id,
                        title_override=state.title_override,
                        folder_id=state.folder_id,
                        output_text=state.output_text,
                        output_updated_at=state.output_updated_at,
                        files=list(state.files),
                        artifacts=[dict(item) for item in state.artifacts],
                        workspace_root=state.workspace_root,
                        policy_profile=state.policy_profile,
                        yolo_armed=state.yolo_armed,
                        yolo_armed_at=state.yolo_armed_at,
                        tools_state=(
                            dict(state.tools_state) if isinstance(state.tools_state, dict) else None
                        ),
                        mode=state.mode,
                        active_plan=(
                            dict(state.active_plan) if state.active_plan is not None else None
                        ),
                        active_task=(
                            dict(state.active_task) if state.active_task is not None else None
                        ),
                        auto_state=(
                            dict(state.auto_state) if isinstance(state.auto_state, dict) else None
                        ),
                    ),
                )
            items.sort(
                key=lambda item: self._parse_session_timestamp(item.updated_at),
                reverse=True,
            )
            return items

    async def import_sessions(
        self,
        sessions: list[PersistedSession],
        *,
        principal_id: str,
        mode: Literal["replace", "merge"] = "replace",
    ) -> int:
        normalized_principal = self._normalize_principal_id(principal_id)
        async with self._lock:
            if mode == "replace":
                self._drop_sessions_locked(
                    [
                        session_id
                        for session_id, state in self._sessions.items()
                        if state.principal_id == normalized_principal
                    ]
                )
            imported = 0
            for item in sessions:
                decision = dict(item.decision) if item.decision is not None else None
                last_decision_id: str | None = None
                if decision is not None:
                    decision_id = decision.get("id")
                    if isinstance(decision_id, str):
                        last_decision_id = decision_id
                restored_updated_at = _utc_iso_now()
                created_at = item.created_at or restored_updated_at
                existing = self._sessions.get(item.session_id)
                if existing is not None and existing.principal_id != normalized_principal:
                    continue
                subscribers = dict(existing.subscribers) if existing is not None else {}
                self._sessions[item.session_id] = _SessionState(
                    principal_id=normalized_principal,
                    messages=[dict(message) for message in item.messages],
                    output_text=item.output_text,
                    output_updated_at=item.output_updated_at,
                    files=list(item.files),
                    artifacts=[dict(entry) for entry in item.artifacts],
                    subscribers=subscribers,
                    last_decision_id=last_decision_id,
                    decision_packet=decision,
                    status_state=self._normalize_status(item.status),
                    model_provider=item.model_provider,
                    model_id=item.model_id,
                    title_override=item.title_override,
                    folder_id=item.folder_id,
                    workspace_root=item.workspace_root,
                    policy_profile=self._normalize_policy_profile(item.policy_profile),
                    yolo_armed=bool(item.yolo_armed),
                    yolo_armed_at=item.yolo_armed_at,
                    tools_state=(
                        dict(item.tools_state) if isinstance(item.tools_state, dict) else None
                    ),
                    mode=self._normalize_mode(item.mode),
                    active_plan=(dict(item.active_plan) if item.active_plan is not None else None),
                    active_task=(dict(item.active_task) if item.active_task is not None else None),
                    auto_state=normalize_auto_state(item.auto_state),
                    created_at=created_at,
                    updated_at=restored_updated_at,
                )
                self._persist_session_locked(item.session_id)
                imported += 1
            self._prune_sessions_locked()
            return imported

    async def get_session(self, session_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            decision = dict(state.decision_packet) if state.decision_packet is not None else None
            status_value = self._normalize_status(state.status_state)
            selected_model = self._selected_model_payload(state)
            chat_messages = self._messages_for_lane(state.messages, lane="chat")
            workspace_messages = self._messages_for_lane(state.messages, lane="workspace")
            chat_count, workspace_count, last_lane = self._session_lane_stats(state.messages)
            return {
                "session_id": session_id,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "status": status_value,
                "messages": chat_messages,
                "workspace_messages": workspace_messages,
                "lane_stats": {
                    "chat_message_count": chat_count,
                    "workspace_message_count": workspace_count,
                    "last_message_lane": last_lane,
                },
                "output": {
                    "content": state.output_text,
                    "updated_at": state.output_updated_at,
                },
                "files": list(state.files),
                "artifacts": [dict(item) for item in state.artifacts],
                "decision": decision,
                "selected_model": selected_model,
                "title_override": state.title_override,
                "folder_id": state.folder_id,
                "workspace_root": state.workspace_root,
                "policy": {
                    "profile": state.policy_profile,
                    "yolo_armed": state.yolo_armed,
                    "yolo_armed_at": state.yolo_armed_at,
                },
                "tools_state": (
                    dict(state.tools_state) if isinstance(state.tools_state, dict) else None
                ),
                "mode": state.mode,
                "active_plan": dict(state.active_plan) if state.active_plan is not None else None,
                "active_task": dict(state.active_task) if state.active_task is not None else None,
                "auto_state": dict(state.auto_state) if state.auto_state is not None else None,
            }

    async def get_session_decision(self, session_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.decision_packet is None:
                return None
            return dict(state.decision_packet)

    async def get_workspace_root(self, session_id: str) -> str | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return state.workspace_root

    async def set_workspace_root(self, session_id: str, root_path: str | None) -> None:
        normalized = root_path.strip() if isinstance(root_path, str) and root_path.strip() else None
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            if state.workspace_root == normalized:
                return
            state.workspace_root = normalized
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)

    async def get_session_policy(self, session_id: str) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return {"profile": "sandbox", "yolo_armed": False, "yolo_armed_at": None}
            return {
                "profile": state.policy_profile,
                "yolo_armed": state.yolo_armed,
                "yolo_armed_at": state.yolo_armed_at,
            }

    async def set_session_policy(
        self,
        session_id: str,
        *,
        profile: str | None = None,
        yolo_armed: bool | None = None,
        yolo_armed_at: str | None = None,
    ) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            changed = False
            if profile is not None:
                normalized_profile = self._normalize_policy_profile(profile)
                if state.policy_profile != normalized_profile:
                    state.policy_profile = normalized_profile
                    changed = True
            if yolo_armed is not None and state.yolo_armed != yolo_armed:
                state.yolo_armed = yolo_armed
                changed = True
            if yolo_armed is not None:
                next_armed_at = yolo_armed_at if yolo_armed else None
                if state.yolo_armed_at != next_armed_at:
                    state.yolo_armed_at = next_armed_at
                    changed = True
            elif yolo_armed_at is not None and state.yolo_armed_at != yolo_armed_at:
                state.yolo_armed_at = yolo_armed_at
                changed = True
            if changed:
                state.updated_at = _utc_iso_now()
                self._persist_session_locked(session_id)
                self._prune_sessions_locked(keep_session_id=session_id)
            return {
                "profile": state.policy_profile,
                "yolo_armed": state.yolo_armed,
                "yolo_armed_at": state.yolo_armed_at,
            }

    async def consume_yolo_once(self, session_id: str) -> bool:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or not state.yolo_armed:
                return False
            state.yolo_armed = False
            state.yolo_armed_at = None
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            return True

    async def get_session_tools_state(self, session_id: str) -> dict[str, bool] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or not isinstance(state.tools_state, dict):
                return None
            return dict(state.tools_state)

    async def set_session_tools_state(
        self,
        session_id: str,
        *,
        tools_state: dict[str, bool] | None,
        merge: bool = True,
    ) -> dict[str, bool] | None:
        normalized = self._normalize_tools_state(tools_state)
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            current = dict(state.tools_state) if isinstance(state.tools_state, dict) else {}
            next_value: dict[str, bool] | None
            if normalized is None:
                next_value = None if not merge else (current or None)
            elif merge:
                merged = dict(current)
                merged.update(normalized)
                next_value = merged or None
            else:
                next_value = dict(normalized)
            if state.tools_state == next_value:
                return dict(next_value) if isinstance(next_value, dict) else None
            state.tools_state = dict(next_value) if isinstance(next_value, dict) else None
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            return dict(state.tools_state) if isinstance(state.tools_state, dict) else None

    async def set_session_model(self, session_id: str, provider: str, model_id: str) -> None:
        event: dict[str, JSONValue] | None = None
        subscribers: list[_SubscriberState] = []
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            changed = state.model_provider != provider or state.model_id != model_id
            state.model_provider = provider
            state.model_id = model_id
            if not changed:
                return
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            selected_model: dict[str, JSONValue] = {"provider": provider, "model": model_id}
            payload: dict[str, JSONValue] = {
                "session_id": session_id,
                "selected_model": selected_model,
            }
            subscribers = list(state.subscribers.values())
            event = self._build_event("session.model", payload)
            self._append_event_record_locked(state, event)
        if event is None:
            return
        self._publish_to_subscribers(subscribers, event)

    async def get_session_model(self, session_id: str) -> dict[str, str] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            payload = self._selected_model_payload(state)
            if payload is None:
                return None
            provider = payload.get("provider")
            model_id = payload.get("model")
            if not isinstance(provider, str) or not isinstance(model_id, str):
                return None
            return {"provider": provider, "model": model_id}

    async def set_session_status(self, session_id: str, status_state: str) -> None:
        normalized = self._normalize_status(status_state)
        event: dict[str, JSONValue] | None = None
        subscribers: list[_SubscriberState] = []
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            if state.status_state == normalized:
                return
            state.status_state = normalized
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            payload = self._status_payload(session_id, normalized)
            subscribers = list(state.subscribers.values())
            event = self._build_event("status", payload)
            self._append_event_record_locked(state, event)
        if event is None:
            return
        self._publish_to_subscribers(subscribers, event)

    async def get_session_status_event(self, session_id: str) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
                self._persist_session_locked(session_id)
            payload = self._status_payload(session_id, state.status_state)
        return self._build_event("status", payload)

    async def get_session_workflow_event(self, session_id: str) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
                self._persist_session_locked(session_id)
            payload: dict[str, JSONValue] = {
                "session_id": session_id,
                "mode": state.mode,
                "active_plan": (dict(state.active_plan) if state.active_plan is not None else None),
                "active_task": (dict(state.active_task) if state.active_task is not None else None),
                "auto_state": (dict(state.auto_state) if state.auto_state is not None else None),
            }
        return self._build_event("session.workflow", payload)

    async def get_events_since(
        self,
        session_id: str,
        *,
        last_event_id: str | None,
    ) -> tuple[list[dict[str, JSONValue]], bool]:
        normalized_last_id = (
            last_event_id.strip()
            if isinstance(last_event_id, str) and last_event_id.strip()
            else None
        )
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
                self._persist_session_locked(session_id)
            self._trim_event_buffer_locked(state)
            if normalized_last_id is None:
                return [], False
            if not state.event_buffer:
                return [], True
            events = list(state.event_buffer)
            replay_start = -1
            for idx, record in enumerate(events):
                if record.event_id == normalized_last_id:
                    replay_start = idx + 1
                    break
            if replay_start < 0:
                return [], True
            replay = [dict(record.event) for record in events[replay_start:]]
            return replay, False

    async def get_session_workflow(self, session_id: str) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
                self._persist_session_locked(session_id)
            return {
                "mode": state.mode,
                "active_plan": dict(state.active_plan) if state.active_plan is not None else None,
                "active_task": dict(state.active_task) if state.active_task is not None else None,
                "auto_state": dict(state.auto_state) if state.auto_state is not None else None,
            }

    async def start_plan_task_if_possible(
        self,
        session_id: str,
        *,
        expected_mode: str,
        expected_plan_id: str,
        expected_plan_hash: str,
        expected_plan_revision: int,
        running_plan: dict[str, JSONValue],
        task_payload: dict[str, JSONValue],
        next_mode: str = "act",
    ) -> tuple[bool, dict[str, JSONValue], str | None]:
        event: dict[str, JSONValue] | None = None
        subscribers: list[_SubscriberState] = []
        reason: str | None = None
        succeeded = False
        payload: dict[str, JSONValue] | None = None
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                payload = {
                    "session_id": session_id,
                    "mode": "ask",
                    "active_plan": None,
                    "active_task": None,
                    "auto_state": None,
                }
                reason = "plan_not_found"
            else:
                if state.mode != self._normalize_mode(expected_mode):
                    reason = "mode_mismatch"
                current_plan = state.active_plan if isinstance(state.active_plan, dict) else None
                current_task = state.active_task if isinstance(state.active_task, dict) else None
                if reason is None and isinstance(current_task, dict):
                    task_status = current_task.get("status")
                    if task_status == "running":
                        reason = "task_already_running"
                if reason is None and current_plan is None:
                    reason = "plan_not_found"
                if reason is None and current_plan is not None:
                    current_plan_id = current_plan.get("plan_id")
                    current_plan_hash = current_plan.get("plan_hash")
                    current_plan_revision_raw = current_plan.get("plan_revision")
                    current_plan_revision = (
                        current_plan_revision_raw
                        if (
                            isinstance(current_plan_revision_raw, int)
                            and current_plan_revision_raw > 0
                        )
                        else 1
                    )
                    if (
                        current_plan_id != expected_plan_id
                        or current_plan_hash != expected_plan_hash
                        or current_plan_revision != expected_plan_revision
                    ):
                        reason = "plan_mismatch"
                    elif current_plan.get("status") != "approved":
                        reason = "plan_not_approved"
                if reason is None:
                    normalized_mode = self._normalize_mode(next_mode)
                    normalized_plan = self._normalize_optional_object(running_plan)
                    normalized_task = self._normalize_optional_object(task_payload)
                    state.mode = normalized_mode
                    state.active_plan = normalized_plan
                    state.active_task = normalized_task
                    state.updated_at = _utc_iso_now()
                    self._persist_session_locked(session_id)
                    self._prune_sessions_locked(keep_session_id=session_id)
                    payload = {
                        "session_id": session_id,
                        "mode": state.mode,
                        "active_plan": (
                            dict(state.active_plan) if state.active_plan is not None else None
                        ),
                        "active_task": (
                            dict(state.active_task) if state.active_task is not None else None
                        ),
                        "auto_state": (
                            dict(state.auto_state) if state.auto_state is not None else None
                        ),
                    }
                    subscribers = list(state.subscribers.values())
                    event = self._build_event("session.workflow", payload)
                    self._append_event_record_locked(state, event)
                    succeeded = True
                else:
                    payload = {
                        "session_id": session_id,
                        "mode": state.mode,
                        "active_plan": (
                            dict(state.active_plan) if state.active_plan is not None else None
                        ),
                        "active_task": (
                            dict(state.active_task) if state.active_task is not None else None
                        ),
                        "auto_state": (
                            dict(state.auto_state) if state.auto_state is not None else None
                        ),
                    }
        if event is not None:
            self._publish_to_subscribers(subscribers, event)
        if payload is None:
            payload = {
                "session_id": session_id,
                "mode": "ask",
                "active_plan": None,
                "active_task": None,
                "auto_state": None,
            }
        return succeeded, payload, reason

    async def set_session_workflow(
        self,
        session_id: str,
        *,
        mode: str | None = None,
        active_plan: dict[str, JSONValue] | None | object = _UNSET,
        active_task: dict[str, JSONValue] | None | object = _UNSET,
        auto_state: dict[str, JSONValue] | None | object = _UNSET,
    ) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            changed = False
            if mode is not None:
                normalized_mode = self._normalize_mode(mode)
                if state.mode != normalized_mode:
                    state.mode = normalized_mode
                    changed = True
            if active_plan is not _UNSET:
                next_plan = self._normalize_optional_object(active_plan)
                if state.active_plan != next_plan:
                    state.active_plan = next_plan
                    changed = True
            if active_task is not _UNSET:
                next_task = self._normalize_optional_object(active_task)
                if state.active_task != next_task:
                    state.active_task = next_task
                    changed = True
            if auto_state is not _UNSET:
                next_auto_state = normalize_auto_state(auto_state)
                if state.auto_state != next_auto_state:
                    state.auto_state = next_auto_state
                    changed = True
            payload: dict[str, JSONValue] = {
                "session_id": session_id,
                "mode": state.mode,
                "active_plan": (dict(state.active_plan) if state.active_plan is not None else None),
                "active_task": (dict(state.active_task) if state.active_task is not None else None),
                "auto_state": (dict(state.auto_state) if state.auto_state is not None else None),
            }
            if not changed:
                subscribers: list[_SubscriberState] = []
                event: dict[str, JSONValue] | None = None
            else:
                state.updated_at = _utc_iso_now()
                self._persist_session_locked(session_id)
                self._prune_sessions_locked(keep_session_id=session_id)
                subscribers = list(state.subscribers.values())
                event = self._build_event("session.workflow", payload)
                self._append_event_record_locked(state, event)
        if subscribers:
            if event is not None:
                self._publish_to_subscribers(subscribers, event)
        return payload

    async def append_message(
        self,
        session_id: str,
        message: dict[str, JSONValue],
        *,
        lane: MessageLane = "chat",
    ) -> dict[str, JSONValue]:
        normalized_message = _normalize_message_payload(message)
        normalized_message["lane"] = _normalize_message_lane(lane)
        event: dict[str, JSONValue] | None = None
        subscribers: list[_SubscriberState] = []
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            state.messages.append(normalized_message)
            if (
                self._max_messages_per_session > 0
                and len(state.messages) > self._max_messages_per_session
            ):
                state.messages = state.messages[-self._max_messages_per_session :]
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            subscribers = list(state.subscribers.values())
            event = self._build_event(
                "message.append",
                {"session_id": session_id, "message": dict(normalized_message)},
            )
            self._append_event_record_locked(state, event)
        if event is None:
            return dict(normalized_message)
        self._publish_to_subscribers(subscribers, event)
        return dict(normalized_message)

    async def subscribe(self, session_id: str) -> asyncio.Queue[dict[str, JSONValue]]:
        queue: asyncio.Queue[dict[str, JSONValue]] = asyncio.Queue(
            maxsize=self._subscriber_queue_maxsize
        )
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
                self._persist_session_locked(session_id)
            subscriber = _SubscriberState(session_id=session_id, queue=queue)
            state.subscribers[queue] = subscriber
            self._flush_pending_resync(subscriber)
        return queue

    async def unsubscribe(
        self,
        session_id: str,
        queue: asyncio.Queue[dict[str, JSONValue]],
    ) -> None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.subscribers.pop(queue, None)

    async def publish(self, session_id: str, event: dict[str, JSONValue]) -> None:
        subscribers: list[_SubscriberState] = []
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            hydrated = self._ensure_event_fields(event)
            self._append_event_record_locked(state, hydrated)
            subscribers = list(state.subscribers.values())
        self._publish_to_subscribers(subscribers, hydrated)

    async def set_session_decision(
        self,
        session_id: str,
        decision: dict[str, JSONValue] | None,
    ) -> None:
        event: dict[str, JSONValue] | None = None
        subscribers: list[_SubscriberState] = []
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            previous_decision = (
                dict(state.decision_packet) if state.decision_packet is not None else None
            )
            next_decision = dict(decision) if decision is not None else None
            if previous_decision == next_decision:
                return
            state.decision_packet = next_decision
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            if state.decision_packet is None:
                state.last_decision_id = None
                return
            decision_id = state.decision_packet.get("id")
            if isinstance(decision_id, str):
                state.last_decision_id = decision_id
            else:
                state.last_decision_id = None
            subscribers = list(state.subscribers.values())
            decision_payload = dict(state.decision_packet)
            event = self._build_event(
                "decision.packet",
                {"session_id": session_id, "decision": decision_payload},
            )
            self._append_event_record_locked(state, event)
        if event is None:
            return
        self._publish_to_subscribers(subscribers, event)

    async def transition_session_decision(
        self,
        session_id: str,
        *,
        expected_id: str,
        expected_status: str,
        next_decision: dict[str, JSONValue],
    ) -> tuple[bool, dict[str, JSONValue] | None]:
        event: dict[str, JSONValue] | None = None
        subscribers: list[_SubscriberState] = []
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.decision_packet is None:
                return False, None
            current = dict(state.decision_packet)
            current_id_raw = current.get("id")
            current_status_raw = current.get("status")
            current_id = current_id_raw if isinstance(current_id_raw, str) else None
            current_status = current_status_raw if isinstance(current_status_raw, str) else None
            if current_id != expected_id or current_status != expected_status:
                return False, current

            previous_decision = dict(state.decision_packet)
            candidate = dict(next_decision)
            if previous_decision == candidate:
                return True, candidate

            state.decision_packet = candidate
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            decision_id_raw = candidate.get("id")
            if isinstance(decision_id_raw, str):
                state.last_decision_id = decision_id_raw
            else:
                state.last_decision_id = None
            subscribers = list(state.subscribers.values())
            decision_payload = dict(candidate)
            event = self._build_event(
                "decision.packet",
                {"session_id": session_id, "decision": decision_payload},
            )
            self._append_event_record_locked(state, event)
        if event is None:
            return True, decision_payload
        self._publish_to_subscribers(subscribers, event)
        return True, decision_payload

    def _ensure_event_fields(self, event: dict[str, JSONValue]) -> dict[str, JSONValue]:
        if "id" not in event:
            event = {**event, "id": uuid.uuid4().hex}
        if "ts" not in event:
            event = {**event, "ts": _utc_iso_now()}
        return event

    def _build_event(self, event_type: str, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return {
            "id": uuid.uuid4().hex,
            "type": event_type,
            "ts": _utc_iso_now(),
            "payload": payload,
        }

    def build_resync_required_event(
        self,
        *,
        session_id: str,
        reason: str,
        resync_only: bool,
    ) -> dict[str, JSONValue]:
        return self._build_resync_required_event(
            session_id=session_id,
            reason=reason,
            resync_only=resync_only,
        )

    def _selected_model_payload(self, state: _SessionState) -> dict[str, JSONValue] | None:
        provider = state.model_provider
        model_id = state.model_id
        if not provider or not model_id:
            return None
        return {"provider": provider, "model": model_id}

    def _status_payload(self, session_id: str, status_state: str) -> dict[str, JSONValue]:
        return {
            "session_id": session_id,
            "state": status_state,
            "ok": status_state != "error",
        }

    def _normalize_status(self, status_state: str) -> SessionStatus:
        if status_state == "ok":
            return "ok"
        if status_state == "busy":
            return "busy"
        if status_state == "error":
            return "error"
        return "ok"

    def _normalize_policy_profile(self, value: str | None) -> PolicyProfile:
        if value == "index":
            return "index"
        if value == "yolo":
            return "yolo"
        return "sandbox"

    def _normalize_principal_id(self, value: str | None) -> str:
        normalized = value.strip() if isinstance(value, str) else ""
        return normalized or DEFAULT_LEGACY_PRINCIPAL_ID

    def _normalize_mode(self, value: str | None) -> SessionMode:
        normalized = value.strip().lower() if isinstance(value, str) else ""
        if normalized == "plan":
            return "plan"
        if normalized == "act":
            return "act"
        if normalized == "auto":
            return "auto"
        return "ask"

    def _normalize_optional_object(
        self,
        value: dict[str, JSONValue] | None | object,
    ) -> dict[str, JSONValue] | None:
        if value is None or value is _UNSET:
            return None
        if not isinstance(value, dict):
            return None
        normalized: dict[str, JSONValue] = {}
        for key, item in value.items():
            normalized[str(key)] = item
        return normalized

    def _normalize_tools_state(
        self,
        value: dict[str, bool] | None,
    ) -> dict[str, bool] | None:
        if not isinstance(value, dict):
            return None
        normalized: dict[str, bool] = {}
        for key, item in value.items():
            if isinstance(key, str) and isinstance(item, bool):
                normalized[key] = item
        return normalized or None

    def _publish_to_subscribers(
        self,
        subscribers: list[_SubscriberState],
        event: dict[str, JSONValue],
    ) -> None:
        for subscriber in subscribers:
            self._flush_pending_resync(subscriber)
            try:
                subscriber.queue.put_nowait(event)
            except asyncio.QueueFull:
                self._handle_subscriber_queue_overflow(subscriber, event)

    def _handle_subscriber_queue_overflow(
        self,
        subscriber: _SubscriberState,
        event: dict[str, JSONValue],
    ) -> None:
        queue = subscriber.queue
        if self._subscriber_drop_policy == "coalesce_deltas":
            if self._is_delta_event(event) and self._coalesce_delta_event(queue, event):
                return
            if self._evict_oldest_delta_event(queue):
                try:
                    queue.put_nowait(event)
                    return
                except asyncio.QueueFull:
                    pass
        if not self._is_control_event(event):
            return
        if self._evict_oldest_non_control_event(queue):
            try:
                queue.put_nowait(event)
                return
            except asyncio.QueueFull:
                pass
        self._mark_subscriber_out_of_sync(subscriber, reason="control_overflow")
        self._flush_pending_resync(subscriber)

    def _is_delta_event(self, event: dict[str, JSONValue]) -> bool:
        event_type = event.get("type")
        return isinstance(event_type, str) and event_type in _DELTA_EVENT_TYPES

    def _is_control_event(self, event: dict[str, JSONValue]) -> bool:
        event_type = event.get("type")
        return isinstance(event_type, str) and event_type in _CONTROL_EVENT_TYPES

    def _coalesce_delta_event(
        self,
        queue: asyncio.Queue[dict[str, JSONValue]],
        event: dict[str, JSONValue],
    ) -> bool:
        queue_items_raw = getattr(queue, "_queue", None)
        if not isinstance(queue_items_raw, deque):
            return False
        queue_items = cast(deque[dict[str, JSONValue]], queue_items_raw)
        for idx in range(len(queue_items) - 1, -1, -1):
            candidate = queue_items[idx]
            if not isinstance(candidate, dict):
                continue
            merged = self._merge_delta_event(candidate, event)
            if merged is None:
                continue
            queue_items[idx] = merged
            return True
        return False

    def _merge_delta_event(
        self,
        queued_event: dict[str, JSONValue],
        incoming_event: dict[str, JSONValue],
    ) -> dict[str, JSONValue] | None:
        queued_type = queued_event.get("type")
        incoming_type = incoming_event.get("type")
        if not isinstance(queued_type, str) or queued_type != incoming_type:
            return None
        if queued_type not in _DELTA_EVENT_TYPES:
            return None
        queued_payload = queued_event.get("payload")
        incoming_payload = incoming_event.get("payload")
        if not isinstance(queued_payload, dict) or not isinstance(incoming_payload, dict):
            return None
        incoming_delta = incoming_payload.get("delta")
        if not isinstance(incoming_delta, str) or not incoming_delta:
            return None
        if queued_type == "chat.stream.delta":
            queued_stream = queued_payload.get("stream_id")
            incoming_stream = incoming_payload.get("stream_id")
            queued_lane = queued_payload.get("lane")
            incoming_lane = incoming_payload.get("lane")
            if queued_stream != incoming_stream or queued_lane != incoming_lane:
                return None
            queued_mode = "replace" if queued_payload.get("mode") == "replace" else "append"
            incoming_mode = "replace" if incoming_payload.get("mode") == "replace" else "append"
            queued_delta_raw = queued_payload.get("delta")
            queued_delta = queued_delta_raw if isinstance(queued_delta_raw, str) else ""
            if incoming_mode == "replace":
                merged_mode: JSONValue = "replace"
                merged_delta = incoming_delta
            elif queued_mode == "replace":
                merged_mode = "replace"
                merged_delta = f"{queued_delta}{incoming_delta}"
            else:
                merged_mode = "append"
                merged_delta = f"{queued_delta}{incoming_delta}"
            merged_payload = dict(queued_payload)
            merged_payload["delta"] = merged_delta
            merged_payload["mode"] = merged_mode
            return {
                **queued_event,
                "payload": merged_payload,
            }
        if queued_type == "canvas.stream.delta":
            queued_artifact = queued_payload.get("artifact_id")
            incoming_artifact = incoming_payload.get("artifact_id")
            if queued_artifact != incoming_artifact:
                return None
            queued_delta_raw = queued_payload.get("delta")
            queued_delta = queued_delta_raw if isinstance(queued_delta_raw, str) else ""
            merged_payload = dict(queued_payload)
            merged_payload["delta"] = f"{queued_delta}{incoming_delta}"
            return {
                **queued_event,
                "payload": merged_payload,
            }
        return None

    def _evict_oldest_delta_event(self, queue: asyncio.Queue[dict[str, JSONValue]]) -> bool:
        queue_items_raw = getattr(queue, "_queue", None)
        if not isinstance(queue_items_raw, deque):
            return False
        queue_items = cast(deque[dict[str, JSONValue]], queue_items_raw)
        for idx, candidate in enumerate(queue_items):
            if isinstance(candidate, dict) and self._is_delta_event(candidate):
                del queue_items[idx]
                return True
        return False

    def _evict_oldest_non_control_event(self, queue: asyncio.Queue[dict[str, JSONValue]]) -> bool:
        queue_items_raw = getattr(queue, "_queue", None)
        if not isinstance(queue_items_raw, deque):
            return False
        queue_items = cast(deque[dict[str, JSONValue]], queue_items_raw)
        for idx, candidate in enumerate(queue_items):
            if not isinstance(candidate, dict):
                del queue_items[idx]
                return True
            if not self._is_control_event(candidate):
                del queue_items[idx]
                return True
        return False

    def _mark_subscriber_out_of_sync(self, subscriber: _SubscriberState, *, reason: str) -> None:
        subscriber.out_of_sync = True
        subscriber.pending_resync = True
        subscriber.out_of_sync_count += 1
        subscriber.resync_reason = reason

    def _build_resync_required_event(
        self,
        *,
        session_id: str,
        reason: str,
        resync_only: bool,
    ) -> dict[str, JSONValue]:
        return self._build_event(
            "session.resync_required",
            {
                "session_id": session_id,
                "reason": reason,
                "resync_only": resync_only,
            },
        )

    def _flush_pending_resync(self, subscriber: _SubscriberState) -> None:
        if not subscriber.pending_resync:
            return
        event = self._build_resync_required_event(
            session_id=subscriber.session_id,
            reason=subscriber.resync_reason,
            resync_only=True,
        )
        queue = subscriber.queue
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            if not self._evict_oldest_non_control_event(queue):
                return
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                return
        subscriber.pending_resync = False
        subscriber.out_of_sync = False

    def _append_event_record_locked(
        self,
        state: _SessionState,
        event: dict[str, JSONValue],
    ) -> None:
        event_id_raw = event.get("id")
        event_id = event_id_raw if isinstance(event_id_raw, str) and event_id_raw.strip() else None
        if event_id is None:
            return
        now = datetime.now(timezone.utc)  # noqa: UP017
        state.event_buffer.append(
            _EventRecord(
                event_id=event_id,
                recorded_at=now,
                event=dict(event),
            )
        )
        self._trim_event_buffer_locked(state, now=now)

    def _trim_event_buffer_locked(
        self,
        state: _SessionState,
        *,
        now: datetime | None = None,
    ) -> None:
        current_now = now or datetime.now(timezone.utc)  # noqa: UP017
        cutoff = current_now - timedelta(seconds=self._event_buffer_ttl_seconds)
        while state.event_buffer and state.event_buffer[0].recorded_at < cutoff:
            state.event_buffer.popleft()
        while len(state.event_buffer) > self._event_buffer_size:
            state.event_buffer.popleft()

    def _restore_sessions(self) -> None:
        for item in self._storage.load_sessions():
            decision = dict(item.decision) if item.decision is not None else None
            last_decision_id: str | None = None
            if decision is not None:
                decision_id = decision.get("id")
                if isinstance(decision_id, str):
                    last_decision_id = decision_id
            self._sessions[item.session_id] = _SessionState(
                principal_id=self._normalize_principal_id(item.principal_id),
                messages=[dict(message) for message in item.messages],
                output_text=item.output_text,
                output_updated_at=item.output_updated_at,
                files=list(item.files),
                artifacts=[dict(entry) for entry in item.artifacts],
                last_decision_id=last_decision_id,
                decision_packet=decision,
                status_state=self._normalize_status(item.status),
                model_provider=item.model_provider,
                model_id=item.model_id,
                title_override=item.title_override,
                folder_id=item.folder_id,
                workspace_root=item.workspace_root,
                policy_profile=self._normalize_policy_profile(item.policy_profile),
                yolo_armed=bool(item.yolo_armed),
                yolo_armed_at=item.yolo_armed_at,
                tools_state=(
                    dict(item.tools_state) if isinstance(item.tools_state, dict) else None
                ),
                mode=self._normalize_mode(item.mode),
                active_plan=(dict(item.active_plan) if item.active_plan is not None else None),
                active_task=(dict(item.active_task) if item.active_task is not None else None),
                auto_state=normalize_auto_state(item.auto_state),
                created_at=item.created_at,
                updated_at=item.updated_at,
            )

    def _restore_folders(self) -> None:
        for item in self._storage.load_folders():
            self._folders[item.folder_id] = _FolderState(
                name=item.name,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )

    def _persist_session_locked(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            return
        self._storage.save_session(
            PersistedSession(
                session_id=session_id,
                principal_id=state.principal_id,
                created_at=state.created_at,
                updated_at=state.updated_at,
                status=state.status_state,
                decision=(
                    dict(state.decision_packet) if state.decision_packet is not None else None
                ),
                messages=[dict(message) for message in state.messages],
                model_provider=state.model_provider,
                model_id=state.model_id,
                title_override=state.title_override,
                folder_id=state.folder_id,
                output_text=state.output_text,
                output_updated_at=state.output_updated_at,
                files=list(state.files),
                artifacts=[dict(entry) for entry in state.artifacts],
                workspace_root=state.workspace_root,
                policy_profile=state.policy_profile,
                yolo_armed=state.yolo_armed,
                yolo_armed_at=state.yolo_armed_at,
                tools_state=(
                    dict(state.tools_state) if isinstance(state.tools_state, dict) else None
                ),
                mode=state.mode,
                active_plan=(dict(state.active_plan) if state.active_plan is not None else None),
                active_task=(dict(state.active_task) if state.active_task is not None else None),
                auto_state=(dict(state.auto_state) if state.auto_state is not None else None),
            ),
        )

    def _prune_sessions_locked(self, *, keep_session_id: str | None = None) -> None:
        to_remove: set[str] = set()
        now = datetime.now(timezone.utc)  # noqa: UP017
        if self._session_ttl_seconds > 0:
            cutoff = now - timedelta(seconds=self._session_ttl_seconds)
            for session_id, state in self._sessions.items():
                if session_id == keep_session_id:
                    continue
                updated_at = self._parse_session_timestamp(state.updated_at)
                if updated_at < cutoff:
                    to_remove.add(session_id)
        if self._max_sessions > 0:
            sessions_sorted = sorted(
                self._sessions.items(),
                key=lambda item: self._parse_session_timestamp(item[1].updated_at),
                reverse=True,
            )
            keep_ids = {session_id for session_id, _ in sessions_sorted[: self._max_sessions]}
            if keep_session_id is not None:
                keep_ids.add(keep_session_id)
            for session_id in self._sessions:
                if session_id not in keep_ids:
                    to_remove.add(session_id)
        self._drop_sessions_locked(to_remove)

    def _drop_sessions_locked(self, session_ids: Iterable[str]) -> None:
        to_remove = [session_id for session_id in session_ids if session_id in self._sessions]
        if not to_remove:
            return
        for session_id in to_remove:
            self._sessions.pop(session_id, None)
        self._storage.delete_sessions(to_remove)

    def _parse_session_timestamp(self, value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)  # noqa: UP017
            return parsed.astimezone(timezone.utc)  # noqa: UP017
        except ValueError:
            return datetime.fromtimestamp(0, tz=timezone.utc)  # noqa: UP017

    def _build_session_title(self, messages: list[dict[str, JSONValue]]) -> str:
        for message in messages:
            lane_raw = message.get("lane")
            lane = _normalize_message_lane(lane_raw, default="chat")
            if lane != "chat":
                continue
            role_raw = message.get("role")
            role = role_raw if isinstance(role_raw, str) else ""
            if role != "user":
                continue
            content_raw = message.get("content")
            raw_content = content_raw.strip() if isinstance(content_raw, str) else ""
            if not raw_content:
                continue
            first_line = raw_content.splitlines()[0].strip()
            compact = " ".join(first_line.split())
            if not compact:
                continue
            if len(compact) <= 48:
                return compact
            return f"{compact[:45].rstrip()}..."
        return "New chat"

    def create_message(
        self,
        *,
        role: str,
        content: str,
        lane: MessageLane = "chat",
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
