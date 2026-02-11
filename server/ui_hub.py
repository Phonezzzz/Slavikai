from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, TypedDict

from server.ui_session_storage import (
    InMemoryUISessionStorage,
    PersistedFolder,
    PersistedSession,
    UISessionStorage,
)
from shared.models import JSONValue


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


SessionStatus = Literal["ok", "busy", "error"]


class SessionListItem(TypedDict):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    title_override: str | None
    folder_id: str | None


class _SessionListSortableItem(TypedDict):
    session_id: str
    title: str
    created_at: str
    updated_at: datetime
    message_count: int
    title_override: str | None
    folder_id: str | None


DEFAULT_SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_SESSIONS = 200
DEFAULT_MAX_MESSAGES_PER_SESSION = 500

_MESSAGE_ROLES = {"user", "assistant", "system"}


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
    trace_id: str | None = None,
    parent_user_message_id: str | None = None,
    attachments: list[dict[str, str]] | None = None,
    message_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, JSONValue]:
    normalized_role = role.strip()
    if normalized_role not in _MESSAGE_ROLES:
        raise ValueError(f"unsupported message role: {role}")
    return {
        "message_id": (message_id or uuid.uuid4().hex).strip(),
        "role": normalized_role,
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
    if not isinstance(created_at, str) or not created_at.strip():
        raise ValueError("created_at required")
    if trace_id is not None and (not isinstance(trace_id, str) or not trace_id.strip()):
        raise ValueError("trace_id must be string or null")
    if parent_user_message_id is not None and (
        not isinstance(parent_user_message_id, str) or not parent_user_message_id.strip()
    ):
        raise ValueError("parent_user_message_id must be string or null")

    normalized_role = role.strip()
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
class _SessionState:
    messages: list[dict[str, JSONValue]] = field(default_factory=list)
    output_text: str | None = None
    output_updated_at: str | None = None
    files: list[str] = field(default_factory=list)
    artifacts: list[dict[str, JSONValue]] = field(default_factory=list)
    subscribers: set[asyncio.Queue[dict[str, JSONValue]]] = field(default_factory=set)
    last_decision_id: str | None = None
    decision_packet: dict[str, JSONValue] | None = None
    status_state: SessionStatus = "ok"
    model_provider: str | None = None
    model_id: str | None = None
    title_override: str | None = None
    folder_id: str | None = None
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
    ) -> None:
        self._storage: UISessionStorage = storage or InMemoryUISessionStorage()
        self._sessions: dict[str, _SessionState] = {}
        self._folders: dict[str, _FolderState] = {}
        self._session_ttl_seconds = session_ttl_seconds
        self._max_sessions = max_sessions
        self._max_messages_per_session = max_messages_per_session
        self._lock = asyncio.Lock()
        self._restore_sessions()
        self._restore_folders()

    async def get_or_create_session(self, session_id: str | None) -> str:
        normalized = session_id.strip() if session_id else ""
        async with self._lock:
            self._prune_sessions_locked(keep_session_id=normalized if normalized else None)
            if normalized:
                if normalized not in self._sessions:
                    self._sessions[normalized] = _SessionState()
                    self._persist_session_locked(normalized)
                return normalized
            new_id = uuid.uuid4().hex
            while new_id in self._sessions:
                new_id = uuid.uuid4().hex
            self._sessions[new_id] = _SessionState()
            self._persist_session_locked(new_id)
            self._prune_sessions_locked(keep_session_id=new_id)
            return new_id

    async def create_session(self) -> str:
        return await self.get_or_create_session(None)

    async def delete_session(self, session_id: str) -> bool:
        normalized = session_id.strip()
        if not normalized:
            return False
        async with self._lock:
            if normalized not in self._sessions:
                return False
            self._drop_sessions_locked([normalized])
            return True

    async def get_messages(self, session_id: str) -> list[dict[str, JSONValue]]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            return [dict(item) for item in state.messages]

    async def get_session_history(self, session_id: str) -> list[dict[str, JSONValue]] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return [dict(item) for item in state.messages]

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

    async def list_sessions(self) -> list[SessionListItem]:
        async with self._lock:
            self._prune_sessions_locked()
            items: list[_SessionListSortableItem] = []
            for session_id, state in self._sessions.items():
                title = state.title_override or self._build_session_title(state.messages)
                items.append(
                    {
                        "session_id": session_id,
                        "title": title,
                        "created_at": state.created_at,
                        "updated_at": datetime.fromisoformat(state.updated_at),
                        "message_count": len(state.messages),
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

    async def export_sessions(self) -> list[PersistedSession]:
        async with self._lock:
            self._prune_sessions_locked()
            items: list[PersistedSession] = []
            for session_id, state in self._sessions.items():
                items.append(
                    PersistedSession(
                        session_id=session_id,
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
        mode: Literal["replace", "merge"] = "replace",
    ) -> int:
        async with self._lock:
            if mode == "replace":
                self._drop_sessions_locked(list(self._sessions.keys()))
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
                subscribers = set(existing.subscribers) if existing is not None else set()
                self._sessions[item.session_id] = _SessionState(
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
            return {
                "session_id": session_id,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "status": status_value,
                "messages": [dict(item) for item in state.messages],
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
            }

    async def get_session_decision(self, session_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.decision_packet is None:
                return None
            return dict(state.decision_packet)

    async def set_session_model(self, session_id: str, provider: str, model_id: str) -> None:
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
            subscribers = list(state.subscribers)
            selected_model: dict[str, JSONValue] = {"provider": provider, "model": model_id}
            payload: dict[str, JSONValue] = {
                "session_id": session_id,
                "selected_model": selected_model,
            }
        event = self._build_event("session.model", payload)
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
            subscribers = list(state.subscribers)
            payload = self._status_payload(session_id, normalized)
        event = self._build_event("status", payload)
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

    async def append_message(
        self,
        session_id: str,
        message: dict[str, JSONValue],
    ) -> dict[str, JSONValue]:
        normalized_message = _normalize_message_payload(message)
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
            subscribers = list(state.subscribers)
        event = self._build_event(
            "message.append",
            {"session_id": session_id, "message": dict(normalized_message)},
        )
        self._publish_to_subscribers(subscribers, event)
        return dict(normalized_message)

    async def subscribe(self, session_id: str) -> asyncio.Queue[dict[str, JSONValue]]:
        queue: asyncio.Queue[dict[str, JSONValue]] = asyncio.Queue()
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
                self._persist_session_locked(session_id)
            state.subscribers.add(queue)
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
            state.subscribers.discard(queue)

    async def publish(self, session_id: str, event: dict[str, JSONValue]) -> None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            subscribers = list(state.subscribers)
        hydrated = self._ensure_event_fields(event)
        self._publish_to_subscribers(subscribers, hydrated)

    async def set_session_decision(
        self,
        session_id: str,
        decision: dict[str, JSONValue] | None,
    ) -> None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            state.decision_packet = dict(decision) if decision is not None else None
            state.updated_at = _utc_iso_now()
            self._persist_session_locked(session_id)
            self._prune_sessions_locked(keep_session_id=session_id)
            if state.decision_packet is None:
                state.last_decision_id = None
                return
            decision_id = state.decision_packet.get("id")
            if isinstance(decision_id, str):
                if decision_id == state.last_decision_id:
                    return
                state.last_decision_id = decision_id
            else:
                state.last_decision_id = None
            subscribers = list(state.subscribers)
            decision_payload = dict(state.decision_packet)
        event = self._build_event(
            "decision.packet",
            {"session_id": session_id, "decision": decision_payload},
        )
        self._publish_to_subscribers(subscribers, event)

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

    def _publish_to_subscribers(
        self,
        subscribers: list[asyncio.Queue[dict[str, JSONValue]]],
        event: dict[str, JSONValue],
    ) -> None:
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    def _restore_sessions(self) -> None:
        for item in self._storage.load_sessions():
            decision = dict(item.decision) if item.decision is not None else None
            last_decision_id: str | None = None
            if decision is not None:
                decision_id = decision.get("id")
                if isinstance(decision_id, str):
                    last_decision_id = decision_id
            self._sessions[item.session_id] = _SessionState(
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
        trace_id: str | None = None,
        parent_user_message_id: str | None = None,
        attachments: list[dict[str, str]] | None = None,
        message_id: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, JSONValue]:
        return _build_message(
            role=role,
            content=content,
            trace_id=trace_id,
            parent_user_message_id=parent_user_message_id,
            attachments=attachments,
            message_id=message_id,
            created_at=created_at,
        )
