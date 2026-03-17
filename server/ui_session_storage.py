from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from shared.models import JSONValue
from shared.sanitize import safe_json_loads


@dataclass(frozen=True)
class WorkspaceRecord:
    workspace_id: str
    principal_id: str
    title: str
    root_path: str | None
    policy: dict[str, JSONValue]
    tools_state: dict[str, bool] | None
    pending_decision: dict[str, JSONValue] | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ChatRecord:
    chat_id: str
    workspace_id: str
    title: str
    selected_model_provider: str | None = None
    selected_model_id: str | None = None
    artifacts: list[dict[str, JSONValue]] = field(default_factory=list)
    output_text: str | None = None
    output_updated_at: str | None = None
    files: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class MessageRecord:
    message_id: str
    chat_id: str
    lane: str
    role: str
    content: str
    created_at: str
    trace_id: str | None = None
    parent_user_message_id: str | None = None
    attachments: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeStateRecord:
    chat_id: str
    mode: str
    active_plan: dict[str, JSONValue] | None
    active_task: dict[str, JSONValue] | None
    auto_state: dict[str, JSONValue] | None
    decision: dict[str, JSONValue] | None
    status: str
    updated_at: str


@dataclass(frozen=True)
class WorkspaceActivityRecord:
    activity_id: str
    workspace_id: str
    kind: str
    summary: str
    payload: dict[str, JSONValue]
    created_at: str


@dataclass(frozen=True)
class PersistedSession:
    session_id: str
    created_at: str
    updated_at: str
    status: str
    decision: dict[str, JSONValue] | None
    messages: list[dict[str, JSONValue]]
    principal_id: str | None = None
    model_provider: str | None = None
    model_id: str | None = None
    title_override: str | None = None
    folder_id: str | None = None
    output_text: str | None = None
    output_updated_at: str | None = None
    files: list[str] = field(default_factory=list)
    artifacts: list[dict[str, JSONValue]] = field(default_factory=list)
    workspace_root: str | None = None
    policy_profile: str | None = None
    yolo_armed: bool = False
    yolo_armed_at: str | None = None
    tools_state: dict[str, bool] | None = None
    mode: str = "ask"
    active_plan: dict[str, JSONValue] | None = None
    active_task: dict[str, JSONValue] | None = None
    auto_state: dict[str, JSONValue] | None = None


@dataclass(frozen=True)
class PersistedFolder:
    folder_id: str
    name: str
    created_at: str
    updated_at: str


class UISessionStorage(Protocol):
    def load_sessions(self) -> list[PersistedSession]: ...

    def save_session(self, session: PersistedSession) -> None: ...

    def delete_sessions(self, session_ids: Sequence[str]) -> None: ...

    def load_workspaces(self) -> list[WorkspaceRecord]: ...

    def save_workspace(self, workspace: WorkspaceRecord) -> None: ...

    def delete_workspaces(self, workspace_ids: Sequence[str]) -> None: ...

    def load_chats(self) -> list[ChatRecord]: ...

    def save_chat(self, chat: ChatRecord) -> None: ...

    def delete_chats(self, chat_ids: Sequence[str]) -> None: ...

    def load_messages(self) -> list[MessageRecord]: ...

    def replace_chat_messages(self, chat_id: str, messages: Sequence[MessageRecord]) -> None: ...

    def delete_messages_for_chats(self, chat_ids: Sequence[str]) -> None: ...

    def load_runtime_states(self) -> list[RuntimeStateRecord]: ...

    def save_runtime_state(self, runtime_state: RuntimeStateRecord) -> None: ...

    def delete_runtime_states(self, chat_ids: Sequence[str]) -> None: ...

    def load_workspace_activity(self) -> list[WorkspaceActivityRecord]: ...

    def append_workspace_activity(self, activity: WorkspaceActivityRecord) -> None: ...

    def delete_workspace_activity(self, workspace_ids: Sequence[str]) -> None: ...


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _message_record_to_payload(message: MessageRecord) -> dict[str, JSONValue]:
    return {
        "message_id": message.message_id,
        "role": message.role,
        "lane": message.lane,
        "content": message.content,
        "created_at": message.created_at,
        "trace_id": message.trace_id,
        "parent_user_message_id": message.parent_user_message_id,
        "attachments": [dict(item) for item in message.attachments],
    }


def _payload_to_message_record(chat_id: str, payload: dict[str, JSONValue]) -> MessageRecord:
    attachments_raw = payload.get("attachments")
    attachments: list[dict[str, str]] = []
    if isinstance(attachments_raw, list):
        for item in attachments_raw:
            if not isinstance(item, dict):
                continue
            name = _optional_str(item.get("name"))
            mime = _optional_str(item.get("mime"))
            content_item = item.get("content")
            if name is None or mime is None or not isinstance(content_item, str):
                continue
            attachments.append({"name": name, "mime": mime, "content": content_item})
    return MessageRecord(
        message_id=_optional_str(payload.get("message_id")) or "",
        chat_id=chat_id,
        lane=_optional_str(payload.get("lane")) or "chat",
        role=_optional_str(payload.get("role")) or "user",
        content=str(payload.get("content") or ""),
        created_at=_optional_str(payload.get("created_at")) or "",
        trace_id=_optional_str(payload.get("trace_id")),
        parent_user_message_id=_optional_str(payload.get("parent_user_message_id")),
        attachments=attachments,
    )


def _safe_json_object(value: object) -> dict[str, JSONValue] | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return None
    loaded = safe_json_loads(value)
    if isinstance(loaded, dict):
        return {str(key): item for key, item in loaded.items()}
    return None


def _safe_json_bool_map(value: object) -> dict[str, bool] | None:
    raw = _safe_json_object(value)
    if raw is None:
        return None
    normalized: dict[str, bool] = {}
    for key, item in raw.items():
        if isinstance(item, bool):
            normalized[key] = item
    return normalized


def _safe_json_list(value: object) -> list[JSONValue]:
    if value in (None, ""):
        return []
    if not isinstance(value, str):
        return []
    loaded = safe_json_loads(value)
    if isinstance(loaded, list):
        return list(loaded)
    return []


def _safe_string_list(value: object) -> list[str]:
    return [item for item in _safe_json_list(value) if isinstance(item, str) and item.strip()]


def _safe_object_list(value: object) -> list[dict[str, JSONValue]]:
    normalized: list[dict[str, JSONValue]] = []
    for item in _safe_json_list(value):
        if isinstance(item, dict):
            normalized.append({str(key): val for key, val in item.items()})
    return normalized


def _safe_attachment_list(value: object) -> list[dict[str, str]]:
    attachments: list[dict[str, str]] = []
    for item in _safe_json_list(value):
        if not isinstance(item, dict):
            continue
        name = _optional_str(item.get("name"))
        mime = _optional_str(item.get("mime"))
        content = item.get("content")
        if name is None or mime is None or not isinstance(content, str):
            continue
        attachments.append({"name": name, "mime": mime, "content": content})
    return attachments


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class InMemoryUISessionStorage:
    def __init__(self) -> None:
        self._workspaces: dict[str, WorkspaceRecord] = {}
        self._chats: dict[str, ChatRecord] = {}
        self._messages: dict[str, list[MessageRecord]] = {}
        self._runtime_states: dict[str, RuntimeStateRecord] = {}
        self._activity: dict[str, list[WorkspaceActivityRecord]] = {}

    def load_workspaces(self) -> list[WorkspaceRecord]:
        return list(self._workspaces.values())

    def save_workspace(self, workspace: WorkspaceRecord) -> None:
        self._workspaces[workspace.workspace_id] = workspace

    def delete_workspaces(self, workspace_ids: Sequence[str]) -> None:
        for workspace_id in workspace_ids:
            self._workspaces.pop(workspace_id, None)
            self._activity.pop(workspace_id, None)

    def load_chats(self) -> list[ChatRecord]:
        return list(self._chats.values())

    def save_chat(self, chat: ChatRecord) -> None:
        self._chats[chat.chat_id] = chat

    def delete_chats(self, chat_ids: Sequence[str]) -> None:
        for chat_id in chat_ids:
            self._chats.pop(chat_id, None)
            self._messages.pop(chat_id, None)
            self._runtime_states.pop(chat_id, None)

    def load_messages(self) -> list[MessageRecord]:
        rows: list[MessageRecord] = []
        for items in self._messages.values():
            rows.extend(items)
        return rows

    def replace_chat_messages(self, chat_id: str, messages: Sequence[MessageRecord]) -> None:
        self._messages[chat_id] = list(messages)

    def delete_messages_for_chats(self, chat_ids: Sequence[str]) -> None:
        for chat_id in chat_ids:
            self._messages.pop(chat_id, None)

    def load_runtime_states(self) -> list[RuntimeStateRecord]:
        return list(self._runtime_states.values())

    def save_runtime_state(self, runtime_state: RuntimeStateRecord) -> None:
        self._runtime_states[runtime_state.chat_id] = runtime_state

    def delete_runtime_states(self, chat_ids: Sequence[str]) -> None:
        for chat_id in chat_ids:
            self._runtime_states.pop(chat_id, None)

    def load_workspace_activity(self) -> list[WorkspaceActivityRecord]:
        rows: list[WorkspaceActivityRecord] = []
        for items in self._activity.values():
            rows.extend(items)
        return rows

    def append_workspace_activity(self, activity: WorkspaceActivityRecord) -> None:
        self._activity.setdefault(activity.workspace_id, []).append(activity)

    def delete_workspace_activity(self, workspace_ids: Sequence[str]) -> None:
        for workspace_id in workspace_ids:
            self._activity.pop(workspace_id, None)

    def load_sessions(self) -> list[PersistedSession]:
        sessions: list[PersistedSession] = []
        for chat in self._chats.values():
            runtime = self._runtime_states.get(chat.chat_id)
            workspace = self._workspaces.get(chat.workspace_id)
            sessions.append(
                PersistedSession(
                    session_id=chat.chat_id,
                    principal_id=workspace.principal_id if workspace is not None else None,
                    created_at=chat.created_at,
                    updated_at=chat.updated_at,
                    status=runtime.status if runtime is not None else "ok",
                    decision=runtime.decision if runtime is not None else None,
                    messages=[
                        _message_record_to_payload(item)
                        for item in self._messages.get(chat.chat_id, [])
                    ],
                    model_provider=chat.selected_model_provider,
                    model_id=chat.selected_model_id,
                    title_override=chat.title,
                    output_text=chat.output_text,
                    output_updated_at=chat.output_updated_at,
                    files=list(chat.files),
                    artifacts=[dict(item) for item in chat.artifacts],
                    workspace_root=workspace.root_path if workspace is not None else None,
                    policy_profile=(
                        _optional_str((workspace.policy or {}).get("profile"))
                        if workspace is not None
                        else None
                    ),
                    yolo_armed=(
                        bool((workspace.policy or {}).get("yolo_armed")) if workspace else False
                    ),
                    yolo_armed_at=(
                        _optional_str((workspace.policy or {}).get("yolo_armed_at"))
                        if workspace is not None
                        else None
                    ),
                    tools_state=workspace.tools_state if workspace is not None else None,
                    mode=runtime.mode if runtime is not None else "ask",
                    active_plan=runtime.active_plan if runtime is not None else None,
                    active_task=runtime.active_task if runtime is not None else None,
                    auto_state=runtime.auto_state if runtime is not None else None,
                )
            )
        return sessions

    def save_session(self, session: PersistedSession) -> None:
        workspace_id = session.session_id
        self.save_workspace(
            WorkspaceRecord(
                workspace_id=workspace_id,
                principal_id=session.principal_id or "legacy",
                title=session.title_override or "Workspace",
                root_path=session.workspace_root,
                policy={
                    "profile": session.policy_profile or "sandbox",
                    "yolo_armed": session.yolo_armed,
                    "yolo_armed_at": session.yolo_armed_at,
                },
                tools_state=(
                    dict(session.tools_state) if isinstance(session.tools_state, dict) else None
                ),
                pending_decision=None,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
        )
        self.save_chat(
            ChatRecord(
                chat_id=session.session_id,
                workspace_id=workspace_id,
                title=session.title_override or "New Chat",
                selected_model_provider=session.model_provider,
                selected_model_id=session.model_id,
                artifacts=[dict(item) for item in session.artifacts],
                output_text=session.output_text,
                output_updated_at=session.output_updated_at,
                files=list(session.files),
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
        )
        self.replace_chat_messages(
            session.session_id,
            [_payload_to_message_record(session.session_id, item) for item in session.messages],
        )
        self.save_runtime_state(
            RuntimeStateRecord(
                chat_id=session.session_id,
                mode=session.mode,
                active_plan=session.active_plan,
                active_task=session.active_task,
                auto_state=session.auto_state,
                decision=session.decision,
                status=session.status,
                updated_at=session.updated_at,
            )
        )

    def delete_sessions(self, session_ids: Sequence[str]) -> None:
        self.delete_chats(session_ids)
        self.delete_runtime_states(session_ids)
        self.delete_messages_for_chats(session_ids)


class SQLiteUISessionStorage:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._initialize_schema()

    def load_workspaces(self) -> list[WorkspaceRecord]:
        rows: list[WorkspaceRecord] = []
        with self._connect() as conn:
            result = conn.execute(
                """
                SELECT workspace_id, principal_id, title, root_path, policy_json, tools_state_json,
                       pending_decision_json, created_at, updated_at
                FROM ui_workspaces
                ORDER BY updated_at DESC
                """
            ).fetchall()
        for row in result:
            rows.append(
                WorkspaceRecord(
                    workspace_id=str(row["workspace_id"]),
                    principal_id=str(row["principal_id"]),
                    title=str(row["title"]),
                    root_path=_optional_str(row["root_path"]),
                    policy=_safe_json_object(row["policy_json"]) or {},
                    tools_state=_safe_json_bool_map(row["tools_state_json"]),
                    pending_decision=_safe_json_object(row["pending_decision_json"]),
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
            )
        return rows

    def save_workspace(self, workspace: WorkspaceRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_workspaces (
                    workspace_id, principal_id, title, root_path, policy_json,
                    tools_state_json, pending_decision_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id)
                DO UPDATE SET
                    principal_id=excluded.principal_id,
                    title=excluded.title,
                    root_path=excluded.root_path,
                    policy_json=excluded.policy_json,
                    tools_state_json=excluded.tools_state_json,
                    pending_decision_json=excluded.pending_decision_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    workspace.workspace_id,
                    workspace.principal_id,
                    workspace.title,
                    workspace.root_path,
                    _json_dumps(workspace.policy),
                    _json_dumps(workspace.tools_state or {}),
                    _json_dumps(workspace.pending_decision)
                    if workspace.pending_decision is not None
                    else None,
                    workspace.created_at,
                    workspace.updated_at,
                ),
            )

    def delete_workspaces(self, workspace_ids: Sequence[str]) -> None:
        if not workspace_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM ui_workspaces WHERE workspace_id = ?",
                [(workspace_id,) for workspace_id in workspace_ids],
            )

    def load_chats(self) -> list[ChatRecord]:
        rows: list[ChatRecord] = []
        with self._connect() as conn:
            result = conn.execute(
                """
                SELECT
                    chat_id,
                    workspace_id,
                    title,
                    selected_model_provider,
                    selected_model_id,
                    artifacts_json,
                    output_text,
                    output_updated_at,
                    files_json,
                    created_at,
                    updated_at
                FROM ui_chats
                ORDER BY updated_at DESC
                """
            ).fetchall()
        for row in result:
            rows.append(
                ChatRecord(
                    chat_id=str(row["chat_id"]),
                    workspace_id=str(row["workspace_id"]),
                    title=str(row["title"]),
                    selected_model_provider=_optional_str(row["selected_model_provider"]),
                    selected_model_id=_optional_str(row["selected_model_id"]),
                    artifacts=_safe_object_list(row["artifacts_json"]),
                    output_text=_optional_str(row["output_text"]),
                    output_updated_at=_optional_str(row["output_updated_at"]),
                    files=_safe_string_list(row["files_json"]),
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
            )
        return rows

    def save_chat(self, chat: ChatRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_chats (
                    chat_id,
                    workspace_id,
                    title,
                    selected_model_provider,
                    selected_model_id,
                    artifacts_json,
                    output_text,
                    output_updated_at,
                    files_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id)
                DO UPDATE SET
                    workspace_id=excluded.workspace_id,
                    title=excluded.title,
                    selected_model_provider=excluded.selected_model_provider,
                    selected_model_id=excluded.selected_model_id,
                    artifacts_json=excluded.artifacts_json,
                    output_text=excluded.output_text,
                    output_updated_at=excluded.output_updated_at,
                    files_json=excluded.files_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    chat.chat_id,
                    chat.workspace_id,
                    chat.title,
                    chat.selected_model_provider,
                    chat.selected_model_id,
                    _json_dumps(chat.artifacts),
                    chat.output_text,
                    chat.output_updated_at,
                    _json_dumps(chat.files),
                    chat.created_at,
                    chat.updated_at,
                ),
            )

    def delete_chats(self, chat_ids: Sequence[str]) -> None:
        if not chat_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM ui_chats WHERE chat_id = ?",
                [(chat_id,) for chat_id in chat_ids],
            )

    def load_messages(self) -> list[MessageRecord]:
        rows: list[MessageRecord] = []
        with self._connect() as conn:
            result = conn.execute(
                """
                SELECT message_id, chat_id, lane, role, content, created_at, trace_id,
                       parent_user_message_id, attachments_json
                FROM ui_messages
                ORDER BY created_at ASC
                """
            ).fetchall()
        for row in result:
            rows.append(
                MessageRecord(
                    message_id=str(row["message_id"]),
                    chat_id=str(row["chat_id"]),
                    lane=_optional_str(row["lane"]) or "chat",
                    role=str(row["role"]),
                    content=str(row["content"]),
                    created_at=str(row["created_at"]),
                    trace_id=_optional_str(row["trace_id"]),
                    parent_user_message_id=_optional_str(row["parent_user_message_id"]),
                    attachments=_safe_attachment_list(row["attachments_json"]),
                )
            )
        return rows

    def replace_chat_messages(self, chat_id: str, messages: Sequence[MessageRecord]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ui_messages WHERE chat_id = ?", (chat_id,))
            conn.executemany(
                """
                INSERT INTO ui_messages (
                    message_id, chat_id, lane, role, content, created_at, trace_id,
                    parent_user_message_id, attachments_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        message.message_id,
                        message.chat_id,
                        message.lane,
                        message.role,
                        message.content,
                        message.created_at,
                        message.trace_id,
                        message.parent_user_message_id,
                        _json_dumps(message.attachments),
                    )
                    for message in messages
                ],
            )

    def delete_messages_for_chats(self, chat_ids: Sequence[str]) -> None:
        if not chat_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM ui_messages WHERE chat_id = ?",
                [(chat_id,) for chat_id in chat_ids],
            )

    def load_runtime_states(self) -> list[RuntimeStateRecord]:
        rows: list[RuntimeStateRecord] = []
        with self._connect() as conn:
            result = conn.execute(
                """
                SELECT chat_id, mode, active_plan_json, active_task_json, auto_state_json,
                       decision_json, status, updated_at
                FROM ui_runtime_states
                ORDER BY updated_at DESC
                """
            ).fetchall()
        for row in result:
            rows.append(
                RuntimeStateRecord(
                    chat_id=str(row["chat_id"]),
                    mode=str(row["mode"]),
                    active_plan=_safe_json_object(row["active_plan_json"]),
                    active_task=_safe_json_object(row["active_task_json"]),
                    auto_state=_safe_json_object(row["auto_state_json"]),
                    decision=_safe_json_object(row["decision_json"]),
                    status=str(row["status"]),
                    updated_at=str(row["updated_at"]),
                )
            )
        return rows

    def save_runtime_state(self, runtime_state: RuntimeStateRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_runtime_states (
                    chat_id, mode, active_plan_json, active_task_json, auto_state_json,
                    decision_json, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id)
                DO UPDATE SET
                    mode=excluded.mode,
                    active_plan_json=excluded.active_plan_json,
                    active_task_json=excluded.active_task_json,
                    auto_state_json=excluded.auto_state_json,
                    decision_json=excluded.decision_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    runtime_state.chat_id,
                    runtime_state.mode,
                    _json_dumps(runtime_state.active_plan)
                    if runtime_state.active_plan is not None
                    else None,
                    _json_dumps(runtime_state.active_task)
                    if runtime_state.active_task is not None
                    else None,
                    _json_dumps(runtime_state.auto_state)
                    if runtime_state.auto_state is not None
                    else None,
                    _json_dumps(runtime_state.decision)
                    if runtime_state.decision is not None
                    else None,
                    runtime_state.status,
                    runtime_state.updated_at,
                ),
            )

    def delete_runtime_states(self, chat_ids: Sequence[str]) -> None:
        if not chat_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM ui_runtime_states WHERE chat_id = ?",
                [(chat_id,) for chat_id in chat_ids],
            )

    def load_workspace_activity(self) -> list[WorkspaceActivityRecord]:
        rows: list[WorkspaceActivityRecord] = []
        with self._connect() as conn:
            result = conn.execute(
                """
                SELECT activity_id, workspace_id, kind, summary, payload_json, created_at
                FROM ui_workspace_activity
                ORDER BY created_at ASC
                """
            ).fetchall()
        for row in result:
            rows.append(
                WorkspaceActivityRecord(
                    activity_id=str(row["activity_id"]),
                    workspace_id=str(row["workspace_id"]),
                    kind=str(row["kind"]),
                    summary=str(row["summary"]),
                    payload=_safe_json_object(row["payload_json"]) or {},
                    created_at=str(row["created_at"]),
                )
            )
        return rows

    def append_workspace_activity(self, activity: WorkspaceActivityRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_workspace_activity (
                    activity_id, workspace_id, kind, summary, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    activity.activity_id,
                    activity.workspace_id,
                    activity.kind,
                    activity.summary,
                    _json_dumps(activity.payload),
                    activity.created_at,
                ),
            )

    def delete_workspace_activity(self, workspace_ids: Sequence[str]) -> None:
        if not workspace_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM ui_workspace_activity WHERE workspace_id = ?",
                [(workspace_id,) for workspace_id in workspace_ids],
            )

    def load_sessions(self) -> list[PersistedSession]:
        sessions: list[PersistedSession] = []
        chats = {chat.chat_id: chat for chat in self.load_chats()}
        workspaces = {workspace.workspace_id: workspace for workspace in self.load_workspaces()}
        runtimes = {runtime.chat_id: runtime for runtime in self.load_runtime_states()}
        messages_by_chat: dict[str, list[dict[str, JSONValue]]] = {}
        for message in self.load_messages():
            messages_by_chat.setdefault(message.chat_id, []).append(
                _message_record_to_payload(message)
            )
        for chat_id, chat in chats.items():
            workspace = workspaces.get(chat.workspace_id)
            runtime = runtimes.get(chat_id)
            sessions.append(
                PersistedSession(
                    session_id=chat_id,
                    principal_id=workspace.principal_id if workspace is not None else None,
                    created_at=chat.created_at,
                    updated_at=chat.updated_at,
                    status=runtime.status if runtime is not None else "ok",
                    decision=runtime.decision if runtime is not None else None,
                    messages=messages_by_chat.get(chat_id, []),
                    model_provider=chat.selected_model_provider,
                    model_id=chat.selected_model_id,
                    title_override=chat.title,
                    output_text=chat.output_text,
                    output_updated_at=chat.output_updated_at,
                    files=list(chat.files),
                    artifacts=[dict(item) for item in chat.artifacts],
                    workspace_root=workspace.root_path if workspace is not None else None,
                    policy_profile=(
                        _optional_str((workspace.policy or {}).get("profile"))
                        if workspace is not None
                        else None
                    ),
                    yolo_armed=(
                        bool((workspace.policy or {}).get("yolo_armed")) if workspace else False
                    ),
                    yolo_armed_at=(
                        _optional_str((workspace.policy or {}).get("yolo_armed_at"))
                        if workspace is not None
                        else None
                    ),
                    tools_state=workspace.tools_state if workspace is not None else None,
                    mode=runtime.mode if runtime is not None else "ask",
                    active_plan=runtime.active_plan if runtime is not None else None,
                    active_task=runtime.active_task if runtime is not None else None,
                    auto_state=runtime.auto_state if runtime is not None else None,
                )
            )
        return sessions

    def save_session(self, session: PersistedSession) -> None:
        workspace_id = session.session_id
        self.save_workspace(
            WorkspaceRecord(
                workspace_id=workspace_id,
                principal_id=session.principal_id or "legacy",
                title=session.title_override or "Workspace",
                root_path=session.workspace_root,
                policy={
                    "profile": session.policy_profile or "sandbox",
                    "yolo_armed": session.yolo_armed,
                    "yolo_armed_at": session.yolo_armed_at,
                },
                tools_state=(
                    dict(session.tools_state) if isinstance(session.tools_state, dict) else None
                ),
                pending_decision=None,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
        )
        self.save_chat(
            ChatRecord(
                chat_id=session.session_id,
                workspace_id=workspace_id,
                title=session.title_override or "New Chat",
                selected_model_provider=session.model_provider,
                selected_model_id=session.model_id,
                artifacts=[dict(item) for item in session.artifacts],
                output_text=session.output_text,
                output_updated_at=session.output_updated_at,
                files=list(session.files),
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
        )
        self.replace_chat_messages(
            session.session_id,
            [_payload_to_message_record(session.session_id, item) for item in session.messages],
        )
        self.save_runtime_state(
            RuntimeStateRecord(
                chat_id=session.session_id,
                mode=session.mode,
                active_plan=session.active_plan,
                active_task=session.active_task,
                auto_state=session.auto_state,
                decision=session.decision,
                status=session.status,
                updated_at=session.updated_at,
            )
        )

    def delete_sessions(self, session_ids: Sequence[str]) -> None:
        self.delete_chats(session_ids)
        self.delete_runtime_states(session_ids)
        self.delete_messages_for_chats(session_ids)

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ui_workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    root_path TEXT,
                    policy_json TEXT NOT NULL,
                    tools_state_json TEXT,
                    pending_decision_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ui_chats (
                    chat_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    selected_model_provider TEXT,
                    selected_model_id TEXT,
                    artifacts_json TEXT NOT NULL,
                    output_text TEXT,
                    output_updated_at TEXT,
                    files_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(workspace_id) REFERENCES ui_workspaces(workspace_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ui_messages (
                    message_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    lane TEXT NOT NULL DEFAULT 'chat',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    trace_id TEXT,
                    parent_user_message_id TEXT,
                    attachments_json TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES ui_chats(chat_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ui_runtime_states (
                    chat_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    active_plan_json TEXT,
                    active_task_json TEXT,
                    auto_state_json TEXT,
                    decision_json TEXT,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES ui_chats(chat_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ui_workspace_activity (
                    activity_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(workspace_id) REFERENCES ui_workspaces(workspace_id)
                        ON DELETE CASCADE
                );
                """
            )
            needs_reset = False
            try:
                columns = {
                    str(row["name"])
                    for row in conn.execute("PRAGMA table_info(ui_messages)").fetchall()
                }
                if "chat_id" not in columns or "lane" not in columns:
                    needs_reset = True
            except sqlite3.DatabaseError:
                needs_reset = True
            if needs_reset:
                conn.executescript(
                    """
                    DROP TABLE IF EXISTS ui_runtime_states;
                    DROP TABLE IF EXISTS ui_workspace_activity;
                    DROP TABLE IF EXISTS ui_messages;
                    DROP TABLE IF EXISTS ui_chats;
                    DROP TABLE IF EXISTS ui_workspaces;
                    DROP TABLE IF EXISTS ui_sessions;

                    CREATE TABLE ui_workspaces (
                        workspace_id TEXT PRIMARY KEY,
                        principal_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        root_path TEXT,
                        policy_json TEXT NOT NULL,
                        tools_state_json TEXT,
                        pending_decision_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE ui_chats (
                        chat_id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        selected_model_provider TEXT,
                        selected_model_id TEXT,
                        artifacts_json TEXT NOT NULL,
                        output_text TEXT,
                        output_updated_at TEXT,
                        files_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(workspace_id) REFERENCES ui_workspaces(workspace_id)
                            ON DELETE CASCADE
                    );

                    CREATE TABLE ui_messages (
                        message_id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        lane TEXT NOT NULL DEFAULT 'chat',
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        trace_id TEXT,
                        parent_user_message_id TEXT,
                        attachments_json TEXT NOT NULL,
                        FOREIGN KEY(chat_id) REFERENCES ui_chats(chat_id) ON DELETE CASCADE
                    );

                    CREATE TABLE ui_runtime_states (
                        chat_id TEXT PRIMARY KEY,
                        mode TEXT NOT NULL,
                        active_plan_json TEXT,
                        active_task_json TEXT,
                        auto_state_json TEXT,
                        decision_json TEXT,
                        status TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(chat_id) REFERENCES ui_chats(chat_id) ON DELETE CASCADE
                    );

                    CREATE TABLE ui_workspace_activity (
                        activity_id TEXT PRIMARY KEY,
                        workspace_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(workspace_id) REFERENCES ui_workspaces(workspace_id)
                            ON DELETE CASCADE
                    );
                    """
                )
