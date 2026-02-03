from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from shared.models import JSONValue


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


@dataclass
class _SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    subscribers: set[asyncio.Queue[dict[str, JSONValue]]] = field(default_factory=set)
    last_decision_id: str | None = None
    decision_packet: dict[str, JSONValue] | None = None
    status_state: str = "ok"
    created_at: str = field(default_factory=_utc_iso_now)
    updated_at: str = field(default_factory=_utc_iso_now)


class UIHub:
    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_session(self, session_id: str | None) -> str:
        normalized = session_id.strip() if session_id else ""
        async with self._lock:
            if normalized:
                if normalized not in self._sessions:
                    self._sessions[normalized] = _SessionState()
                return normalized
            new_id = uuid.uuid4().hex
            while new_id in self._sessions:
                new_id = uuid.uuid4().hex
            self._sessions[new_id] = _SessionState()
            return new_id

    async def create_session(self) -> str:
        return await self.get_or_create_session(None)

    async def get_messages(self, session_id: str) -> list[dict[str, str]]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            return [dict(item) for item in state.messages]

    async def list_sessions(self) -> list[dict[str, JSONValue]]:
        async with self._lock:
            items: list[dict[str, JSONValue]] = []
            for session_id, state in self._sessions.items():
                items.append(
                    {
                        "session_id": session_id,
                        "created_at": state.created_at,
                        "updated_at": state.updated_at,
                        "message_count": len(state.messages),
                    }
                )
            items.sort(key=lambda item: str(item["updated_at"]), reverse=True)
            return items

    async def get_session(self, session_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            decision = dict(state.decision_packet) if state.decision_packet is not None else None
            return {
                "session_id": session_id,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "status": state.status_state,
                "messages": [dict(item) for item in state.messages],
                "decision": decision,
            }

    async def get_session_decision(self, session_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.decision_packet is None:
                return None
            return dict(state.decision_packet)

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
            payload = self._status_payload(session_id, state.status_state)
        return self._build_event("status", payload)

    async def append_message(self, session_id: str, role: str, content: str) -> dict[str, str]:
        message = {"role": role, "content": content}
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            state.messages.append(message)
            state.updated_at = _utc_iso_now()
            subscribers = list(state.subscribers)
        event = self._build_event(
            "message.append",
            {"session_id": session_id, "message": message},
        )
        self._publish_to_subscribers(subscribers, event)
        return message

    async def subscribe(self, session_id: str) -> asyncio.Queue[dict[str, JSONValue]]:
        queue: asyncio.Queue[dict[str, JSONValue]] = asyncio.Queue()
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
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

    def _status_payload(self, session_id: str, status_state: str) -> dict[str, JSONValue]:
        return {
            "session_id": session_id,
            "state": status_state,
            "ok": status_state != "error",
        }

    def _normalize_status(self, status_state: str) -> str:
        if status_state in {"ok", "busy", "error"}:
            return status_state
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
