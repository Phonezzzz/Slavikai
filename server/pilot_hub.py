from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from shared.models import JSONValue


@dataclass
class _SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    subscribers: set[asyncio.Queue[dict[str, JSONValue]]] = field(default_factory=set)
    last_decision_id: str | None = None


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


class PilotHub:
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

    async def get_messages(self, session_id: str) -> list[dict[str, str]]:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return []
            return [dict(item) for item in state.messages]

    async def append_message(self, session_id: str, role: str, content: str) -> dict[str, str]:
        message = {"role": role, "content": content}
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            state.messages.append(message)
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

    async def maybe_publish_decision(
        self,
        session_id: str,
        decision: dict[str, JSONValue],
    ) -> None:
        decision_id = decision.get("id")
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = _SessionState()
                self._sessions[session_id] = state
            if isinstance(decision_id, str):
                if decision_id == state.last_decision_id:
                    return
                state.last_decision_id = decision_id
            else:
                state.last_decision_id = None
            subscribers = list(state.subscribers)
        event = self._build_event(
            "decision.packet",
            {"session_id": session_id, "decision": decision},
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
