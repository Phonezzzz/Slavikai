from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from config.model_whitelist import ModelNotAllowedError
from core.approval_policy import ApprovalPrompt, ApprovalRequest
from core.tracer import Tracer
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from server.http_api import create_app
from server.ui_session_storage import InMemoryUISessionStorage
from shared.models import LLMMessage


class DummyBrain(Brain):
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text=self._text)


def _forbidden_model_config(provider: str) -> ModelConfig:
    return ModelConfig(provider=provider, model="forbidden-model")


class DummyAgent:
    def __init__(self, trace_path: Path) -> None:
        self.tracer = Tracer(path=trace_path)
        self.tools_enabled = {"safe_mode": True}
        self.brain = DummyBrain("ok")
        self.last_approval_request: ApprovalRequest | None = None
        self.last_chat_interaction_id: str | None = None
        self._feedback_events: list[dict[str, object]] = []
        self.session_id: str | None = None
        self.approved_categories: set[str] = set()

    def set_session_context(self, session_id: str | None, approved_categories: set[str]) -> None:
        self.session_id = session_id
        self.approved_categories = set(approved_categories)

    def respond(self, messages) -> str:
        content = messages[-1].content if messages else ""
        self.tracer.log("user_input", content)
        self.last_approval_request = None
        response_text = "ok"

        required = {"SUDO", "EXEC_ARBITRARY"}
        if "danger" in content.lower() and not required.issubset(self.approved_categories):
            prompt = ApprovalPrompt(
                what="Выполнить команду: rm -rf /",
                why="Для выполнения запроса пользователя.",
                risk="Повышенные права, риск для системы.",
                changes=["Команда: rm -rf /"],
            )
            self.last_approval_request = ApprovalRequest(
                category="SUDO",
                required_categories=["SUDO", "EXEC_ARBITRARY"],
                prompt=prompt,
                tool="shell",
                details={"command": "rm -rf /"},
                session_id=self.session_id,
            )
            response_text = "[Требуется подтверждение действия]"
        interaction_id = str(uuid.uuid4())
        self.last_chat_interaction_id = interaction_id
        self.tracer.log(
            "interaction_logged",
            "Chat interaction stored",
            {"interaction_id": interaction_id},
        )
        return response_text

    def record_feedback_event(
        self,
        *,
        interaction_id: str,
        rating,
        labels=None,
        free_text=None,
    ) -> None:
        self._feedback_events.append(
            {
                "interaction_id": interaction_id,
                "rating": rating,
                "labels": labels or [],
                "free_text": free_text,
            }
        )


async def _create_client(agent: DummyAgent, trace_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr("server.http_api.TRACE_LOG", trace_path)
    app = create_app(
        agent=agent,
        max_request_bytes=1_000_000,
        ui_storage=InMemoryUISessionStorage(),
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


async def _create_client_without_agent(trace_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr("server.http_api.TRACE_LOG", trace_path)
    app = create_app(
        agent=None,
        max_request_bytes=1_000_000,
        ui_storage=InMemoryUISessionStorage(),
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


def test_models_endpoint(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    agent = DummyAgent(trace_path)

    async def run() -> None:
        client = await _create_client(agent, trace_path, monkeypatch)
        try:
            resp = await client.get("/v1/models")
            assert resp.status == 200
            payload = await resp.json()
            ids = [item.get("id") for item in payload.get("data", [])]
            assert ids == ["slavik"]
        finally:
            await client.close()

    asyncio.run(run())


def test_chat_completions_returns_meta(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    agent = DummyAgent(trace_path)

    async def run() -> None:
        client = await _create_client(agent, trace_path, monkeypatch)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "Привет"}],
                    "stream": False,
                },
            )
            assert resp.status == 200
            payload = await resp.json()
            meta = payload.get("slavik_meta", {})
            assert isinstance(meta.get("trace_id"), str)
            assert isinstance(meta.get("session_id"), str)
            assert meta.get("safe_mode") is True
        finally:
            await client.close()

    asyncio.run(run())


def test_stream_not_supported(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    agent = DummyAgent(trace_path)

    async def run() -> None:
        client = await _create_client(agent, trace_path, monkeypatch)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "Привет"}],
                    "stream": True,
                },
            )
            assert resp.status == 400
            payload = await resp.json()
            error = payload.get("error", {})
            assert error.get("type") == "not_supported"
            assert error.get("code") == "streaming_not_supported"
        finally:
            await client.close()

    asyncio.run(run())


def test_unknown_sampling_param_logged(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    agent = DummyAgent(trace_path)

    async def run() -> None:
        client = await _create_client(agent, trace_path, monkeypatch)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "Объясни VectorIndex"}],
                    "top_k": 50,
                },
            )
            assert resp.status == 200
            payload = await resp.json()
            trace_id = payload.get("slavik_meta", {}).get("trace_id")
            assert isinstance(trace_id, str)

            trace_resp = await client.get(f"/slavik/trace/{trace_id}")
            assert trace_resp.status == 200
            trace_payload = await trace_resp.json()
            events = trace_payload.get("events", [])
            assert any(event.get("event") == "request_warning" for event in events)
        finally:
            await client.close()

    asyncio.run(run())


def test_unknown_structural_field_rejected(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    agent = DummyAgent(trace_path)

    async def run() -> None:
        client = await _create_client(agent, trace_path, monkeypatch)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "Привет"}],
                    "unknown_field": 123,
                },
            )
            assert resp.status == 400
            payload = await resp.json()
            error = payload.get("error", {})
            assert error.get("type") == "invalid_request_error"
        finally:
            await client.close()

    asyncio.run(run())


def test_create_app_without_agent_does_not_raise(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"

    async def run() -> None:
        client = await _create_client_without_agent(trace_path, monkeypatch)
        try:
            resp = await client.get("/v1/models")
            assert resp.status == 200
        finally:
            await client.close()

    asyncio.run(run())


def test_chat_completions_returns_409_when_model_not_selected(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    monkeypatch.setattr("core.agent.load_model_configs", lambda: None)

    async def run() -> None:
        client = await _create_client_without_agent(trace_path, monkeypatch)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "Привет"}],
                    "stream": False,
                },
            )
            assert resp.status == 409
            payload = await resp.json()
            error = payload.get("error", {})
            assert error.get("code") == "model_not_selected"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openrouter"])
def test_chat_completions_returns_409_when_model_not_whitelisted(
    monkeypatch,
    tmp_path,
    provider: str,
) -> None:
    trace_path = tmp_path / "trace.log"
    monkeypatch.setattr(
        "core.agent.load_model_configs",
        lambda: _forbidden_model_config(provider),
    )

    async def run() -> None:
        client = await _create_client_without_agent(trace_path, monkeypatch)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "Привет"}],
                    "stream": False,
                },
            )
            assert resp.status == 409
            payload = await resp.json()
            error = payload.get("error", {})
            assert error.get("type") == "configuration_error"
            assert error.get("code") == "model_not_allowed"
            details = error.get("details")
            assert isinstance(details, dict)
            assert details.get("model") == "forbidden-model"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("provider", ["openrouter"])
def test_agent_init_fails_when_model_not_whitelisted(monkeypatch, provider: str) -> None:
    monkeypatch.setattr(
        "core.agent.load_model_configs",
        lambda: _forbidden_model_config(provider),
    )

    with pytest.raises(ModelNotAllowedError):
        from core.agent import Agent

        Agent(brain=DummyBrain("ok"))


def test_trace_endpoint_returns_events(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    agent = DummyAgent(trace_path)

    async def run() -> None:
        client = await _create_client(agent, trace_path, monkeypatch)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "Тест"}],
                },
            )
            assert resp.status == 200
            payload = await resp.json()
            trace_id = payload.get("slavik_meta", {}).get("trace_id")
            assert isinstance(trace_id, str)

            trace_resp = await client.get(f"/slavik/trace/{trace_id}")
            assert trace_resp.status == 200
            trace_payload = await trace_resp.json()
            events = trace_payload.get("events", [])
            assert isinstance(events, list)
            assert events
        finally:
            await client.close()

    asyncio.run(run())


def test_approve_session_reflected(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    agent = DummyAgent(trace_path)

    async def run() -> None:
        client = await _create_client(agent, trace_path, monkeypatch)
        try:
            session_id = "session-123"
            approve_resp = await client.post(
                "/slavik/approve-session",
                json={"session_id": session_id, "categories": ["EXEC_ARBITRARY"]},
            )
            assert approve_resp.status == 200
            approve_payload = await approve_resp.json()
            assert approve_payload.get("session_approved") is True
            assert approve_payload.get("approved_categories") == ["EXEC_ARBITRARY"]

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "Привет"}],
                },
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            meta = payload.get("slavik_meta", {})
            assert meta.get("session_approved") is True
        finally:
            await client.close()

    asyncio.run(run())


def test_approval_required_response(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    agent = DummyAgent(trace_path)

    async def run() -> None:
        client = await _create_client(agent, trace_path, monkeypatch)
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "danger action"}],
                },
            )
            assert resp.status == 400
            payload = await resp.json()
            error = payload.get("error", {})
            assert error.get("code") == "approval_required"
            details = error.get("details", {})
            assert details.get("category") == "SUDO"
            assert "prompt" in details
        finally:
            await client.close()

    asyncio.run(run())


def test_approval_flow_and_session_reset(monkeypatch, tmp_path) -> None:
    trace_path = tmp_path / "trace.log"
    agent = DummyAgent(trace_path)

    async def run() -> None:
        client = await _create_client(agent, trace_path, monkeypatch)
        try:
            session_id = "session-approve"
            first = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "danger action"}],
                },
                headers={"X-Slavik-Session": session_id},
            )
            assert first.status == 400

            approve = await client.post(
                "/slavik/approve-session",
                json={
                    "session_id": session_id,
                    "categories": ["SUDO", "EXEC_ARBITRARY"],
                },
            )
            assert approve.status == 200

            second = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "danger action"}],
                },
                headers={"X-Slavik-Session": session_id},
            )
            assert second.status == 200

            new_session = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "danger action"}],
                },
                headers={"X-Slavik-Session": "session-new"},
            )
            assert new_session.status == 400
        finally:
            await client.close()

    asyncio.run(run())
