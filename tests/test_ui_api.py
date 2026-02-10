from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from config.memory_config import (
    load_memory_config as load_memory_config_from_path,
)
from config.memory_config import (
    save_memory_config as save_memory_config_to_path,
)
from config.tools_config import (
    load_tools_config as load_tools_config_from_path,
)
from config.tools_config import (
    save_tools_config as save_tools_config_to_path,
)
from core.tracer import Tracer
from server.http_api import create_app
from server.ui_hub import UIHub
from server.ui_session_storage import InMemoryUISessionStorage, SQLiteUISessionStorage
from tools.tool_logger import ToolCallLogger


class DummyAgent:
    def __init__(self) -> None:
        self._session_id: str | None = None
        self._approved_categories: set[str] = set()

    def set_session_context(self, session_id: str | None, approved_categories: set[str]) -> None:
        self._session_id = session_id
        self._approved_categories = set(approved_categories)

    def respond(self, messages) -> str:
        return "ok"

    def reconfigure_models(self, main_config, main_api_key=None, *, persist=True) -> None:
        del main_config, main_api_key, persist


class LongCodeAgent(DummyAgent):
    def respond(self, messages) -> str:
        del messages
        lines = [f"def step_{idx}() -> int:\n    return {idx}" for idx in range(1, 26)]
        return "```python\n" + "\n\n".join(lines) + "\n```"


class ShortCodeAgent(DummyAgent):
    def respond(self, messages) -> str:
        del messages
        lines = [f"const item{idx} = {idx};" for idx in range(1, 16)]
        return "```ts\n" + "\n".join(lines) + "\n```"


class CaptureConfigAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.last_provider: str | None = None
        self.last_model: str | None = None
        self.last_api_key: str | None = None

    def reconfigure_models(self, main_config, main_api_key=None, *, persist=True) -> None:
        del persist
        self.last_provider = getattr(main_config, "provider", None)
        self.last_model = getattr(main_config, "model", None)
        self.last_api_key = main_api_key


class ProjectCommandAgent(DummyAgent):
    def respond(self, messages) -> str:
        if not messages:
            return "empty"
        last = messages[-1]
        content = getattr(last, "content", "")
        if isinstance(content, str) and content.startswith("/project "):
            return f"Командный режим (без MWV)\n{content}"
        return "ok"


class UIReportAgent(DummyAgent):
    def respond(self, messages) -> str:
        del messages
        return (
            'ok\nMWV_REPORT_JSON={"route":"chat","trace_id":null,"attempts":{"current":1,"max":1}}'
        )


class ToolCallCaptureAgent(DummyAgent):
    def __init__(self, trace_log: Path, tool_log: Path) -> None:
        super().__init__()
        self._tracer = Tracer(path=trace_log)
        self._tool_logger = ToolCallLogger(path=tool_log)
        self._counter = 0
        self.last_chat_interaction_id: str | None = None

    def respond(self, messages) -> str:
        del messages
        self._counter += 1
        interaction_id = f"interaction-{self._counter}"
        self._tracer.log("user_input", "tool call capture")
        self._tool_logger.log(
            "workspace_write",
            ok=True,
            args={"path": "src/generated/demo.py", "content": "print('ok')"},
        )
        self._tracer.log(
            "interaction_logged",
            "tool call capture done",
            {"interaction_id": interaction_id},
        )
        self.last_chat_interaction_id = interaction_id
        return "print('ok')"


class StaleTraceIdAgent(DummyAgent):
    def __init__(self, trace_id: str) -> None:
        super().__init__()
        self.last_chat_interaction_id = trace_id

    def respond(self, messages) -> str:
        del messages
        return "plain short answer"


class DecisionEchoAgent:
    def __init__(self) -> None:
        self._session_id: str | None = None

    def set_session_context(self, session_id: str | None, approved_categories: set[str]) -> None:
        del approved_categories
        self._session_id = session_id

    def respond(self, messages) -> str:
        del messages
        session_id = self._session_id or "missing-session"
        return json.dumps(
            {
                "id": f"decision-{session_id}",
                "created_at": "2026-01-01T00:00:00+00:00",
                "reason": "need_user_input",
                "summary": f"Decision for {session_id}",
                "context": {"session_id": session_id},
                "options": [
                    {
                        "id": "ask_user",
                        "title": "Ask user",
                        "action": "ask_user",
                        "payload": {},
                        "risk": "low",
                    },
                    {
                        "id": "proceed_safe",
                        "title": "Proceed safely",
                        "action": "proceed_safe",
                        "payload": {},
                        "risk": "low",
                    },
                    {
                        "id": "abort",
                        "title": "Abort",
                        "action": "abort",
                        "payload": {},
                        "risk": "low",
                    },
                ],
                "default_option_id": "ask_user",
                "ttl_seconds": 600,
                "policy": {"require_user_choice": True},
            },
        )

    def reconfigure_models(self, main_config, main_api_key=None, *, persist=True) -> None:
        del main_config, main_api_key, persist


class DecisionOnlyForSessionAAgent:
    def __init__(self) -> None:
        self._session_id: str | None = None

    def set_session_context(self, session_id: str | None, approved_categories: set[str]) -> None:
        del approved_categories
        self._session_id = session_id

    def respond(self, messages) -> str:
        del messages
        session_id = self._session_id or "missing-session"
        if session_id != "session-a":
            return "plain text response"
        return json.dumps(
            {
                "id": "decision-session-a",
                "created_at": "2026-01-01T00:00:00+00:00",
                "reason": "need_user_input",
                "summary": "Decision for session-a",
                "context": {"session_id": "session-a"},
                "options": [
                    {
                        "id": "ask_user",
                        "title": "Ask user",
                        "action": "ask_user",
                        "payload": {},
                        "risk": "low",
                    },
                    {
                        "id": "proceed_safe",
                        "title": "Proceed safely",
                        "action": "proceed_safe",
                        "payload": {},
                        "risk": "low",
                    },
                    {
                        "id": "abort",
                        "title": "Abort",
                        "action": "abort",
                        "payload": {},
                        "risk": "low",
                    },
                ],
                "default_option_id": "ask_user",
                "ttl_seconds": 600,
                "policy": {"require_user_choice": True},
            },
        )

    def reconfigure_models(self, main_config, main_api_key=None, *, persist=True) -> None:
        del main_config, main_api_key, persist


class DelayedFirstUserMessageHub(UIHub):
    def __init__(self, delayed_session_id: str) -> None:
        super().__init__()
        self._delayed_session_id = delayed_session_id
        self._delay_done = False

    async def append_message(self, session_id: str, role: str, content: str) -> dict[str, str]:
        if not self._delay_done and session_id == self._delayed_session_id and role == "user":
            self._delay_done = True
            await asyncio.sleep(0.05)
        return await super().append_message(session_id, role, content)


async def _create_client(agent: DummyAgent) -> TestClient:
    app = create_app(
        agent=agent,
        max_request_bytes=1_000_000,
        ui_storage=InMemoryUISessionStorage(),
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


async def _read_first_sse_event(response) -> dict[str, object]:
    while True:
        line = await response.content.readline()
        if not line:
            raise AssertionError("SSE stream closed before first event")
        if line.startswith(b"data: "):
            raw = line.removeprefix(b"data: ").decode("utf-8").strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise AssertionError("SSE event payload is not an object")
            return parsed


async def _read_sse_event_types(response, *, max_events: int = 20) -> list[str]:
    types: list[str] = []
    while len(types) < max_events:
        try:
            line = await asyncio.wait_for(response.content.readline(), timeout=2)
        except TimeoutError:
            break
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        raw = line.removeprefix(b"data: ").decode("utf-8").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            continue
        event_type = parsed.get("type")
        if isinstance(event_type, str):
            types.append(event_type)
    return types


async def _select_local_model(client: TestClient, session_id: str) -> str:
    models_response = await client.get("/ui/api/models?provider=local")
    assert models_response.status == 200
    models_payload = await models_response.json()
    providers_raw = models_payload.get("providers")
    assert isinstance(providers_raw, list)
    assert providers_raw
    first_provider = providers_raw[0]
    assert isinstance(first_provider, dict)
    models_raw = first_provider.get("models")
    assert isinstance(models_raw, list)
    model_id = next((item for item in models_raw if isinstance(item, str) and item.strip()), None)
    assert isinstance(model_id, str)

    response = await client.post(
        "/ui/api/session-model",
        headers={"X-Slavik-Session": session_id},
        json={"provider": "local", "model": model_id},
    )
    assert response.status == 200
    return model_id


def test_ui_status_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            resp = await client.get("/ui/api/status")
            assert resp.status == 200
            payload = await resp.json()
            assert payload.get("ok") is True
            session_id = payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            assert resp.headers.get("X-Slavik-Session") == session_id
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_requires_model_selection() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post("/ui/api/chat/send", json={"content": "Ping"})
            assert response.status == 409
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "model_not_selected"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            selected_model = await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Ping"},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            response_session_id = payload.get("session_id")
            messages = payload.get("messages", [])
            assert isinstance(response_session_id, str)
            assert response_session_id
            assert isinstance(messages, list)
            assert messages
            assert resp.headers.get("X-Slavik-Session") == response_session_id
            selected = payload.get("selected_model")
            assert isinstance(selected, dict)
            assert selected.get("provider") == "local"
            assert selected.get("model") == selected_model
            display = payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "chat"
            assert display.get("forced") is False
            artifacts = payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts == []
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_strips_mwv_report_block() -> None:
    async def run() -> None:
        client = await _create_client(UIReportAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Ping"},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            report = payload.get("mwv_report")
            assert isinstance(report, dict)
            assert report.get("route") == "chat"
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            content = last.get("content")
            assert isinstance(content, str)
            assert content == "ok"
            output_payload = payload.get("output")
            assert isinstance(output_payload, dict)
            output_content = output_payload.get("content")
            assert isinstance(output_content, str)
            assert output_content == "ok"
            assert "MWV_REPORT_JSON=" not in output_content
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_auto_routes_long_output_to_canvas() -> None:
    async def run() -> None:
        client = await _create_client(LongCodeAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Generate long module"},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            last_content = last.get("content")
            assert isinstance(last_content, str)
            assert last_content.startswith("Статус: результат сформирован в Canvas")
            display = payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "canvas"
            assert display.get("forced") is False
            output_payload = payload.get("output")
            assert isinstance(output_payload, dict)
            output_content = output_payload.get("content")
            assert isinstance(output_content, str)
            assert output_content.startswith("```python")
            artifacts = payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_short_code_stays_in_chat_and_registers_artifact() -> None:
    async def run() -> None:
        client = await _create_client(ShortCodeAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "React button sample"},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            display = payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "chat"
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            last_content = last.get("content")
            assert isinstance(last_content, str)
            assert last_content.startswith("```ts")
            artifacts = payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_force_canvas_override() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Short response", "force_canvas": True},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            last_content = last.get("content")
            assert isinstance(last_content, str)
            assert last_content.startswith("Статус: результат сформирован в Canvas")
            display = payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "canvas"
            assert display.get("forced") is True
            output_payload = payload.get("output")
            assert isinstance(output_payload, dict)
            assert output_payload.get("content") == "ok"
            artifacts = payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_models_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.http_api._fetch_provider_models",
        lambda provider: ([f"{provider}-1", f"{provider}-2"], None),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/models")
            assert response.status == 200
            payload = await response.json()
            providers = payload.get("providers")
            assert isinstance(providers, list)
            names = {item.get("provider") for item in providers if isinstance(item, dict)}
            assert names == {"local", "openrouter", "xai"}
            for item in providers:
                assert isinstance(item, dict)
                models = item.get("models")
                assert isinstance(models, list)
                assert len(models) == 2
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_project_tool_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(ProjectCommandAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            response = await client.post(
                "/ui/api/tools/project",
                headers={"X-Slavik-Session": session_id},
                json={"command": "find", "args": "README"},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload.get("session_id") == session_id
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert len(messages) >= 2
            last_message = messages[-1]
            assert isinstance(last_message, dict)
            content = last_message.get("content")
            assert isinstance(content, str)
            assert "Командный режим (без MWV)" in content
            assert "/project find README" in content
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_project_tool_endpoint_rejects_unknown_command() -> None:
    async def run() -> None:
        client = await _create_client(ProjectCommandAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            response = await client.post(
                "/ui/api/tools/project",
                headers={"X-Slavik-Session": session_id},
                json={"command": "drop_db", "args": ""},
            )
            assert response.status == 400
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "invalid_request_error"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_project_github_import_requires_approval() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            response = await client.post(
                "/ui/api/tools/project",
                headers={"X-Slavik-Session": session_id},
                json={
                    "command": "github_import",
                    "args": "https://github.com/example/repo",
                },
            )
            assert response.status == 200
            payload = await response.json()
            approval_request = payload.get("approval_request")
            assert isinstance(approval_request, dict)
            required_categories = approval_request.get("required_categories")
            assert isinstance(required_categories, list)
            assert "NETWORK_RISK" in required_categories
            assert "EXEC_ARBITRARY" in required_categories
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_project_github_import_runs_after_approval(monkeypatch, tmp_path) -> None:
    async def fake_clone(**kwargs):
        del kwargs
        return True, "ok"

    monkeypatch.setattr("server.http_api._clone_github_repository", fake_clone)
    monkeypatch.setattr(
        "server.http_api._resolve_github_target",
        lambda repo_url: (tmp_path / "repo", "github/example/repo"),
    )
    monkeypatch.setattr(
        "server.http_api._index_imported_project",
        lambda relative_path: (True, f"Code=1, Docs=1 ({relative_path})"),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            approve_resp = await client.post(
                "/slavik/approve-session",
                json={
                    "session_id": session_id,
                    "categories": ["NETWORK_RISK", "EXEC_ARBITRARY"],
                },
            )
            assert approve_resp.status == 200

            response = await client.post(
                "/ui/api/tools/project",
                headers={"X-Slavik-Session": session_id},
                json={
                    "command": "github_import",
                    "args": "https://github.com/example/repo",
                },
            )
            assert response.status == 200
            payload = await response.json()
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last_message = messages[-1]
            assert isinstance(last_message, dict)
            content = last_message.get("content")
            assert isinstance(content, str)
            assert "GitHub import completed" in content
            assert "Code=1, Docs=1" in content
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_session_model_not_found_suggests_closest(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.http_api._fetch_provider_models",
        lambda provider: (["grok-4", "grok-3-mini"], None)
        if provider == "xai"
        else (["local-default"], None),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/session-model",
                headers={"X-Slavik-Session": "session-1"},
                json={"provider": "xai", "model": "grok-4x"},
            )
            assert response.status == 404
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "model_not_found"
            message = error.get("message")
            assert isinstance(message, str)
            assert "сам придумал, сам и страдай" in message
            details = error.get("details")
            assert isinstance(details, dict)
            assert details.get("suggestion") == "grok-4"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_decision_isolated_between_sessions() -> None:
    async def run() -> None:
        app = create_app(
            agent=DecisionEchoAgent(),
            max_request_bytes=1_000_000,
            ui_storage=InMemoryUISessionStorage(),
        )
        app["ui_hub"] = DelayedFirstUserMessageHub("session-a")
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            await _select_local_model(client, "session-a")
            await _select_local_model(client, "session-b")

            async def send(
                session_id: str,
                content: str,
            ) -> tuple[int, dict[str, object], str | None]:
                response = await client.post(
                    "/ui/api/chat/send",
                    json={"content": content},
                    headers={"X-Slavik-Session": session_id},
                )
                payload = await response.json()
                return response.status, payload, response.headers.get("X-Slavik-Session")

            result_a, result_b = await asyncio.gather(
                send("session-a", "Message A"),
                send("session-b", "Message B"),
            )

            for expected_session, result in (
                ("session-a", result_a),
                ("session-b", result_b),
            ):
                status, payload, header_session = result
                assert status == 200
                assert payload.get("session_id") == expected_session
                assert header_session == expected_session
                decision = payload.get("decision")
                assert isinstance(decision, dict)
                assert decision.get("id") == f"decision-{expected_session}"
                context = decision.get("context")
                assert isinstance(context, dict)
                assert context.get("session_id") == expected_session

            decision_a = result_a[1]["decision"]
            decision_b = result_b[1]["decision"]
            assert isinstance(decision_a, dict)
            assert isinstance(decision_b, dict)
            assert decision_a.get("id") != decision_b.get("id")
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_no_decision_leak_from_other_session() -> None:
    async def run() -> None:
        app = create_app(
            agent=DecisionOnlyForSessionAAgent(),
            max_request_bytes=1_000_000,
            ui_storage=InMemoryUISessionStorage(),
        )
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            await _select_local_model(client, "session-a")
            await _select_local_model(client, "session-b")
            response_a = await client.post(
                "/ui/api/chat/send",
                json={"content": "Message A"},
                headers={"X-Slavik-Session": "session-a"},
            )
            assert response_a.status == 200
            payload_a = await response_a.json()
            decision_a = payload_a.get("decision")
            assert isinstance(decision_a, dict)
            assert decision_a.get("id") == "decision-session-a"

            response_b = await client.post(
                "/ui/api/chat/send",
                json={"content": "Message B"},
                headers={"X-Slavik-Session": "session-b"},
            )
            assert response_b.status == 200
            payload_b = await response_b.json()
            assert payload_b.get("session_id") == "session-b"
            assert payload_b.get("decision") is None
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_first_event_is_status() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            assert status_resp.status == 200
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            first_event = await _read_first_sse_event(stream_resp)
            assert first_event.get("type") == "status"
            payload = first_event.get("payload")
            assert isinstance(payload, dict)
            assert payload.get("session_id") == session_id
            assert payload.get("ok") is True
            assert payload.get("state") == "ok"
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_includes_agent_activity() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Ping with activity"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200

            event_types = await _read_sse_event_types(stream_resp, max_events=16)
            assert "agent.activity" in event_types
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_includes_canvas_stream_events() -> None:
    async def run() -> None:
        client = await _create_client(LongCodeAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Stream this long output"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200

            event_types = await _read_sse_event_types(stream_resp, max_events=64)
            assert "canvas.stream.start" in event_types
            assert "canvas.stream.delta" in event_types
            assert "canvas.stream.done" in event_types
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_sessions_api_create_send_get_history() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            created = create_payload.get("session")
            assert isinstance(created, dict)
            session_id = created.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Ping"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200

            get_resp = await client.get(f"/ui/api/sessions/{session_id}")
            assert get_resp.status == 200
            get_payload = await get_resp.json()
            session = get_payload.get("session")
            assert isinstance(session, dict)
            messages = session.get("messages")
            assert isinstance(messages, list)
            assert len(messages) >= 2
            first = messages[0]
            second = messages[1]
            assert isinstance(first, dict)
            assert isinstance(second, dict)
            assert first.get("role") == "user"
            assert second.get("role") == "assistant"
            assert second.get("content") == "ok"

            history_resp = await client.get(f"/ui/api/sessions/{session_id}/history")
            assert history_resp.status == 200
            history_payload = await history_resp.json()
            history_messages = history_payload.get("messages")
            assert isinstance(history_messages, list)
            assert len(history_messages) == len(messages)

            output_resp = await client.get(f"/ui/api/sessions/{session_id}/output")
            assert output_resp.status == 200
            output_payload = await output_resp.json()
            output = output_payload.get("output")
            assert isinstance(output, dict)
            assert output.get("content") == "ok"
            session_artifacts = session.get("artifacts")
            assert isinstance(session_artifacts, list)
            assert session_artifacts == []

            files_resp = await client.get(f"/ui/api/sessions/{session_id}/files")
            assert files_resp.status == 200
            files_payload = await files_resp.json()
            files = files_payload.get("files")
            assert isinstance(files, list)
            assert files == []

            list_resp = await client.get("/ui/api/sessions")
            assert list_resp.status == 200
            list_payload = await list_resp.json()
            sessions = list_payload.get("sessions")
            assert isinstance(sessions, list)
            selected = next(
                (item for item in sessions if item.get("session_id") == session_id),
                None,
            )
            assert isinstance(selected, dict)
            assert selected.get("message_count") == len(messages)
            assert selected.get("title") == "Ping"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_stores_files_from_tool_calls(monkeypatch, tmp_path) -> None:
    trace_log = tmp_path / "trace.log"
    tool_log = tmp_path / "tool_calls.log"
    monkeypatch.setattr("server.http_api.TRACE_LOG", trace_log)
    monkeypatch.setattr("server.http_api.TOOL_CALLS_LOG", tool_log)

    async def run() -> None:
        client = await _create_client(ToolCallCaptureAgent(trace_log, tool_log))
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session = create_payload.get("session")
            assert isinstance(session, dict)
            session_id = session.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Generate file"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            output = send_payload.get("output")
            assert isinstance(output, dict)
            assert output.get("content") == "print('ok')"
            display = send_payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "chat"
            files = send_payload.get("files")
            assert isinstance(files, list)
            assert "src/generated/demo.py" in files
            artifacts = send_payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts

            files_resp = await client.get(f"/ui/api/sessions/{session_id}/files")
            assert files_resp.status == 200
            files_payload = await files_resp.json()
            files_from_endpoint = files_payload.get("files")
            assert isinstance(files_from_endpoint, list)
            assert "src/generated/demo.py" in files_from_endpoint
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_ignores_stale_trace_id(monkeypatch, tmp_path) -> None:
    trace_log = tmp_path / "trace.log"
    tool_log = tmp_path / "tool_calls.log"
    stale_trace_id = "stale-interaction"
    tracer = Tracer(path=trace_log)
    tool_logger = ToolCallLogger(path=tool_log)
    tracer.log("user_input", "old input")
    tool_logger.log(
        "workspace_write",
        ok=True,
        args={"path": "src/generated/old.py", "content": "print('old')"},
    )
    tracer.log(
        "interaction_logged",
        "old interaction done",
        {"interaction_id": stale_trace_id},
    )
    monkeypatch.setattr("server.http_api.TRACE_LOG", trace_log)
    monkeypatch.setattr("server.http_api.TOOL_CALLS_LOG", tool_log)

    async def run() -> None:
        client = await _create_client(StaleTraceIdAgent(stale_trace_id))
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session = create_payload.get("session")
            assert isinstance(session, dict)
            session_id = session.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Simple reply please"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            display = send_payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "chat"
            files = send_payload.get("files")
            assert isinstance(files, list)
            assert files == []
            artifacts = send_payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts == []
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_sessions_api_delete_chat() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            created = create_payload.get("session")
            assert isinstance(created, dict)
            session_id = created.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Delete me"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200

            delete_resp = await client.delete(f"/ui/api/sessions/{session_id}")
            assert delete_resp.status == 200
            delete_payload = await delete_resp.json()
            assert delete_payload.get("session_id") == session_id
            assert delete_payload.get("deleted") is True

            get_resp = await client.get(f"/ui/api/sessions/{session_id}")
            assert get_resp.status == 404
            get_payload = await get_resp.json()
            error = get_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "session_not_found"

            list_resp = await client.get("/ui/api/sessions")
            assert list_resp.status == 200
            list_payload = await list_resp.json()
            sessions = list_payload.get("sessions")
            assert isinstance(sessions, list)
            assert all(
                item.get("session_id") != session_id for item in sessions if isinstance(item, dict)
            )

            delete_missing_resp = await client.delete(f"/ui/api/sessions/{session_id}")
            assert delete_missing_resp.status == 404
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_sessions_persist_after_restart(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "ui_sessions.db"

        storage_before = SQLiteUISessionStorage(db_path)
        app_before = create_app(
            agent=DecisionEchoAgent(),
            max_request_bytes=1_000_000,
            ui_storage=storage_before,
        )
        server_before = TestServer(app_before)
        client_before = TestClient(server_before)
        await client_before.start_server()

        session_id: str | None = None
        try:
            create_resp = await client_before.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            created = create_payload.get("session")
            assert isinstance(created, dict)
            raw_session_id = created.get("session_id")
            assert isinstance(raw_session_id, str)
            assert raw_session_id
            session_id = raw_session_id
            selected_model = await _select_local_model(client_before, session_id)

            send_resp = await client_before.post(
                "/ui/api/chat/send",
                json={"content": "Persist me"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            decision = send_payload.get("decision")
            assert isinstance(decision, dict)
            assert decision.get("id") == f"decision-{session_id}"
        finally:
            await client_before.close()

        assert isinstance(session_id, str)
        storage_after = SQLiteUISessionStorage(db_path)
        app_after = create_app(
            agent=DummyAgent(),
            max_request_bytes=1_000_000,
            ui_storage=storage_after,
        )
        server_after = TestServer(app_after)
        client_after = TestClient(server_after)
        await client_after.start_server()
        try:
            get_resp = await client_after.get(f"/ui/api/sessions/{session_id}")
            assert get_resp.status == 200
            get_payload = await get_resp.json()
            session = get_payload.get("session")
            assert isinstance(session, dict)
            assert session.get("session_id") == session_id
            messages = session.get("messages")
            assert isinstance(messages, list)
            assert len(messages) >= 2
            first = messages[0]
            second = messages[1]
            assert isinstance(first, dict)
            assert isinstance(second, dict)
            assert first.get("role") == "user"
            assert second.get("role") == "assistant"
            second_content = second.get("content")
            assert isinstance(second_content, str)
            assert f"decision-{session_id}" in second_content
            restored_decision = session.get("decision")
            assert isinstance(restored_decision, dict)
            assert restored_decision.get("id") == f"decision-{session_id}"
            restored_output = session.get("output")
            assert isinstance(restored_output, dict)
            output_content = restored_output.get("content")
            assert isinstance(output_content, str)
            assert f"decision-{session_id}" in output_content
            restored_files = session.get("files")
            assert isinstance(restored_files, list)
            assert restored_files == []
            restored_artifacts = session.get("artifacts")
            assert isinstance(restored_artifacts, list)
            assert restored_artifacts == []
            restored_model = session.get("selected_model")
            assert isinstance(restored_model, dict)
            assert restored_model.get("provider") == "local"
            assert restored_model.get("model") == selected_model

            status_resp = await client_after.get(
                "/ui/api/status",
                headers={"X-Slavik-Session": session_id},
            )
            assert status_resp.status == 200
            status_payload = await status_resp.json()
            decision_from_status = status_payload.get("decision")
            assert isinstance(decision_from_status, dict)
            assert decision_from_status.get("id") == f"decision-{session_id}"
            selected_from_status = status_payload.get("selected_model")
            assert isinstance(selected_from_status, dict)
            assert selected_from_status.get("provider") == "local"
            assert selected_from_status.get("model") == selected_model
        finally:
            await client_after.close()

    asyncio.run(run())


def test_ui_settings_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/settings")
            assert response.status == 200
            payload = await response.json()
            settings = payload.get("settings")
            assert isinstance(settings, dict)
            personalization = settings.get("personalization")
            assert isinstance(personalization, dict)
            assert isinstance(personalization.get("tone"), str)
            assert isinstance(personalization.get("system_prompt"), str)
            memory = settings.get("memory")
            assert isinstance(memory, dict)
            assert isinstance(memory.get("auto_save_dialogue"), bool)
            assert isinstance(memory.get("inbox_max_items"), int)
            assert isinstance(memory.get("embeddings_model"), str)
            tools = settings.get("tools")
            assert isinstance(tools, dict)
            state = tools.get("state")
            assert isinstance(state, dict)
            assert "safe_mode" in state
            providers = settings.get("providers")
            assert isinstance(providers, list)
            provider_names = {item.get("provider") for item in providers if isinstance(item, dict)}
            assert provider_names == {"local", "openrouter", "xai", "openai"}
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_settings_update_endpoint(monkeypatch, tmp_path) -> None:
    memory_path = tmp_path / "memory.json"
    tools_path = tmp_path / "tools.json"
    ui_settings_path = tmp_path / "ui_settings.json"

    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)
    monkeypatch.setattr(
        "server.http_api.load_memory_config",
        lambda: load_memory_config_from_path(path=memory_path),
    )
    monkeypatch.setattr(
        "server.http_api.save_memory_config",
        lambda config: save_memory_config_to_path(config, path=memory_path),
    )
    monkeypatch.setattr(
        "server.http_api.load_tools_config",
        lambda: load_tools_config_from_path(path=tools_path),
    )
    monkeypatch.setattr(
        "server.http_api.save_tools_config",
        lambda config: save_tools_config_to_path(config, path=tools_path),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/settings",
                json={
                    "personalization": {
                        "tone": "direct",
                        "system_prompt": "Focus on concise implementation details.",
                    },
                    "memory": {
                        "auto_save_dialogue": True,
                        "inbox_max_items": 77,
                        "inbox_ttl_days": 14,
                        "inbox_writes_per_minute": 9,
                        "embeddings_model": "test-embeddings-v1",
                    },
                    "tools": {
                        "state": {
                            "safe_mode": False,
                            "web": True,
                        },
                    },
                    "providers": {
                        "xai": {"api_key": "xai-test-key"},
                        "openrouter": {"api_key": "or-test-key"},
                        "openai": {"api_key": "openai-test-key"},
                    },
                },
            )
            assert response.status == 200
            payload = await response.json()
            settings = payload.get("settings")
            assert isinstance(settings, dict)
            personalization = settings.get("personalization")
            assert isinstance(personalization, dict)
            assert personalization.get("tone") == "direct"
            assert (
                personalization.get("system_prompt") == "Focus on concise implementation details."
            )
            memory = settings.get("memory")
            assert isinstance(memory, dict)
            assert memory.get("auto_save_dialogue") is True
            assert memory.get("inbox_max_items") == 77
            assert memory.get("inbox_ttl_days") == 14
            assert memory.get("inbox_writes_per_minute") == 9
            assert memory.get("embeddings_model") == "test-embeddings-v1"
            tools = settings.get("tools")
            assert isinstance(tools, dict)
            state = tools.get("state")
            assert isinstance(state, dict)
            assert state.get("safe_mode") is False
            assert state.get("web") is True
            providers = settings.get("providers")
            assert isinstance(providers, list)
            provider_by_name = {
                item.get("provider"): item for item in providers if isinstance(item, dict)
            }
            xai_provider = provider_by_name.get("xai")
            assert isinstance(xai_provider, dict)
            assert xai_provider.get("api_key_set") is True
            assert xai_provider.get("api_key_source") == "settings"
            assert xai_provider.get("api_key_value") == "xai-test-key"
            openrouter_provider = provider_by_name.get("openrouter")
            assert isinstance(openrouter_provider, dict)
            assert openrouter_provider.get("api_key_set") is True
            assert openrouter_provider.get("api_key_source") == "settings"
            assert openrouter_provider.get("api_key_value") == "or-test-key"

            saved_payload = json.loads(ui_settings_path.read_text(encoding="utf-8"))
            assert isinstance(saved_payload, dict)
            providers_blob = saved_payload.get("providers")
            assert isinstance(providers_blob, dict)
            xai_saved = providers_blob.get("xai")
            assert isinstance(xai_saved, dict)
            assert xai_saved.get("api_key") == "xai-test-key"
            openrouter_saved = providers_blob.get("openrouter")
            assert isinstance(openrouter_saved, dict)
            assert openrouter_saved.get("api_key") == "or-test-key"
            openai_provider = provider_by_name.get("openai")
            assert isinstance(openai_provider, dict)
            assert openai_provider.get("api_key_set") is True
            assert openai_provider.get("api_key_source") == "settings"
            assert openai_provider.get("api_key_value") == "openai-test-key"
            openai_saved = providers_blob.get("openai")
            assert isinstance(openai_saved, dict)
            assert openai_saved.get("api_key") == "openai-test-key"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_uses_api_key_from_settings(monkeypatch, tmp_path) -> None:
    ui_settings_path = tmp_path / "ui_settings.json"
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)
    monkeypatch.setattr(
        "server.http_api._fetch_provider_models",
        lambda provider: (["grok-4x"], None)
        if provider == "openrouter"
        else (["local-default"], None),
    )

    async def run() -> None:
        agent = CaptureConfigAgent()
        client = await _create_client(agent)
        try:
            settings_resp = await client.post(
                "/ui/api/settings",
                json={"providers": {"openrouter": {"api_key": "or-ui-test-key"}}},
            )
            assert settings_resp.status == 200

            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_payload = create_payload.get("session")
            assert isinstance(session_payload, dict)
            session_id = session_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id

            select_resp = await client.post(
                "/ui/api/session-model",
                headers={"X-Slavik-Session": session_id},
                json={"provider": "openrouter", "model": "grok-4x"},
            )
            assert select_resp.status == 200

            send_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "ping"},
            )
            assert send_resp.status == 200
            assert agent.last_provider == "openrouter"
            assert agent.last_model == "grok-4x"
            assert agent.last_api_key == "or-ui-test-key"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chats_export_import_endpoints() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_payload = create_payload.get("session")
            assert isinstance(session_payload, dict)
            session_id = session_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)
            send_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "export me"},
            )
            assert send_resp.status == 200

            export_resp = await client.get("/ui/api/settings/chats/export")
            assert export_resp.status == 200
            export_payload = await export_resp.json()
            exported_sessions = export_payload.get("sessions")
            assert isinstance(exported_sessions, list)
            assert len(exported_sessions) >= 1

            import_resp = await client.post(
                "/ui/api/settings/chats/import",
                json={
                    "mode": "replace",
                    "sessions": [
                        {
                            "session_id": "imported-session",
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                            "status": "ok",
                            "messages": [
                                {"role": "user", "content": "hello import"},
                                {"role": "assistant", "content": "import ok"},
                            ],
                            "decision": None,
                            "selected_model": {"provider": "local", "model": "local-default"},
                        },
                    ],
                },
            )
            assert import_resp.status == 200
            import_payload = await import_resp.json()
            assert import_payload.get("imported") == 1
            assert import_payload.get("mode") == "replace"

            imported_resp = await client.get("/ui/api/sessions/imported-session")
            assert imported_resp.status == 200
            imported_payload = await imported_resp.json()
            imported_session = imported_payload.get("session")
            assert isinstance(imported_session, dict)
            messages = imported_session.get("messages")
            assert isinstance(messages, list)
            assert len(messages) == 2
            first = messages[0]
            second = messages[1]
            assert isinstance(first, dict)
            assert isinstance(second, dict)
            assert first.get("content") == "hello import"
            assert second.get("content") == "import ok"
        finally:
            await client.close()

    asyncio.run(run())
