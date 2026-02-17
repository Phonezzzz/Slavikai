from __future__ import annotations

# ruff: noqa: F401
import asyncio
import io
import json
import zipfile
from pathlib import Path

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from config.http_server_config import HttpAuthConfig
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
from core.approval_policy import ApprovalPrompt, ApprovalRequest, ApprovalRequired
from core.tracer import Tracer
from server.http_api import _resolve_workspace_root_candidate, create_app
from server.ui_hub import UIHub
from server.ui_session_storage import (
    InMemoryUISessionStorage,
    PersistedSession,
    SQLiteUISessionStorage,
)
from shared.models import JSONValue, ToolResult
from tools.tool_logger import ToolCallLogger

TEST_API_TOKEN = "test-ui-api-token"
TEST_AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_TOKEN}"}


class DummyAgent:
    def __init__(self) -> None:
        self._session_id: str | None = None
        self._approved_categories: set[str] = set()
        self.tools_enabled: dict[str, bool] = {}

    def set_session_context(self, session_id: str | None, approved_categories: set[str]) -> None:
        self._session_id = session_id
        self._approved_categories = set(approved_categories)

    def respond(self, messages) -> str:
        return "ok"

    def reconfigure_models(self, main_config, main_api_key=None, *, persist=True) -> None:
        del main_config, main_api_key, persist

    def update_tools_enabled(self, state: dict[str, bool]) -> None:
        self.tools_enabled.update(state)

    def call_tool(
        self,
        name: str,
        args: dict[str, JSONValue] | None = None,
        raw_input: str | None = None,
    ) -> ToolResult:
        del args, raw_input
        return ToolResult.failure(f"Инструмент {name} не поддерживается в тестовом агенте")


class WorkspaceDecisionAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.tool_calls: list[tuple[str, dict[str, JSONValue]]] = []

    def call_tool(
        self,
        name: str,
        args: dict[str, JSONValue] | None = None,
        raw_input: str | None = None,
    ) -> ToolResult:
        del raw_input
        if name == "workspace_list":
            return ToolResult.success({"tree": []})
        if name == "workspace_read":
            path_raw = args.get("path") if isinstance(args, dict) else ""
            path = path_raw if isinstance(path_raw, str) else "main.py"
            return ToolResult.success({"output": f"# file: {path}\n"})
        if name in {"workspace_write", "workspace_run"}:
            if "EXEC_ARBITRARY" not in self._approved_categories:
                request = ApprovalRequest(
                    category="EXEC_ARBITRARY",
                    required_categories=["EXEC_ARBITRARY"],
                    prompt=ApprovalPrompt(
                        what=f"Разрешить {name} в workspace.",
                        why="Нужно выполнить потенциально рискованное действие.",
                        risk="Выполнение/запись может изменить проект.",
                        changes=["workspace files"],
                    ),
                    tool=name,
                    details={"args": dict(args or {})},
                    session_id=self._session_id,
                )
                raise ApprovalRequired(request)
            call_args = dict(args or {})
            self.tool_calls.append((name, call_args))
            if name == "workspace_write":
                path_raw = call_args.get("path")
                path = path_raw if isinstance(path_raw, str) else "main.py"
                return ToolResult.success({"output": "saved", "path": path})
            return ToolResult.success({"output": "ran", "stderr": "", "exit_code": 0})
        return ToolResult.failure(f"unsupported tool {name}")


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


class TracedStreamingAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self._counter = 0
        self.last_chat_interaction_id: str | None = None

    def _next_trace_id(self) -> str:
        self._counter += 1
        trace_id = f"stream-trace-{self._counter}"
        self.last_chat_interaction_id = trace_id
        return trace_id

    def respond_stream(self, messages):
        del messages
        self._next_trace_id()
        yield "Hello"
        yield " "
        yield "stream"

    def respond(self, messages) -> str:
        del messages
        self._next_trace_id()
        return "Hello stream"


class NamedFileArtifactsAgent(DummyAgent):
    def respond(self, messages) -> str:
        del messages
        return (
            "Код (`clock.py`):\n"
            "```python\n"
            "import time\n"
            "print(time.strftime('%H:%M:%S'))\n"
            "```\n\n"
            "Скрипт (`clock.sh`):\n"
            "```sh\n"
            "#!/bin/bash\n"
            "date '+%H:%M:%S'\n"
            "```\n"
        )


class StreamNamedFileArtifactsAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.last_stream_response_raw: str | None = None

    def respond_stream(self, messages):
        del messages
        parts = [
            "Вот мини-приложение.\n\n",
            "Код (`clock.py`):\n```python\nimport time\nprint(time.time())\n```\n",
        ]
        yield from parts
        self.last_stream_response_raw = "".join(parts)

    def respond(self, messages) -> str:
        del messages
        return (
            "Вот мини-приложение.\n\n"
            "Код (`clock.py`):\n```python\nimport time\nprint(time.time())\n```\n"
        )


class LateNamedFileStreamAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.last_stream_response_raw: str | None = None

    def respond_stream(self, messages):
        del messages
        intro = "Это подготовка результата для пользователя. " * 8
        tail = "\nКод (`clock.py`):\n```python\nimport time\nprint(time.time())\n```\n"
        yield intro
        yield tail
        self.last_stream_response_raw = f"{intro}{tail}"

    def respond(self, messages) -> str:
        del messages
        intro = "Это подготовка результата для пользователя. " * 8
        tail = "\nКод (`clock.py`):\n```python\nimport time\nprint(time.time())\n```\n"
        return f"{intro}{tail}"


class EscapedFenceNamedFileAgent(DummyAgent):
    def respond(self, messages) -> str:
        del messages
        return "Код (`clock.py`):\\n```python\\nimport time\\nprint(time.time())\\n```\\n"


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


class LiveToolsAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.tools_enabled = {"web": False, "safe_mode": True}
        self.update_calls: list[dict[str, bool]] = []

    def update_tools_enabled(self, state: dict[str, bool]) -> None:
        self.update_calls.append(dict(state))
        super().update_tools_enabled(state)

    def respond(self, messages) -> str:
        if not messages:
            return "ok"
        last = messages[-1]
        content = getattr(last, "content", "")
        if isinstance(content, str) and content.startswith("/web "):
            if not self.tools_enabled.get("web", False):
                return "Инструмент web отключён"
            return "WEB_OK"
        return "ok"


class LiveEmbeddingsAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings_provider = "local"
        self.embeddings_local_model = "all-MiniLM-L6-v2"
        self.embeddings_openai_model = "text-embedding-3-small"
        self.set_calls: list[dict[str, str | None]] = []

    def set_embeddings_model(self, model_name: str) -> None:
        self.embeddings_provider = "local"
        self.embeddings_local_model = model_name
        self.set_calls.append(
            {
                "provider": "local",
                "local_model": model_name,
                "openai_model": self.embeddings_openai_model,
                "openai_api_key": None,
            }
        )

    def set_embeddings_config(
        self,
        *,
        provider: str,
        local_model: str,
        openai_model: str,
        openai_api_key: str | None,
    ) -> None:
        self.embeddings_provider = provider
        self.embeddings_local_model = local_model
        self.embeddings_openai_model = openai_model
        self.set_calls.append(
            {
                "provider": provider,
                "local_model": local_model,
                "openai_model": openai_model,
                "openai_api_key": openai_api_key,
            }
        )


class MemoryConflictAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self._conflicts: dict[str, dict[str, JSONValue]] = {
            "policy:avoid_emoji": {
                "atom_id": "atom-1",
                "stable_key": "policy:avoid_emoji",
                "claim_type": "policy",
                "value_json": {"rule": "avoid emoji"},
                "confidence": 0.7,
                "support_count": 1,
                "contradict_count": 2,
                "last_seen_at": "2026-01-01T00:00:00+00:00",
                "status": "conflict",
                "summary_text": 'policy:avoid_emoji={"rule":"avoid emoji"}',
            }
        }

    def list_memory_conflicts(self, limit: int = 50) -> list[dict[str, JSONValue]]:
        items = list(self._conflicts.values())
        return [dict(item) for item in items[:limit]]

    def resolve_memory_conflict(
        self,
        *,
        stable_key: str,
        action: str,
        value_json: JSONValue | None = None,
    ) -> dict[str, JSONValue] | None:
        current = self._conflicts.get(stable_key)
        if current is None:
            return None
        next_item = dict(current)
        if action == "activate":
            next_item["status"] = "active"
        elif action == "deprecate":
            next_item["status"] = "deprecated"
        elif action == "set_value":
            next_item["status"] = "active"
            next_item["value_json"] = value_json
        else:
            raise ValueError("invalid action")
        self._conflicts[stable_key] = next_item
        return dict(next_item)


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

    async def append_message(
        self,
        session_id: str,
        message: dict[str, JSONValue],
    ) -> dict[str, JSONValue]:
        role_raw = message.get("role")
        role = role_raw if isinstance(role_raw, str) else ""
        if not self._delay_done and session_id == self._delayed_session_id and role == "user":
            self._delay_done = True
            await asyncio.sleep(0.05)
        return await super().append_message(session_id, dict(message))


class FakeSttResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


async def _create_client(agent: DummyAgent) -> TestClient:
    app = create_app(
        agent=agent,
        max_request_bytes=1_000_000,
        ui_storage=InMemoryUISessionStorage(),
        auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
    )
    server = TestServer(app)
    client = TestClient(server, headers=TEST_AUTH_HEADERS)
    await client.start_server()
    return client


async def _set_session_policy_via_api(
    client: TestClient,
    *,
    session_id: str,
    policy_profile: str,
    confirm_yolo: bool | None = None,
) -> tuple[int, dict[str, JSONValue]]:
    policy: dict[str, JSONValue] = {"profile": policy_profile}
    if policy_profile == "yolo" and confirm_yolo is True:
        policy["yolo_confirm"] = True
        policy["yolo_confirm_text"] = "YOLO"
    payload: dict[str, JSONValue] = {"policy": policy}
    response = await client.post(
        "/ui/api/session/security",
        headers={"X-Slavik-Session": session_id},
        json=payload,
    )
    body = await response.json()
    return response.status, body if isinstance(body, dict) else {}


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


async def _read_sse_events(response, *, max_events: int = 20) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    while len(events) < max_events:
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
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


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


async def _enter_act_mode(client: TestClient, session_id: str, *, goal: str = "test run") -> None:
    to_plan = await client.post(
        "/ui/api/mode",
        headers={"X-Slavik-Session": session_id},
        json={"mode": "plan"},
    )
    assert to_plan.status == 200

    draft_resp = await client.post(
        "/ui/api/plan/draft",
        headers={"X-Slavik-Session": session_id},
        json={"goal": goal},
    )
    assert draft_resp.status == 200
    draft_payload = await draft_resp.json()
    draft_plan = draft_payload.get("active_plan")
    assert isinstance(draft_plan, dict)
    plan_revision = draft_plan.get("plan_revision")
    assert isinstance(plan_revision, int)
    assert plan_revision > 0

    approve_resp = await client.post(
        "/ui/api/plan/approve",
        headers={"X-Slavik-Session": session_id},
    )
    assert approve_resp.status == 200
    approve_payload = await approve_resp.json()
    approved_plan = approve_payload.get("active_plan")
    assert isinstance(approved_plan, dict)
    approved_revision = approved_plan.get("plan_revision")
    assert isinstance(approved_revision, int)

    execute_resp = await client.post(
        "/ui/api/plan/execute",
        headers={"X-Slavik-Session": session_id},
        json={"plan_revision": approved_revision},
    )
    assert execute_resp.status == 409
    execute_payload = await execute_resp.json()
    decision = execute_payload.get("decision")
    assert isinstance(decision, dict)
    decision_id = decision.get("id")
    assert isinstance(decision_id, str)
    respond_resp = await client.post(
        "/ui/api/decision/respond",
        headers={"X-Slavik-Session": session_id},
        json={
            "session_id": session_id,
            "decision_id": decision_id,
            "choice": "approve_once",
        },
    )
    assert respond_resp.status == 200
    respond_payload = await respond_resp.json()
    assert respond_payload.get("mode") == "act"


__all__ = [name for name in globals() if not name.startswith("__")]
