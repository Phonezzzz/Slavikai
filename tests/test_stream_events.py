from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.agent import Agent
from core.agent_response import ResponseProduced
from core.tool_loop import AgentToolLoopResult
from llm.brain_base import Brain
from llm.deepseek_brain import DeepSeekBrain
from llm.inception_brain import InceptionBrain
from llm.local_http_brain import LocalHttpBrain
from llm.openrouter_brain import OpenRouterBrain
from llm.stream_model import (
    Done,
    Error,
    StreamEvent,
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    iter_openai_sse_events,
)
from llm.types import LLMResult, ModelConfig, ToolCall, ToolSpec
from llm.xai_brain import XAiBrain
from shared.models import LLMMessage, ToolResult


class _StreamResponse:
    status_code = 200
    encoding = "utf-8"

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = True):  # noqa: ANN201
        del decode_unicode
        yield 'data: {"choices":[{"delta":{"content":"A"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"B"}}]}'
        yield "data: [DONE]"


def test_stream_events_text_delta_ordering() -> None:
    events = list(
        iter_openai_sse_events(
            [
                'data: {"choices":[{"delta":{"content":"A"}}]}',
                (
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                    '"id":"call-1","function":{"name":"workspace_read",'
                    '"arguments":"{\\"path\\":"}}]}}]}'
                ),
                (
                    'data: {"choices":[{"delta":{"content":"B","tool_calls":['
                    '{"index":0,"function":{"arguments":"\\"README.md\\"}"}}]}}]}'
                ),
                "data: [DONE]",
            ]
        )
    )

    assert [type(event) for event in events] == [
        TextDelta,
        ToolCallStarted,
        ToolCallArgumentsDelta,
        TextDelta,
        ToolCallArgumentsDelta,
        ToolCallCompleted,
        Done,
    ]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["A", "B"]
    completed = next(event for event in events if isinstance(event, ToolCallCompleted))
    assert completed.call.arguments == {"path": "README.md"}


@pytest.mark.parametrize(
    "provider",
    ["local", "deepseek", "openrouter", "inception", "xai"],
)
def test_all_providers_emit_stream_events(provider: str, monkeypatch) -> None:
    def fake_post(*args, **kwargs):  # noqa: ANN002,ANN003,ANN202
        del args, kwargs
        return _StreamResponse()

    monkeypatch.setattr("requests.post", fake_post)
    config = ModelConfig(provider=provider, model=f"{provider}-model")
    brain: Brain
    if provider == "local":
        brain = LocalHttpBrain(default_config=config)
    elif provider == "deepseek":
        brain = DeepSeekBrain(api_key="test-key", default_config=config)
    elif provider == "openrouter":
        brain = OpenRouterBrain(api_key="test-key", default_config=config)
    elif provider == "inception":
        brain = InceptionBrain(api_key="test-key", default_config=config)
    else:
        brain = XAiBrain(api_key="test-key", default_config=config)

    events = list(brain.generate_stream_events([LLMMessage(role="user", content="ping")]))

    assert events
    assert all(isinstance(event, StreamEvent) for event in events)
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["A", "B"]
    assert isinstance(events[-1], Done)


class _WebAndFileBrain(Brain):
    supports_native_tools = True
    supports_streaming_tools = True

    def __init__(self) -> None:
        self.calls = 0
        self.messages_seen: list[list[LLMMessage]] = []

    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        del config
        self.calls += 1
        self.messages_seen.append(list(messages))
        tool_names = {tool.name for tool in tools or []}
        assert {"web", "workspace_read"}.issubset(tool_names)
        if self.calls == 1:
            return LLMResult(
                text="",
                tool_calls=[ToolCall(id="web-call", name="web", arguments={"query": "SlavikAI"})],
            )
        if self.calls == 2:
            return LLMResult(
                text="",
                tool_calls=[
                    ToolCall(
                        id="file-call",
                        name="workspace_read",
                        arguments={"path": "README.md"},
                    )
                ],
            )
        return LLMResult(text="Ответ использует веб и файл.")


@pytest.mark.parametrize("termination", ["normal", "error", "cancelled"])
def test_agent_stream_runs_web_and_file_tools_in_one_response(
    tmp_path: Path, monkeypatch, termination: str
) -> None:
    brain = _WebAndFileBrain()
    agent = Agent(
        brain=brain,
        enable_tools={"web": True, "safe_mode": False},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.runtime_mode = "ask"
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.tool_registry.register(
        "web",
        lambda request: ToolResult.success({"output": f"web:{request.args['query']}"}),
        enabled=True,
        capability="read",
        description="Веб-поиск",
        parameters_schema={"type": "object"},
        chat_exposed=True,
    )
    agent.tool_registry.register(
        "workspace_read",
        lambda request: ToolResult.success({"content": f"file:{request.args['path']}"}),
        enabled=True,
        capability="read",
        description="Чтение файла",
        parameters_schema={"type": "object"},
        chat_exposed=True,
    )

    token = asyncio.Event()
    if termination != "normal":
        provider_stream = brain.generate_stream_events

        def _interrupted_provider(*args, **kwargs):
            if brain.calls == 1:
                if termination == "error":
                    yield Error(message="provider interrupted", code="provider_model_error")
                    yield Done(finish_reason="error")
                else:
                    token.set()
                    yield Done(finish_reason="cancelled")
                return
            yield from provider_stream(*args, **kwargs)

        monkeypatch.setattr(brain, "generate_stream_events", _interrupted_provider)
    events = list(
        agent.respond_stream(
            [LLMMessage(role="user", content="Сверь веб и README")], cancellation_token=token
        )
    )

    completed = [event for event in events if isinstance(event, ToolCallCompleted)]
    expected_tools = ["web", "workspace_read"] if termination == "normal" else ["web"]
    assert [event.call.name for event in completed] == expected_tools
    assert all(event.result is not None and event.result.ok for event in completed)
    if termination == "normal":
        assert "".join(event.text for event in events if isinstance(event, TextDelta)) == (
            "Ответ использует веб и файл."
        )
    assert isinstance(events[-1], Done)
    assert sum(isinstance(event, Done) for event in events) == 1
    assert isinstance(events[-2], ResponseProduced)
    if termination == "normal":
        assert "Ответ использует веб и файл." in events[-2].response.text
    result = events[-2].response.runtime_result
    assert isinstance(result, AgentToolLoopResult)
    expected_ids = ["web-call", "file-call"] if termination == "normal" else ["web-call"]
    assert [item.call.id for item in result.tool_calls] == expected_ids
    assert all(
        item.result is event.result
        for item, event in zip(result.tool_calls, completed, strict=True)
    )
    if termination == "normal":
        assert brain.calls == 3
        assert brain.messages_seen[1][-1].role == "tool"
        assert brain.messages_seen[2][-1].role == "tool"
    else:
        assert brain.calls == 1
        assert events[-1].finish_reason == termination
        assert result.cancelled is (termination == "cancelled")
        assert result.error == ("provider interrupted" if termination == "error" else None)
