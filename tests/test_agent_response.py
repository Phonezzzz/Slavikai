from __future__ import annotations

from pathlib import Path

import pytest

from core.agent import Agent
from core.agent_response import ResponseProduced
from core.tool_loop import AgentToolLoopResult
from llm.brain_base import Brain
from llm.stream_model import Done
from llm.types import LLMResult, ModelConfig, ToolCall, ToolSpec, WebSearchEvidence
from shared.models import LLMMessage, ToolRequest, ToolResult


class SimpleBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.messages: list[LLMMessage] = []
        self.evidence: WebSearchEvidence | None = None
        self.last_result: LLMResult | None = None

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        self.messages = list(messages)
        self.last_result = LLMResult(text=self.text, web_search_evidence=self.evidence)
        return self.last_result


class ToolLoopBrain(Brain):
    supports_native_tools = True
    supports_streaming_tools = True

    def __init__(self) -> None:
        self.calls = 0
        self.seen_tools: list[ToolSpec] = []
        self.messages_seen: list[list[LLMMessage]] = []

    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        del config
        self.calls += 1
        self.seen_tools = list(tools or [])
        self.messages_seen.append(list(messages))
        if self.calls == 1:
            return LLMResult(
                text="need lookup",
                tool_calls=[
                    ToolCall(
                        id="lookup-1",
                        name="chat_lookup",
                        arguments={"query": "ping"},
                    )
                ],
            )
        assert messages[-1].role == "tool"
        return LLMResult(text=f"tool loop final: {messages[-1].content}")


class FakeWebTool:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.calls = 0

    def handle(self, request: ToolRequest) -> ToolResult:
        self.calls += 1
        assert request.name == "web"
        assert isinstance(request.args.get("query"), str)
        return self.result


def test_agent_simple_response(tmp_path: Path) -> None:
    brain = SimpleBrain("hello")
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    response = agent.respond([LLMMessage(role="user", content="привет")]).text
    assert "hello" in response
    assert brain.calls >= 1


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("tool_ok", [False, True])
@pytest.mark.parametrize("projection_failure", [None, "review", "log", "trace"])
def test_agent_chat_response_can_use_read_only_native_tool_loop(
    tmp_path: Path, monkeypatch, streaming: bool, tool_ok: bool, projection_failure: str | None
) -> None:
    brain = ToolLoopBrain()
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    observed = (
        ToolResult.success({"output": "lookup:ping"})
        if tool_ok
        else ToolResult.failure("lookup failed", data={"output": "lookup:ping"})
    )
    agent.tool_registry.register(
        "chat_lookup",
        lambda request: observed,
        enabled=True,
        capability="read",
        description="Read-only chat lookup",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        chat_exposed=True,
    )

    if projection_failure:

        def _fail(*args, **kwargs):
            raise RuntimeError("projection unavailable")

        if projection_failure == "trace":
            original_trace = agent.tracer.log

            def _fail_trace(event, *args, **kwargs):
                if event in {"native_tool_loop", "error"}:
                    raise RuntimeError("projection unavailable")
                return original_trace(event, *args, **kwargs)

            monkeypatch.setattr(agent.tracer, "log", _fail_trace)
        else:
            monkeypatch.setattr(
                agent,
                "_review_answer" if projection_failure == "review" else "_log_chat_interaction",
                _fail,
            )
    messages = [LLMMessage(role="user", content="use lookup")]
    if streaming:
        events = list(agent.respond_stream(messages))
        produced = [event for event in events if isinstance(event, ResponseProduced)]
        assert len(produced) == 1
        envelope = produced[0].response
        assert isinstance(events[-1], Done)
        assert sum(isinstance(event, Done) for event in events) == 1
    else:
        envelope = agent.respond(messages)
    response = envelope.text
    assert isinstance(envelope.runtime_result, AgentToolLoopResult)
    assert envelope.runtime_result.tool_calls[0].call.id == "lookup-1"
    assert envelope.runtime_result.tool_calls[0].result is observed
    assert envelope.runtime_result.tool_calls[0].result.ok is tool_ok
    if projection_failure:
        assert envelope.failure is not None
        assert "projection unavailable" in envelope.failure.message
    else:
        assert envelope.failure is None
        assert "tool loop final" in response
    assert brain.calls == 2
    assert (
        ToolSpec(
            name="chat_lookup",
            description="Read-only chat lookup",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        in brain.seen_tools
    )
    assert brain.messages_seen[-1][-1].role == "tool"


def test_agent_local_web_search_executes_before_non_xai_answer(tmp_path: Path) -> None:
    brain = SimpleBrain("answer from verified search")
    agent = Agent(
        brain=brain,
        main_config=ModelConfig(provider="local", model="local", web_search_enabled=True),
        enable_tools={"web": True, "safe_mode": False},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    web_tool = FakeWebTool(ToolResult.success({"output": "Source — https://example.test"}))
    agent.tool_registry.register("web", web_tool.handle, enabled=True, capability="read")

    response = agent.respond([LLMMessage(role="user", content="latest info")]).text

    assert "answer from verified search" in response
    assert web_tool.calls == 1
    assert brain.calls == 1
    assert any("Verified runtime web search evidence" in item.content for item in brain.messages)


def test_agent_local_web_search_error_blocks_final_answer(tmp_path: Path) -> None:
    brain = SimpleBrain("should not be emitted")
    agent = Agent(
        brain=brain,
        main_config=ModelConfig(provider="local", model="local", web_search_enabled=True),
        enable_tools={"web": True, "safe_mode": False},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    web_tool = FakeWebTool(ToolResult.failure("SERPER_API_KEY missing"))
    agent.tool_registry.register("web", web_tool.handle, enabled=True, capability="read")

    response = agent.respond([LLMMessage(role="user", content="latest info")]).text

    assert "web_search_not_executed: SERPER_API_KEY missing" in response
    assert "should not be emitted" not in response
    assert web_tool.calls == 1
    assert brain.calls == 0


def test_agent_xai_web_search_without_evidence_blocks_answer(tmp_path: Path) -> None:
    brain = SimpleBrain("I checked the internet and found this.")
    agent = Agent(
        brain=brain,
        main_config=ModelConfig(provider="xai", model="grok", web_search_enabled=True),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]

    response = agent.respond([LLMMessage(role="user", content="latest info")]).text

    assert "web_search_not_executed: xAI response contained no web search evidence" in response
    assert "I checked the internet" not in response


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("projection_failure", [False, True])
def test_agent_xai_web_search_with_evidence_allows_answer(
    tmp_path: Path, monkeypatch, streaming: bool, projection_failure: bool
) -> None:
    brain = SimpleBrain("answer from xAI native web search")
    brain.evidence = WebSearchEvidence(
        requested=True,
        executed=True,
        provider="xai_native",
        tool_call_seen=True,
        citations_count=1,
    )
    agent = Agent(
        brain=brain,
        main_config=ModelConfig(provider="xai", model="grok", web_search_enabled=True),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]

    if projection_failure:

        def _fail_review(_text):
            raise RuntimeError("review unavailable")

        monkeypatch.setattr(agent, "_review_answer", _fail_review)
    messages = [LLMMessage(role="user", content="latest info")]
    if streaming:
        events = list(agent.respond_stream(messages))
        produced = [event for event in events if isinstance(event, ResponseProduced)]
        assert len(produced) == 1
        envelope = produced[0].response
    else:
        envelope = agent.respond(messages)
    assert isinstance(envelope.runtime_result, LLMResult)
    assert envelope.runtime_result is brain.last_result
    assert envelope.runtime_result.web_search_evidence is brain.evidence
    assert envelope.runtime_result.text == "answer from xAI native web search"
    assert brain.calls == 1
    if projection_failure:
        assert envelope.failure is not None
        assert "review unavailable" in envelope.failure.message
    else:
        assert envelope.failure is None
        assert "answer from xAI native web search" in envelope.text
        assert "web_search_not_executed" not in envelope.text


def test_agent_blocks_web_claim_without_runtime_evidence(tmp_path: Path) -> None:
    brain = SimpleBrain("I checked the internet and found this.")
    agent = Agent(
        brain=brain,
        main_config=ModelConfig(provider="local", model="local"),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]

    response = agent.respond([LLMMessage(role="user", content="hello")]).text

    assert (
        "web_search_not_executed: assistant claimed web access without runtime evidence" in response
    )
    assert "I checked the internet" not in response


def test_agent_plan_execution_path(tmp_path: Path) -> None:
    brain = SimpleBrain("ok")
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    result = agent.respond([LLMMessage(role="user", content="планируй задачу")]).text
    assert "ok" in result


def test_agent_never_auto_saves_dialogue(tmp_path: Path) -> None:
    brain = SimpleBrain("hello")
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]

    calls = {"count": 0}

    def _save(_item: object) -> None:
        calls["count"] += 1

    agent.memory.save = _save  # type: ignore[method-assign]
    _ = agent.respond([LLMMessage(role="user", content="привет")]).text
    assert calls["count"] == 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_ask_network_approval_reaches_existing_handler(
    tmp_path: Path, streaming: bool, prefetch: bool
) -> None:
    brain = ToolLoopBrain()
    agent = Agent(
        brain=brain,
        main_config=ModelConfig(provider="local", model="local", web_search_enabled=prefetch),
        enable_tools={"safe_mode": True},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.runtime_mode = "ask"
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    executions: list[ToolRequest] = []
    tool_name = "web" if prefetch else "chat_lookup"
    agent.tool_registry.register(
        tool_name,
        lambda request: executions.append(request) or ToolResult.success({"output": "network"}),
        enabled=True,
        capability="read",
        risk_classes=["network"],
        description="Read-only network lookup",
        parameters_schema={"type": "object"},
        chat_exposed=True,
    )
    messages = [LLMMessage(role="user", content="lookup")]
    if streaming:
        events = list(agent.respond_stream(messages))
        produced = [event for event in events if isinstance(event, ResponseProduced)]
        assert len(produced) == 1
        envelope = produced[0].response
        assert isinstance(events[-1], Done)
        assert events[-1].finish_reason != "error"
    else:
        envelope = agent.respond(messages)
    assert agent.last_approval_request is not None
    assert agent.last_approval_request.tool == tool_name
    assert agent.last_approval_request.required_categories == ["NETWORK_RISK"]
    assert "APPROVAL_REQUIRED" in envelope.text
    assert envelope.failure is None
    assert envelope.runtime_result is None
    assert executions == []
    assert brain.calls == (0 if prefetch else 1)
