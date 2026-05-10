from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig, ToolCall, ToolSpec, WebSearchEvidence
from shared.models import LLMMessage, PlanStep, PlanStepStatus, TaskPlan, ToolRequest, ToolResult


class SimpleBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.messages: list[LLMMessage] = []
        self.evidence: WebSearchEvidence | None = None

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        self.messages = list(messages)
        return LLMResult(text=self.text, web_search_evidence=self.evidence)


class ToolLoopBrain(Brain):
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


class FakePlanner:
    def classify_complexity(self, _: str):
        from shared.models import TaskComplexity

        return TaskComplexity.COMPLEX

    def build_plan(self, goal: str, brain=None, model_config=None) -> TaskPlan:
        return TaskPlan(
            goal=goal, steps=[PlanStep(description="step1"), PlanStep(description="step2")]
        )

    def _parse_plan_text(self, text: str):
        return [line.strip() for line in text.splitlines() if line.strip()]


class FakeExecutor:
    def __init__(self) -> None:
        self.run_called = False

    def run(self, plan: TaskPlan, tool_gateway=None) -> TaskPlan:
        self.run_called = True
        for step in plan.steps:
            step.status = PlanStepStatus.DONE
            step.result = "ok"
        return plan


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
    response = agent.respond([LLMMessage(role="user", content="привет")])
    assert "hello" in response
    assert brain.calls >= 1


def test_agent_chat_response_can_use_read_only_native_tool_loop(tmp_path: Path) -> None:
    brain = ToolLoopBrain()
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.tool_registry.register(
        "chat_lookup",
        lambda request: ToolResult.success({"output": f"lookup:{request.args['query']}"}),
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

    response = agent.respond([LLMMessage(role="user", content="use lookup")])

    assert "tool loop final" in response
    assert brain.calls == 2
    assert brain.seen_tools == [
        ToolSpec(
            name="chat_lookup",
            description="Read-only chat lookup",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
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

    response = agent.respond([LLMMessage(role="user", content="latest info")])

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

    response = agent.respond([LLMMessage(role="user", content="latest info")])

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

    response = agent.respond([LLMMessage(role="user", content="latest info")])

    assert "web_search_not_executed: xAI response contained no web search evidence" in response
    assert "I checked the internet" not in response


def test_agent_xai_web_search_with_evidence_allows_answer(tmp_path: Path) -> None:
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

    response = agent.respond([LLMMessage(role="user", content="latest info")])

    assert "answer from xAI native web search" in response
    assert "web_search_not_executed" not in response


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

    response = agent.respond([LLMMessage(role="user", content="hello")])

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
    agent.planner = FakePlanner()  # type: ignore[assignment]
    agent.executor = FakeExecutor()  # type: ignore[assignment]
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    result = agent.respond([LLMMessage(role="user", content="планируй задачу")])
    assert "ok" in result
    assert not agent.executor.run_called  # type: ignore[attr-defined]


def test_agent_does_not_auto_save_dialogue_by_default(tmp_path: Path) -> None:
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

    def _save_to_memory(_prompt: str, _answer: str) -> None:
        calls["count"] += 1

    agent.save_to_memory = _save_to_memory  # type: ignore[method-assign]
    assert agent.memory_config.auto_save_dialogue is False

    _ = agent.respond([LLMMessage(role="user", content="привет")])
    assert calls["count"] == 0
