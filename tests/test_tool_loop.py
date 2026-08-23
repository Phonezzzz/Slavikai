from __future__ import annotations

from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.stream_model import Done, ToolCallCompleted
from llm.types import LLMResult, ToolCall, ToolSpec
from shared.models import LLMMessage, ToolResult
from tools.tool_registry import ToolRegistry


class LoopBrain:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_tool_specs: list[ToolSpec] = []

    def generate(self, messages, config=None, tools=None):  # type: ignore[override]
        del config
        self.calls += 1
        self.seen_tool_specs = list(tools or [])
        if self.calls == 1:
            return LLMResult(
                text="calling",
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"value": messages[-1].content})
                ],
            )
        assert messages[-1].role == "tool"
        return LLMResult(text=f"final:{messages[-1].content}")


def test_agent_tool_loop_executes_native_tool_call_and_appends_tool_message() -> None:
    registry = ToolRegistry()
    registry.register(
        "echo",
        lambda request: ToolResult.success({"output": request.args["value"]}),
        description="Echo a value",
        parameters_schema={"type": "object"},
    )
    brain = LoopBrain()

    result = AgentToolLoop().run(
        brain=brain,  # type: ignore[arg-type]
        gateway=ToolGateway(registry),
        messages=[LLMMessage(role="user", content="ping")],
        tools=registry.list_tool_specs(),
    )

    assert brain.calls == 2
    assert brain.seen_tool_specs == [
        ToolSpec(name="echo", description="Echo a value", parameters_schema={"type": "object"})
    ]
    assert result.iterations == 2
    assert result.tool_calls[0].call.name == "echo"
    assert result.tool_calls[0].result.ok
    assert result.messages[-2].role == "tool"
    assert result.text.startswith("final:")


def test_agent_tool_loop_stops_after_max_iterations() -> None:
    class AlwaysToolBrain:
        def generate(self, messages, config=None, tools=None):  # type: ignore[override]
            del messages, config, tools
            return LLMResult(
                text="again",
                tool_calls=[ToolCall(id="call-1", name="noop", arguments={})],
            )

    registry = ToolRegistry()
    registry.register("noop", lambda _request: ToolResult.success({}))

    result = AgentToolLoop(max_iterations=2).run(
        brain=AlwaysToolBrain(),  # type: ignore[arg-type]
        gateway=ToolGateway(registry),
        messages=[LLMMessage(role="user", content="go")],
        tools=registry.list_tool_specs(),
    )

    assert result.iterations == 2
    assert len(result.tool_calls) == 2


def test_agent_tool_loop_blocks_registered_tool_not_exposed_to_model() -> None:
    class HiddenToolBrain:
        def generate(self, messages, config=None, tools=None):  # type: ignore[override]
            del messages, config, tools
            return LLMResult(
                text="hidden",
                tool_calls=[ToolCall(id="hidden-1", name="runtime_cleanup", arguments={})],
            )

    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        "runtime_cleanup",
        lambda request: calls.append(request.name) or ToolResult.success({}),
        model_exposed=False,
    )

    result = AgentToolLoop(max_iterations=1).run(
        brain=HiddenToolBrain(),  # type: ignore[arg-type]
        gateway=ToolGateway(registry),
        messages=[LLMMessage(role="user", content="go")],
        tools=registry.list_tool_specs(),
    )

    assert calls == []
    assert result.tool_calls[0].result.meta["policy_reason"] == "model_tool_not_exposed"


def test_streaming_tool_loop_blocks_registered_tool_not_exposed_to_model() -> None:
    class HiddenStreamingBrain:
        def generate_stream_events(self, messages, config=None, tools=None):  # type: ignore[override]
            del messages, config, tools
            yield ToolCallCompleted(
                call=ToolCall(id="hidden-stream-1", name="runtime_cleanup", arguments={})
            )
            yield Done()

    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(
        "runtime_cleanup",
        lambda request: calls.append(request.name) or ToolResult.success({}),
        model_exposed=False,
    )

    events = list(
        AgentToolLoop(max_iterations=1).run_stream_events(
            brain=HiddenStreamingBrain(),  # type: ignore[arg-type]
            gateway=ToolGateway(registry),
            messages=[LLMMessage(role="user", content="go")],
            tools=registry.list_tool_specs(),
        )
    )

    completed = next(event for event in events if isinstance(event, ToolCallCompleted))
    assert calls == []
    assert completed.result is not None
    assert completed.result.meta["policy_reason"] == "model_tool_not_exposed"
