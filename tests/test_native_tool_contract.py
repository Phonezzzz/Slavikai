from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, LLMStreamChunk, ModelConfig, ToolCall, ToolSpec
from shared.models import JSONValue, LLMMessage, ToolRequest, ToolResult
from tools.tool_descriptors import TOOL_METADATA
from tools.tool_registry import ToolRegistry


def test_llm_message_supports_tool_role_and_provider_payload() -> None:
    message = LLMMessage(role="tool", content='{"ok": true}', tool_call_id="call-1")

    assert message.to_provider_dict() == {
        "role": "tool",
        "content": '{"ok": true}',
        "tool_call_id": "call-1",
    }


def test_llm_result_carries_native_tool_calls() -> None:
    result = LLMResult(
        text="",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="workspace_read",
                arguments={"path": "README.md"},
            )
        ],
    )

    assert result.tool_calls[0].name == "workspace_read"
    assert result.tool_calls[0].arguments == {"path": "README.md"}


def test_brain_contract_accepts_tools_for_generate_and_stream() -> None:
    class EchoBrain(Brain):
        def generate(
            self,
            messages: list[LLMMessage],
            config: ModelConfig | None = None,
            tools: list[ToolSpec] | None = None,
        ) -> LLMResult:
            del config
            tool_names = ",".join(tool.name for tool in tools or [])
            return LLMResult(text=f"{messages[-1].content}:{tool_names}")

    brain = EchoBrain()
    tools = [
        ToolSpec(
            name="workspace_read",
            description="Read a workspace file",
            parameters_schema={"type": "object"},
        )
    ]

    result = brain.generate([LLMMessage(role="user", content="ping")], tools=tools)
    stream = "".join(brain.generate_stream([LLMMessage(role="user", content="ping")], tools=tools))

    assert result.text == "ping:workspace_read"
    assert stream == "ping:workspace_read"


def test_brain_stream_chunks_pass_tools_to_generate_stream() -> None:
    class ChunkBrain(Brain):
        def generate(
            self,
            messages: list[LLMMessage],
            config: ModelConfig | None = None,
            tools: list[ToolSpec] | None = None,
        ) -> LLMResult:
            del messages, config, tools
            return LLMResult(text="")

        def generate_stream(
            self,
            messages: list[LLMMessage],
            config: ModelConfig | None = None,
            tools: list[ToolSpec] | None = None,
        ) -> Iterator[str]:
            del messages, config
            yield (tools or [ToolSpec(name="none", description="none")])[0].name

    chunks = list(
        ChunkBrain().generate_stream_chunks(
            [LLMMessage(role="user", content="ping")],
            tools=[ToolSpec(name="workspace_write", description="Write")],
        )
    )

    assert chunks == [LLMStreamChunk(text="workspace_write")]


def test_tool_registry_descriptor_exposes_llm_metadata() -> None:
    registry = ToolRegistry()
    schema: dict[str, JSONValue] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    registry.register(
        "workspace_read",
        _echo_tool,
        description="Read a file from the selected workspace",
        parameters_schema=schema,
        capability="read",
    )

    descriptor = registry.get_descriptor("workspace_read")
    assert descriptor is not None
    assert descriptor.description == "Read a file from the selected workspace"
    assert descriptor.parameters_schema == schema


def test_agent_runtime_tools_have_complete_llm_metadata(tmp_path: Path) -> None:
    agent = Agent(
        brain=EchoBrainForMetadata(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )

    for tool_name in TOOL_METADATA:
        descriptor = agent.tool_registry.get_descriptor(tool_name)
        assert descriptor is not None, tool_name
        assert descriptor.description.strip(), tool_name
        assert descriptor.parameters_schema.get("type") == "object", tool_name


class EchoBrainForMetadata(Brain):
    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        del messages, config, tools
        return LLMResult(text="ok")


def _echo_tool(request: ToolRequest) -> ToolResult:
    return ToolResult.success({"name": request.name})
