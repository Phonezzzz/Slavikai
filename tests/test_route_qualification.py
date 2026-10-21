from __future__ import annotations

from llm.brain_factory import create_brain
from llm.types import ModelConfig

CUSTOM = "custom-" + "1" * 32


def test_custom_missing_key_does_not_inherit_local_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "fixture-local-only")
    brain = create_brain(
        ModelConfig(
            provider=CUSTOM, model="fixture", base_url="https://fixture.invalid/v1/chat/completions"
        )
    )
    assert brain.api_key is None


def test_qualification_is_exact_and_revocable_before_dispatch(monkeypatch):
    from dataclasses import replace

    import pytest

    from llm.capabilities import NativeToolQualification
    from llm.local_http_brain import LocalHttpBrain
    from llm.types import ToolSpec
    from shared.models import LLMMessage

    config = ModelConfig(
        provider=CUSTOM, model="fixture", base_url="https://fixture.invalid/v1/chat/completions"
    )
    probe = LocalHttpBrain(config, api_key="fixture-key")
    current = [True]
    proof = NativeToolQualification(probe.request_binding(), lambda: current[0])
    brain = create_brain(config, api_key="fixture-key", qualification=proof)
    assert brain.native_tool_capability().allowed
    assert brain.native_tool_capability().model.source == "inert_protocol_v1"

    def reject(*args, **kwargs):
        raise AssertionError("Invalid proof dispatched")

    monkeypatch.setattr("requests.post", reject)
    for changed in [
        replace(config, model="other"),
        replace(config, base_url="https://other.invalid/v1/chat/completions"),
        replace(config, extra_headers={"X-Revision": "other"}),
    ]:
        with pytest.raises(RuntimeError, match="native_tools_required"):
            brain.generate(
                [LLMMessage(role="user", content="inert")],
                config=changed,
                tools=[ToolSpec(name="inert", description="test")],
            )
    assert (
        not create_brain(config, api_key="changed-key", qualification=proof)
        .native_tool_capability()
        .allowed
    )
    assert (
        not create_brain(
            replace(config, base_url="https://other.invalid/v1/chat/completions"),
            api_key="fixture-key",
            qualification=proof,
        )
        .native_tool_capability()
        .allowed
    )
    current[0] = False
    assert not brain.native_tool_capability().allowed
    with pytest.raises(RuntimeError, match="native_tools_required"):
        list(brain.generate_stream_events([], tools=[ToolSpec(name="inert", description="test")]))


def test_qualified_route_keeps_tool_gateway_policy(monkeypatch):
    import requests

    from core.tool_gateway import ToolGateway
    from core.tool_loop import AgentToolLoop
    from llm.capabilities import NativeToolQualification
    from llm.local_http_brain import LocalHttpBrain
    from shared.models import LLMMessage, ToolResult
    from tools.tool_registry import ToolRegistry

    calls = []
    registry = ToolRegistry()
    registry.register(
        "inert_fixture",
        lambda request: calls.append(request) or ToolResult.success({}),
        model_exposed=False,
    )

    def response(*args, **kwargs):
        result = requests.Response()
        result.status_code = 200
        result._content = (
            b'{"choices":[{"message":{"content":"","tool_calls":[{"id":"denied",'
            b'"function":{"name":"inert_fixture","arguments":"{}"}}]}}]}'
        )
        return result

    monkeypatch.setattr("requests.post", response)
    for provider in ["local", CUSTOM]:
        config = ModelConfig(
            provider=provider,
            model="fixture",
            base_url="https://fixture.invalid/v1/chat/completions",
        )
        probe = LocalHttpBrain(config, api_key="fixture")
        proof = NativeToolQualification(probe.request_binding(), lambda: True)
        brain = create_brain(config, api_key="fixture", qualification=proof)
        result = AgentToolLoop(max_iterations=1).run(
            brain=brain,
            gateway=ToolGateway(registry),
            messages=[LLMMessage(role="user", content="inert")],
            tools=registry.list_tool_specs(),
            config=config,
        )
        assert result.tool_calls[0].result.meta["policy_reason"] == "model_tool_not_exposed"
    assert calls == []
