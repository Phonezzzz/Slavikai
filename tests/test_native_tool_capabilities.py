from __future__ import annotations

import pytest

from core.plan_compiler import PlanCompilationError, compile_structured_plan_steps
from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.types import LLMResult, ModelConfig


class UnsupportedBrain(Brain):
    def generate(self, messages, config=None, tools=None):
        raise AssertionError("Admission must prevent provider dispatch")


@pytest.mark.parametrize("provider", ["local", "deepseek"])
def test_builtin_declaration_does_not_claim_model_qualification(provider):
    brain = create_brain(ModelConfig(provider=provider, model="opaque-model"), api_key="test")
    capability = brain.native_tool_capability()
    assert capability.allowed
    assert capability.eligibility == "legacy_adapter"
    assert capability.adapter.state == "supported"
    assert capability.adapter.source == "adapter_declaration"
    assert capability.model.state == "unknown"
    assert capability.model.source == "not_observed"


@pytest.mark.parametrize(
    "model,endpoint,key",
    [
        ("first", "http://127.0.0.1:1234/v1", "secret-one"),
        ("second", "http://127.0.0.1:5678/v1", "secret-two"),
    ],
)
def test_custom_route_remains_unqualified_without_dispatch(monkeypatch, model, endpoint, key):
    def reject_network(*args, **kwargs):
        raise AssertionError("Evidence read must not probe the endpoint")

    monkeypatch.setattr("requests.post", reject_network)
    config = ModelConfig(
        provider="custom-0123456789abcdef0123456789abcdef",
        model=model,
        base_url=endpoint,
        api_key=key,
    )
    brain = create_brain(config)
    capability = brain.native_tool_capability()
    assert not capability.allowed
    assert capability.eligibility == "unqualified"
    assert capability.adapter.state == "supported"
    assert capability.model.state == "unknown"
    assert key not in repr(capability)
    with pytest.raises(PlanCompilationError) as error:
        compile_structured_plan_steps(
            brain=brain,
            config=config,
            goal="Inspect",
            audit_log=[],
            available_tools=[],
        )
    assert error.value.code == "native_tools_required"
    assert "не проверена" in error.value.message


def test_unsupported_adapter_is_distinct_from_unqualified_model():
    brain = UnsupportedBrain()
    capability = brain.native_tool_capability()
    assert not capability.allowed
    assert capability.eligibility == "unsupported"
    assert capability.adapter.state == "unsupported"
    assert capability.model.state == "unknown"
    with pytest.raises(PlanCompilationError) as error:
        compile_structured_plan_steps(
            brain=brain,
            config=None,
            goal="Inspect",
            audit_log=[],
            available_tools=[],
        )
    assert "не поддерживает" in error.value.message


def test_assessment_is_a_fresh_immutable_observation():
    class DeclaredBrain(UnsupportedBrain):
        supports_native_tools = True

        def generate(self, messages, config=None, tools=None):
            return LLMResult(text="unused")

    brain = DeclaredBrain()
    old = brain.native_tool_capability()
    brain.supports_native_tools = False
    assert old.allowed
    assert not brain.native_tool_capability().allowed


def test_plan_obeys_typed_admission_even_when_adapter_flag_is_positive():
    from llm.capabilities import CapabilityEvidence, NativeToolCapability

    class UnqualifiedBrain(UnsupportedBrain):
        supports_native_tools = True

        def native_tool_capability(self):
            return NativeToolCapability(
                adapter=CapabilityEvidence("supported", "adapter_declaration"),
                model=CapabilityEvidence("unknown", "not_observed"),
                eligibility="unqualified",
            )

    with pytest.raises(PlanCompilationError) as error:
        compile_structured_plan_steps(
            brain=UnqualifiedBrain(),
            config=None,
            goal="Inspect",
            audit_log=[],
            available_tools=[],
        )
    assert error.value.code == "native_tools_required"
    assert "не проверена" in error.value.message
