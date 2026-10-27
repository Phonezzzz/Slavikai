import json

import pytest

from llm.deepseek_brain import DeepSeekBrain
from llm.inception_brain import InceptionBrain
from llm.local_http_brain import LocalHttpBrain
from llm.openrouter_brain import OpenRouterBrain
from llm.stream_model import (
    Done,
    Error,
    TextDelta,
    ToolCallCompleted,
    Usage,
    iter_openai_sse_events,
)
from llm.types import LLMUsage, ModelConfig, ToolSpec
from llm.xai_brain import XAiBrain
from shared.models import LLMMessage
from tests.test_llm_clients import _mock_response


def payload(usage):
    return {
        "choices": [
            {
                "message": {
                    "content": "fixture answer",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                }
            }
        ],
        "usage": usage,
    }


@pytest.mark.parametrize("usage", [{"total_tokens": 4}, {"total_tokens": None}])
def test_local_generate_keeps_response_and_tools_with_unknown_usage(monkeypatch, usage):
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _mock_response(payload(usage)))
    brain = LocalHttpBrain(
        ModelConfig(provider="local", model="fixture"),
        base_url="http://fixture.invalid/v1/chat/completions",
        api_key="fixture",
    )
    result = brain.generate(
        [LLMMessage(role="user", content="go")],
        tools=[ToolSpec(name="echo", description="fixture")],
    )
    assert result.text == "fixture answer"
    assert result.tool_calls[0].name == "echo"
    assert result.usage is None


@pytest.mark.parametrize("usage", [{"total_tokens": 4}, {"total_tokens": None}])
def test_stream_keeps_text_tools_done_without_false_zero_usage(usage):
    chunk = {
        "choices": [
            {
                "delta": {
                    "content": "fixture",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                }
            }
        ],
        "usage": usage,
    }
    events = list(iter_openai_sse_events(["data: " + json.dumps(chunk), "data: [DONE]"]))
    assert any(isinstance(event, TextDelta) and event.text == "fixture" for event in events)
    assert any(
        isinstance(event, ToolCallCompleted) and event.call.name == "echo" for event in events
    )
    assert isinstance(events[-1], Done)
    assert not any(isinstance(event, (Usage, Error)) for event in events)


PROVIDERS = ["local", "deepseek", "inception", "openrouter", "xai"]
UNKNOWN_USAGE = [None, {}, [], "PRIVATE_USAGE", 0]
for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
    complete = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    UNKNOWN_USAGE.append({key: value for key, value in complete.items() if key != field})
    for invalid in [None, True, "PRIVATE_USAGE", 2.0, -1]:
        UNKNOWN_USAGE.append({**complete, field: invalid})


def brain_for(provider, *, web_search=False):
    config = ModelConfig(
        provider=provider,
        model="fixture",
        web_search_enabled=web_search,
        base_url="https://fixture.invalid/v1/chat/completions",
    )
    if provider == "local":
        return LocalHttpBrain(config, api_key="fixture")
    cls = {
        "deepseek": DeepSeekBrain,
        "inception": InceptionBrain,
        "openrouter": OpenRouterBrain,
        "xai": XAiBrain,
    }[provider]
    return cls(api_key="fixture", default_config=config)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("usage", UNKNOWN_USAGE)
def test_sync_adapters_preserve_valid_response_with_unknown_metadata(monkeypatch, provider, usage):
    requested = []

    def respond(*args, **kwargs):
        requested.append(True)
        return _mock_response(payload(usage))

    monkeypatch.setattr("requests.post", respond)
    result = brain_for(provider).generate([LLMMessage(role="user", content="go")])
    assert result.text == "fixture answer" and result.usage is None
    assert requested == [True]
    if provider in {"local", "deepseek"}:
        assert result.tool_calls[0].name == "echo"


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("counts", [(0, 0, 0), (3, 2, 5), (1, 2, 99)])
def test_explicit_complete_counts_are_reported_without_coercion(monkeypatch, provider, counts):
    usage = dict(zip(["prompt_tokens", "completion_tokens", "total_tokens"], counts, strict=True))
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _mock_response(payload(usage)))
    result = brain_for(provider).generate([LLMMessage(role="user", content="go")])
    assert result.usage == LLMUsage(*counts)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_missing_usage_block_is_unknown(monkeypatch, provider):
    response = payload(None)
    del response["usage"]
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _mock_response(response))
    result = brain_for(provider).generate([LLMMessage(role="user", content="go")])
    assert result.text == "fixture answer" and result.usage is None


@pytest.mark.parametrize("usage", UNKNOWN_USAGE)
def test_stream_unknown_metadata_never_creates_usage_event(usage):
    events = list(
        iter_openai_sse_events(
            [
                "data: "
                + json.dumps({"choices": [{"delta": {"content": "kept"}}], "usage": usage}),
                "data: [DONE]",
            ]
        )
    )
    assert any(isinstance(event, TextDelta) and event.text == "kept" for event in events)
    assert isinstance(events[-1], Done)
    assert not any(isinstance(event, (Usage, Error)) for event in events)


@pytest.mark.parametrize(
    "usage,expected",
    [
        ({"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}, LLMUsage(4, 2, 6)),
        (
            {"prompt_tokens": 0, "input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            LLMUsage(0, 2, 6),
        ),
        ({"prompt_tokens": None, "input_tokens": 4, "output_tokens": 2, "total_tokens": 6}, None),
        ({"input_tokens": True, "output_tokens": 2, "total_tokens": 6}, None),
        ({"input_tokens": 4, "output_tokens": 2}, None),
    ],
)
@pytest.mark.parametrize("responses", [False, True])
def test_xai_existing_aliases_and_precedence(monkeypatch, usage, expected, responses):
    response = payload(usage)
    if responses:
        response = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "fixture answer"}]}
            ],
            "usage": usage,
            "citations": [{"url": "https://fixture.invalid/source"}],
        }
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: _mock_response(response))
    result = brain_for("xai", web_search=responses).generate(
        [LLMMessage(role="user", content="go")]
    )
    assert result.text == "fixture answer" and result.usage == expected
    if responses:
        assert result.citations == response["citations"]


def test_stream_preserves_separate_complete_updates_and_explicit_zero():
    chunks = [
        {"usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}}
        for p, c, t in [(0, 0, 0), (1, 2, 3), (2, 3, 5)]
    ]
    events = list(
        iter_openai_sse_events(
            [*("data: " + json.dumps(chunk) for chunk in chunks), "data: [DONE]"]
        )
    )
    assert [event.usage for event in events if isinstance(event, Usage)] == [
        LLMUsage(0, 0, 0),
        LLMUsage(1, 2, 3),
        LLMUsage(2, 3, 5),
    ]
    assert isinstance(events[-1], Done)


def test_invalid_usage_does_not_hide_provider_error():
    events = list(
        iter_openai_sse_events(
            [
                'data: {"error":{"message":"upstream rejected"},'
                '"usage":{"total_tokens":"PRIVATE_USAGE"}}',
                "data: [DONE]",
            ]
        )
    )
    assert any(
        isinstance(event, Error) and event.message == "upstream rejected" for event in events
    )
    assert not any(isinstance(event, Usage) for event in events)
