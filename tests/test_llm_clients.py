from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
import requests

from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.cancellation import cancel_generation
from llm.inception_brain import InceptionBrain
from llm.local_http_brain import LocalHttpBrain
from llm.openrouter_brain import OpenRouterBrain
from llm.retry import ProviderRequestError
from llm.stream_model import Done, TextDelta
from llm.types import LLMResult, ModelConfig, ToolSpec
from llm.xai_brain import XAiBrain
from shared.models import LLMMessage


def _mock_response(payload: dict[str, Any]):
    class Response:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return payload

        def raise_for_status(self) -> None:
            return None

    return Response()


def _mock_stream_response(lines: list[str]):
    class Response:
        status_code = 200
        encoding = "utf-8"

        def iter_lines(self, decode_unicode: bool = True):
            del decode_unicode
            yield from lines

        def raise_for_status(self) -> None:
            return None

    return Response()


def test_openrouter_generate(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_post(url, json, headers, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        return _mock_response(
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    monkeypatch.setattr("llm.openrouter_brain.requests.post", fake_post)
    config = ModelConfig(provider="openrouter", model="test-model", temperature=0.1)
    brain = OpenRouterBrain(api_key="test-key", default_config=config)

    result = brain.generate([LLMMessage(role="user", content="ping")])
    assert result.text == "hi"
    assert calls["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer test-key"
    assert calls["json"]["model"] == "test-model"


def test_local_http_generate(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_post(url, json, headers, timeout):
        calls["url"] = url
        calls["json"] = json
        return _mock_response({"choices": [{"message": {"content": "pong"}}]})

    monkeypatch.setattr("llm.local_http_brain.requests.post", fake_post)
    config = ModelConfig(
        provider="local",
        model="local-model",
        temperature=0.2,
        base_url="http://localhost:9999/v1/chat/completions",
    )
    brain = LocalHttpBrain(default_config=config)

    result = brain.generate([LLMMessage(role="user", content="hello")])
    assert result.text == "pong"
    assert calls["url"] == "http://localhost:9999/v1/chat/completions"
    assert calls["json"]["model"] == "local-model"


def test_local_http_generate_preserves_custom_provider_identity_in_errors(
    monkeypatch,
) -> None:
    provider_id = "custom-0123456789abcdef0123456789abcdef"

    def fake_post(url, json, headers, timeout, allow_redirects):
        del url, json, headers, timeout, allow_redirects
        response = requests.Response()
        response.status_code = 401
        raise requests.HTTPError("401 Client Error", response=response)

    monkeypatch.setattr("llm.local_http_brain.requests.post", fake_post)
    config = ModelConfig(
        provider=provider_id,
        model="opaque/model",
        temperature=0.2,
        base_url="https://example.test/v1/chat/completions",
    )
    brain = LocalHttpBrain(default_config=config, native_tools=False)

    with pytest.raises(ProviderRequestError) as exc_info:
        brain.generate([LLMMessage(role="user", content="hello")])

    assert exc_info.value.provider == provider_id
    assert exc_info.value.user_message == (f"Провайдер {provider_id} отклонил запрос (HTTP 401).")


def test_create_brain_rejects_arbitrary_provider_with_base_url() -> None:
    with pytest.raises(ValueError, match="Неизвестный провайдер"):
        create_brain(
            ModelConfig(
                provider="totally-legit",
                model="m",
                base_url="https://example.test/v1",
            )
        )


def test_create_brain_rejects_unqualified_custom_id_with_base_url() -> None:
    with pytest.raises(ValueError, match="Неизвестный провайдер"):
        create_brain(
            ModelConfig(
                provider="custom-evil",
                model="m",
                base_url="https://example.test/v1",
            )
        )


def test_create_brain_rejects_unknown_provider_without_base_url() -> None:
    with pytest.raises(ValueError, match="Неизвестный провайдер"):
        create_brain(ModelConfig(provider="nope", model="m"))


def test_create_brain_accepts_qualified_custom_provider() -> None:
    brain = create_brain(
        ModelConfig(
            provider="custom-0123456789abcdef0123456789abcdef",
            model="opaque/model",
            base_url="https://example.test/v1/chat/completions",
        )
    )
    assert isinstance(brain, LocalHttpBrain)
    assert brain.supports_native_tools is False
    assert brain.supports_streaming_tools is False


def test_dynamic_provider_streaming_falls_back_to_non_streaming_generate(
    monkeypatch,
) -> None:
    calls: dict[str, Any] = {}

    def fake_post(url, json, headers, timeout, allow_redirects):
        del url, headers, timeout, allow_redirects
        calls["json"] = json
        # The endpoint rejects stream=true: the brain must not request SSE.
        assert json.get("stream") is not True
        return _mock_response({"choices": [{"message": {"content": "hello world"}}]})

    monkeypatch.setattr("llm.local_http_brain.requests.post", fake_post)
    config = ModelConfig(
        provider="custom-0123456789abcdef0123456789abcdef",
        model="opaque/model",
        base_url="https://example.test/v1/chat/completions",
    )
    brain = LocalHttpBrain(default_config=config, native_tools=False)

    events = list(brain.generate_stream_events([LLMMessage(role="user", content="hi")]))

    assert calls["json"].get("stream") is not True
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "hello world"
    assert isinstance(events[-1], Done)


def test_dynamic_provider_plain_completion_is_cancellable(monkeypatch) -> None:
    provider_id = "custom-0123456789abcdef0123456789abcdef"
    token = asyncio.Event()
    post_calls: list[dict[str, Any]] = []
    body_read_started = threading.Event()
    release_body = threading.Event()

    class HangingResponse:
        status_code = 200

        def __init__(self) -> None:
            self._closed = False

        def raise_for_status(self) -> None:
            return None

        def close(self) -> None:
            self._closed = True
            release_body.set()

        def json(self) -> dict[str, Any]:
            body_read_started.set()
            # Simulate a hanging upstream: the body read blocks until the
            # response is closed (cancellation) or the wait below expires.
            release_body.wait(timeout=30)
            if self._closed:
                raise requests.exceptions.ChunkedEncodingError("connection closed")
            return {"choices": [{"message": {"content": "hi"}}]}

    def fake_post(url, json, headers, timeout, **kwargs):
        del url, headers, timeout, kwargs
        post_calls.append(json)
        # Plain completion: the OpenAI "stream" parameter must stay absent.
        assert json.get("stream") is not True
        return HangingResponse()

    monkeypatch.setattr("llm.local_http_brain.requests.post", fake_post)
    config = ModelConfig(
        provider=provider_id,
        model="opaque/model",
        base_url="https://example.test/v1/chat/completions",
    )
    brain = LocalHttpBrain(default_config=config, native_tools=False)

    events: list[Any] = []
    errors: list[BaseException] = []

    def run_generation() -> None:
        try:
            events.extend(
                brain.generate_stream_events(
                    [LLMMessage(role="user", content="hi")],
                    cancellation_token=token,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    started = time.monotonic()
    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()
    assert body_read_started.wait(timeout=10), "request did not reach the hanging body read"

    cancel_generation(token)
    thread.join(timeout=20)
    elapsed = time.monotonic() - started

    assert not thread.is_alive(), "cancellation did not stop the blocked generation"
    assert not errors, f"generation leaked an exception: {errors!r}"
    assert len(post_calls) == 1, "retry must not run after cancellation"
    assert elapsed < 15, "cancellation waited out the provider timeout"
    assert isinstance(events[-1], Done)
    assert events[-1].finish_reason == "cancelled"


def test_local_http_generate_sends_and_parses_native_tools(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_post(url, json, headers, timeout):
        del url, headers, timeout
        calls["json"] = json
        return _mock_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "workspace_read",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("llm.local_http_brain.requests.post", fake_post)
    config = ModelConfig(provider="local", model="local-model")
    brain = LocalHttpBrain(default_config=config)
    tools = [
        ToolSpec(
            name="workspace_read",
            description="Read a workspace file",
            parameters_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
    ]

    result = brain.generate([LLMMessage(role="user", content="read")], tools=tools)

    assert calls["json"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "workspace_read",
                "description": "Read a workspace file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    assert calls["json"]["tool_choice"] == "auto"
    assert result.text == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].name == "workspace_read"
    assert result.tool_calls[0].arguments == {"path": "README.md"}


def test_non_primary_providers_reject_generic_native_tools(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    tool = ToolSpec(name="workspace_read", description="Read")

    with pytest.raises(RuntimeError, match="OpenRouter"):
        OpenRouterBrain(
            api_key="openrouter-key",
            default_config=ModelConfig(provider="openrouter", model="debug-model"),
        ).generate([LLMMessage(role="user", content="read")], tools=[tool])

    with pytest.raises(RuntimeError, match="xAI"):
        XAiBrain(
            api_key="xai-key",
            default_config=ModelConfig(provider="xai", model="xai-model"),
        ).generate([LLMMessage(role="user", content="read")], tools=[tool])

    with pytest.raises(RuntimeError, match="Inception"):
        InceptionBrain(
            api_key="inception-key",
            default_config=ModelConfig(provider="inception", model="inception-model"),
        ).generate([LLMMessage(role="user", content="read")], tools=[tool])


def test_openrouter_without_key_raises(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    config = ModelConfig(provider="openrouter", model="test-model")
    brain = OpenRouterBrain(api_key=None, default_config=config)
    with pytest.raises(RuntimeError):
        brain.generate([LLMMessage(role="user", content="ping")])


def test_xai_generate(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_post(url, json, headers, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        return _mock_response(
            {
                "choices": [{"message": {"content": "xai-ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        )

    monkeypatch.setattr("llm.xai_brain.requests.post", fake_post)
    config = ModelConfig(provider="xai", model="xai-model", temperature=0.3)
    brain = XAiBrain(api_key="xai-key", default_config=config)

    result = brain.generate([LLMMessage(role="user", content="ping")])
    assert result.text == "xai-ok"
    assert calls["url"] == "https://api.x.ai/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer xai-key"
    assert calls["json"]["model"] == "xai-model"
    assert "tools" not in calls["json"]


def test_xai_generate_web_search_uses_responses_api(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_post(url, json, headers, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        del timeout
        return _mock_response(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "searched answer"},
                        ],
                    },
                ],
                "citations": [{"url": "https://example.test/source"}],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            }
        )

    monkeypatch.setattr("llm.xai_brain.requests.post", fake_post)
    config = ModelConfig(provider="xai", model="xai-model", web_search_enabled=True)
    brain = XAiBrain(api_key="xai-key", default_config=config)

    result = brain.generate([LLMMessage(role="user", content="latest news")])
    assert result.text == "searched answer"
    assert result.citations == [{"url": "https://example.test/source"}]
    assert result.usage is not None
    assert result.usage.prompt_tokens == 4
    assert calls["url"] == "https://api.x.ai/v1/responses"
    assert calls["headers"]["Authorization"] == "Bearer xai-key"
    assert calls["json"] == {
        "model": "xai-model",
        "input": [{"role": "user", "content": "latest news"}],
        "tools": [{"type": "web_search"}],
        "include": ["web_search_call.action.sources"],
    }
    assert "messages" not in calls["json"]
    assert "search_parameters" not in calls["json"]
    assert result.web_search_evidence is not None
    assert result.web_search_evidence.executed is True
    assert result.web_search_evidence.citations_count == 1


def test_xai_web_search_accepts_output_tool_call_evidence(monkeypatch) -> None:
    def fake_post(url, json, headers, timeout):
        del url, json, headers, timeout
        return _mock_response(
            {
                "output": [
                    {"type": "web_search_call", "name": "web_search"},
                    {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
                ],
            }
        )

    monkeypatch.setattr("llm.xai_brain.requests.post", fake_post)
    config = ModelConfig(provider="xai", model="xai-model", web_search_enabled=True)
    brain = XAiBrain(api_key="xai-key", default_config=config)

    result = brain.generate([LLMMessage(role="user", content="latest")])

    assert result.text == "answer"
    assert result.web_search_evidence is not None
    assert result.web_search_evidence.executed is True
    assert result.web_search_evidence.tool_call_seen is True


def test_xai_web_search_accepts_action_sources_evidence(monkeypatch) -> None:
    def fake_post(url, json, headers, timeout):
        del url, json, headers, timeout
        return _mock_response(
            {
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "sources": [
                                {"url": "https://example.test/source"},
                                "https://example.test/other",
                            ]
                        },
                    },
                    {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
                ],
            }
        )

    monkeypatch.setattr("llm.xai_brain.requests.post", fake_post)
    config = ModelConfig(provider="xai", model="xai-model", web_search_enabled=True)
    brain = XAiBrain(api_key="xai-key", default_config=config)

    result = brain.generate([LLMMessage(role="user", content="latest")])

    assert result.web_search_evidence is not None
    assert result.web_search_evidence.executed is True
    assert result.web_search_evidence.citations_count == 2


def test_xai_web_search_accepts_output_annotations_evidence(monkeypatch) -> None:
    def fake_post(url, json, headers, timeout):
        del url, json, headers, timeout
        return _mock_response(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "answer [[1]](https://example.test/source)",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.test/source",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr("llm.xai_brain.requests.post", fake_post)
    config = ModelConfig(provider="xai", model="xai-model", web_search_enabled=True)
    brain = XAiBrain(api_key="xai-key", default_config=config)

    result = brain.generate([LLMMessage(role="user", content="latest")])

    assert result.web_search_evidence is not None
    assert result.web_search_evidence.executed is True
    assert result.web_search_evidence.citations_count == 1


def test_xai_web_search_accepts_server_side_usage_evidence(monkeypatch) -> None:
    def fake_post(url, json, headers, timeout):
        del url, json, headers, timeout
        return _mock_response(
            {
                "output_text": "answer",
                "server_side_tool_usage": {"web_search_calls": 1},
            }
        )

    monkeypatch.setattr("llm.xai_brain.requests.post", fake_post)
    config = ModelConfig(provider="xai", model="xai-model", web_search_enabled=True)
    brain = XAiBrain(api_key="xai-key", default_config=config)

    result = brain.generate([LLMMessage(role="user", content="latest")])

    assert result.web_search_evidence is not None
    assert result.web_search_evidence.executed is True
    assert result.web_search_evidence.tool_call_seen is True


def test_xai_web_search_without_evidence_marks_not_executed(monkeypatch) -> None:
    def fake_post(url, json, headers, timeout):
        del url, json, headers, timeout
        return _mock_response({"output_text": "plain answer"})

    monkeypatch.setattr("llm.xai_brain.requests.post", fake_post)
    config = ModelConfig(provider="xai", model="xai-model", web_search_enabled=True)
    brain = XAiBrain(api_key="xai-key", default_config=config)

    result = brain.generate([LLMMessage(role="user", content="latest")])

    assert result.text == "plain answer"
    assert result.web_search_evidence is not None
    assert result.web_search_evidence.executed is False
    assert result.web_search_evidence.error == "xAI response contained no web search evidence"


def test_xai_stream_web_search_uses_responses_api(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_post(url, json, headers, timeout):
        calls["url"] = url
        calls["json"] = json
        del headers, timeout
        return _mock_response({"output_text": "streamed searched answer"})

    monkeypatch.setattr("llm.xai_brain.requests.post", fake_post)
    config = ModelConfig(provider="xai", model="xai-model", web_search_enabled=True)
    brain = XAiBrain(api_key="xai-key", default_config=config)

    events = list(brain.generate_stream_events([LLMMessage(role="user", content="latest")]))
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == (
        "streamed searched answer"
    )
    assert isinstance(events[-1], Done)
    assert calls["url"] == "https://api.x.ai/v1/responses"
    assert calls["json"]["tools"] == [{"type": "web_search"}]
    assert calls["json"]["include"] == ["web_search_call.action.sources"]
    assert "stream" not in calls["json"]


def test_xai_without_key_raises(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "")
    config = ModelConfig(provider="xai", model="xai-model")
    brain = XAiBrain(api_key=None, default_config=config)
    with pytest.raises(RuntimeError):
        brain.generate([LLMMessage(role="user", content="ping")])


def test_inception_generate_uses_reasoning_defaults(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_post(url, json, headers, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        del timeout
        return _mock_response(
            {
                "choices": [{"message": {"content": "inception-ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            }
        )

    monkeypatch.setattr("llm.inception_brain.requests.post", fake_post)
    config = ModelConfig(provider="inception", model="mercury-2")
    brain = InceptionBrain(api_key="inc-key", default_config=config)
    result = brain.generate([LLMMessage(role="user", content="ping")])
    assert result.text == "inception-ok"
    assert calls["url"] == "https://api.inceptionlabs.ai/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer inc-key"
    assert calls["json"]["reasoning_effort"] == "instant"
    assert calls["json"]["reasoning_summary"] is True
    assert calls["json"]["reasoning_summary_wait"] is False
    assert calls["json"]["model"] == "mercury-2"


def test_inception_stream_events_support_replace_mode(monkeypatch) -> None:
    def fake_post(url, json, headers, timeout, stream):  # noqa: ANN001
        del url, headers, timeout
        assert stream is True
        assert json["diffusing"] is True
        return _mock_stream_response(
            [
                'data: {"choices":[{"delta":{"content":"hel"}}]}',
                'data: {"choices":[{"delta":{"content":"hello"}}]}',
                "data: [DONE]",
            ]
        )

    monkeypatch.setattr("llm.inception_brain.requests.post", fake_post)
    config = ModelConfig(provider="inception", model="mercury-2", diffusing=True)
    brain = InceptionBrain(api_key="inc-key", default_config=config)
    events = list(brain.generate_stream_events([LLMMessage(role="user", content="ping")]))
    deltas = [event for event in events if isinstance(event, TextDelta)]
    assert [event.text for event in deltas] == ["hel", "hello"]
    assert [event.mode for event in deltas] == ["replace", "replace"]
    assert isinstance(events[-1], Done)


def test_inception_without_key_raises(monkeypatch) -> None:
    monkeypatch.setenv("INCEPTION_API_KEY", "")
    config = ModelConfig(provider="inception", model="mercury-2")
    brain = InceptionBrain(api_key=None, default_config=config)
    with pytest.raises(RuntimeError):
        brain.generate([LLMMessage(role="user", content="ping")])


def test_brain_factory_supports_all_known_providers() -> None:
    openrouter = create_brain(ModelConfig(provider="openrouter", model="or"))
    xai = create_brain(ModelConfig(provider="xai", model="xai"))
    local = create_brain(ModelConfig(provider="local", model="local"))
    inception = create_brain(ModelConfig(provider="inception", model="mercury-2"))
    assert isinstance(openrouter, OpenRouterBrain)
    assert isinstance(xai, XAiBrain)
    assert isinstance(local, LocalHttpBrain)
    assert isinstance(inception, InceptionBrain)


def test_brain_base_stream_events_default_adapter() -> None:
    class StaticBrain(Brain):
        def generate(self, messages, config=None):  # noqa: ANN001
            del messages, config
            return LLMResult(text="abcdef")

    brain = StaticBrain()
    events = list(brain.generate_stream_events([LLMMessage(role="user", content="ping")]))
    deltas = [event for event in events if isinstance(event, TextDelta)]
    assert deltas
    assert deltas[0].mode == "append"
    assert "".join(event.text for event in deltas) == "abcdef"
    assert isinstance(events[-1], Done)
