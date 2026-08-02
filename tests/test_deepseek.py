from __future__ import annotations

import requests

from llm.brain_factory import create_brain
from llm.deepseek_brain import DeepSeekBrain
from llm.types import LLMMessage, ModelConfig, ToolSpec

DEEPSEEK_CONFIG = ModelConfig(provider="deepseek", model="deepseek-v4-flash")


def test_create_brain_deepseek(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    brain = create_brain(DEEPSEEK_CONFIG)
    assert isinstance(brain, DeepSeekBrain)
    assert brain.api_key == "sk-test"


def test_deepseek_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    brain = DeepSeekBrain(default_config=DEEPSEEK_CONFIG)
    try:
        brain.generate([LLMMessage(role="user", content="test")])
    except RuntimeError as exc:
        assert "deepseek_api_key" in str(exc).lower()
        return
    raise AssertionError("Expected RuntimeError was not raised.")


def test_deepseek_generate_payload(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    capture: list[dict] = []

    def fake_post(url, **kwargs):  # noqa: ARG001
        capture.append({"url": url, "json": kwargs.get("json"), "headers": kwargs.get("headers")})

        class FakeResponse:
            status_code = 200
            encoding = "utf-8"

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "message": {"content": "OK", "role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }

            @staticmethod
            def raise_for_status():
                pass

        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    brain = DeepSeekBrain()
    result = brain.generate([LLMMessage(role="user", content="hi")], config=DEEPSEEK_CONFIG)

    assert result.text == "OK"
    assert result.usage is not None
    assert result.usage.total_tokens == 2
    assert len(capture) == 1
    assert capture[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert capture[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert capture[0]["json"]["model"] == "deepseek-v4-flash"
    assert capture[0]["json"]["temperature"] == 0.7


def test_deepseek_parses_text_reasoning_usage(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def fake_post(url, **kwargs):  # noqa: ARG001
        class FakeResponse:
            status_code = 200
            encoding = "utf-8"

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "42",
                                "reasoning_content": "The answer is the meaning of life.",
                                "role": "assistant",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }

            @staticmethod
            def raise_for_status():
                pass

        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    brain = DeepSeekBrain()
    cfg = DEEPSEEK_CONFIG
    result = brain.generate([LLMMessage(role="user", content="what is 6*7")], config=cfg)

    assert result.text == "42"
    assert result.reasoning == "The answer is the meaning of life."
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


def test_deepseek_parses_native_tool_calls(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    def fake_post(url, **kwargs):  # noqa: ARG001
        class FakeResponse:
            status_code = 200
            encoding = "utf-8"

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"London"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                }

            @staticmethod
            def raise_for_status():
                pass

        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    tool_spec = ToolSpec(
        name="get_weather",
        description="Get weather for a city",
        parameters_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    brain = DeepSeekBrain()
    result = brain.generate(
        [LLMMessage(role="user", content="weather London")],
        config=DEEPSEEK_CONFIG,
        tools=[tool_spec],
    )

    assert result.text == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"city": "London"}


def test_deepseek_streaming(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    chunks = [
        'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
        'data: {"choices":[{"delta":{"content":" world"},"index":0}]}\n\n',
        "data: [DONE]\n\n",
    ]

    class FakeStreamResponse:
        status_code = 200
        encoding = "utf-8"

        def __init__(self) -> None:
            self._lines = chunks

        def raise_for_status(self) -> None:
            pass

        def iter_lines(self, decode_unicode: bool = True):  # noqa: ARG002
            yield from self._lines

    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeStreamResponse())
    brain = DeepSeekBrain()
    output = list(
        brain.generate_stream(
            [LLMMessage(role="user", content="hello")],
            config=DEEPSEEK_CONFIG,
        )
    )
    assert output == ["Hello", " world"]


def test_deepseek_provider_exposed_in_settings() -> None:
    from server.http.common.ui_settings import (
        API_KEY_SETTINGS_PROVIDERS,
        PROVIDER_API_KEY_ENV,
        SUPPORTED_MODEL_PROVIDERS,
    )

    assert "deepseek" in SUPPORTED_MODEL_PROVIDERS
    assert "deepseek" in API_KEY_SETTINGS_PROVIDERS
    assert PROVIDER_API_KEY_ENV.get("deepseek") == "DEEPSEEK_API_KEY"


def test_deepseek_models_exposed_in_model_picker() -> None:
    from server.http.common.ui_settings import (
        SUPPORTED_MODEL_PROVIDERS,
        _build_model_config,
        _provider_models_endpoint,
    )

    assert "deepseek" in SUPPORTED_MODEL_PROVIDERS
    config = _build_model_config("deepseek", "deepseek-v4-pro")
    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-pro"
    endpoint = _provider_models_endpoint("deepseek")
    assert endpoint == "https://api.deepseek.com/models"
