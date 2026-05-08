from __future__ import annotations

from typing import Any

import pytest

from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.inception_brain import InceptionBrain
from llm.local_http_brain import LocalHttpBrain
from llm.openrouter_brain import OpenRouterBrain
from llm.types import ModelConfig
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

    chunks = list(brain.generate_stream([LLMMessage(role="user", content="latest")]))
    assert "".join(chunks) == "streamed searched answer"
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


def test_inception_stream_chunks_support_replace_mode(monkeypatch) -> None:
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
    chunks = list(brain.generate_stream_chunks([LLMMessage(role="user", content="ping")]))
    assert [chunk.text for chunk in chunks] == ["hel", "hello"]
    assert [chunk.mode for chunk in chunks] == ["replace", "replace"]


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


def test_brain_base_stream_chunks_default_adapter() -> None:
    class StaticBrain(Brain):
        def generate(self, messages, config=None):  # noqa: ANN001
            del messages, config
            return type("Result", (), {"text": "abcdef"})()

    brain = StaticBrain()
    chunks = list(brain.generate_stream_chunks([LLMMessage(role="user", content="ping")]))
    assert chunks
    assert chunks[0].mode == "append"
    assert "".join(chunk.text for chunk in chunks) == "abcdef"
