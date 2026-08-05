from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from llm.cancellation import cancel_generation
from llm.deepseek_brain import DeepSeekBrain
from llm.stream_model import Done, TextDelta
from llm.types import ModelConfig
from shared.models import LLMMessage


class _BlockingStreamResponse:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.closed = threading.Event()

    def raise_for_status(self) -> None:
        return

    def iter_lines(self, decode_unicode: bool = True):  # noqa: ANN201, ARG002
        self.started.set()
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'
        if not self.closed.wait(timeout=2):
            raise TimeoutError("test response was not closed")
        raise OSError("response closed")

    def close(self) -> None:
        self.closed.set()


def test_provider_http_response_closed_by_cancellation_token(monkeypatch) -> None:  # noqa: ANN001
    response = _BlockingStreamResponse()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("llm.deepseek_brain.requests.post", lambda *args, **kwargs: response)
    brain = DeepSeekBrain()
    token = asyncio.Event()
    config = ModelConfig(provider="deepseek", model="deepseek-chat")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            list,
            brain.generate_stream_events(
                [LLMMessage(role="user", content="hello")],
                config=config,
                cancellation_token=token,
            ),
        )
        assert response.started.wait(timeout=1)
        assert cancel_generation(token) == []
        events = future.result(timeout=2)

    assert response.closed.is_set()
    assert any(isinstance(event, TextDelta) for event in events)
    assert isinstance(events[-1], Done)
    assert events[-1].finish_reason == "cancelled"
