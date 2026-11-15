from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import aiohttp
import pytest

from core.approval_policy import ApprovalContext, ApprovalRequired
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop, ToolLoopExecutionError
from llm.cancellation import cancel_generation
from llm.chatgpt_plan import ChatGPTPlan, ChatGPTPlanError
from llm.chatgpt_plan_brain import ChatGPTPlanBrain
from llm.stream_model import Done, Error, TextDelta, ToolCallCompleted
from llm.types import ModelConfig
from shared.models import LLMMessage, ToolResult
from tests.fake_provider_http import FakeHttpResponse, HoldingProvider
from tools.tool_registry import ToolRegistry

PROVIDER = "chatgpt-plan-" + "a" * 32
PRINCIPAL = "owner@example.test"
CONFIG = ModelConfig(provider=PROVIDER, model="synthetic-gpt", max_tokens=3, top_p=0.5)
MESSAGES = [LLMMessage("system", "Guide"), LLMMessage("user", "Ping")]
FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "namespace": "slavikai",
    "call_id": "call_1",
    "name": "echo",
    "arguments": '{"value":"Ping"}',
    "status": "completed",
}
REASONING = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"}
USAGE = {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16}


class FakePlan:
    active = True

    def access(self, principal, provider):
        self.validate_grant(principal, provider)
        return "fake-siwc-token"

    def validate_grant(self, principal, provider):
        assert (principal, provider) == (PRINCIPAL, PROVIDER)
        if not self.active:
            raise ChatGPTPlanError("chatgpt_plan_revoked", "Разрешение отозвано.", status=401)


def completed(*items):
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": list(items),
            "usage": USAGE,
        },
    }


def assistant(text):
    return {
        "type": "message",
        "id": "msg_final",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def sse(*events):
    for event in events:
        yield f"data: {json.dumps(event)}\n"
        yield "\n"


@pytest.mark.parametrize("streaming", [False, True])
def test_responses_native_roundtrip_uses_gateway_and_stateless_history(monkeypatch, streaming):
    requests = []
    responses = iter(
        [
            [completed(REASONING, FUNCTION_CALL)],
            [{"type": "response.output_text.delta", "delta": "Do"}, completed(assistant("Done"))],
        ]
    )

    def post(**kwargs):
        requests.append(kwargs)
        yield from sse(*next(responses))

    monkeypatch.setattr("llm.chatgpt_plan_brain.post_stream_lines", post)

    calls = []
    registry = ToolRegistry()

    def echo(request):
        calls.append(request.args)
        return ToolResult.success({"value": request.args["value"]})

    registry.register("echo", echo, parameters_schema={"type": "object"})
    arguments = {
        "brain": ChatGPTPlanBrain(cast(ChatGPTPlan, FakePlan()), PRINCIPAL, CONFIG),
        "gateway": ToolGateway(registry),
        "messages": MESSAGES,
        "tools": registry.list_tool_specs(),
        "config": CONFIG,
    }
    loop = AgentToolLoop()
    if streaming:
        events = list(loop.run_stream_events(**arguments))
        assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "Done"
        assert [event.result.data for event in events if isinstance(event, ToolCallCompleted)] == [
            {"value": "Ping"}
        ]
        assert isinstance(events[-1], Done)
        assert events[-1].responses_items == ()
    else:
        result = loop.run(**arguments, record_model_usage=True)
        assert result.text == "Done"
        assert result.model_usage.total_tokens == 32
        assert "opaque" not in repr(result)
        assert "responses_items" not in result.messages[2].to_provider_dict()

    assert calls == [{"value": "Ping"}]
    first, second = requests
    assert first["url"] == "https://api.openai.com/v1/responses"
    assert first["headers"]["Authorization"] == "Bearer fake-siwc-token"
    assert first["payload"] == {
        "model": CONFIG.model,
        "input": [{"role": "developer", "content": "Guide"}, {"role": "user", "content": "Ping"}],
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "tools": [
            {
                "type": "namespace",
                "name": "slavikai",
                "description": "Инструменты SlavikAI с локальной проверкой полномочий.",
                "tools": [
                    {
                        "type": "function",
                        "name": "echo",
                        "description": "",
                        "parameters": {"type": "object"},
                        "strict": False,
                    }
                ],
            }
        ],
    }
    assert second["payload"]["input"][:-1] == first["payload"]["input"] + [REASONING, FUNCTION_CALL]
    output = second["payload"]["input"][-1]
    assert output["type"] == "function_call_output" and output["call_id"] == "call_1"
    assert json.loads(output["output"]) == {
        "trust": "untrusted_observation",
        "ok": True,
        "data": {"value": "Ping"},
        "error": None,
        "meta": None,
    }


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("terminal", "expected"),
    [
        (None, "chatgpt_plan_interrupted"),
        ({"type": "response.incomplete"}, "chatgpt_plan_incomplete"),
        (
            {
                "type": "response.failed",
                "response": {
                    "error": {
                        "code": "subscription_sharing_usage_limit_exceeded",
                        "message": "Private upstream error detail",
                    }
                },
            },
            "subscription_sharing_usage_limit_exceeded",
        ),
        (completed({**FUNCTION_CALL, "namespace": "other"}), "chatgpt_plan_protocol"),
        (completed(FUNCTION_CALL, FUNCTION_CALL), "chatgpt_plan_protocol"),
    ],
)
def test_unfinished_and_invalid_results_do_not_dispatch(monkeypatch, streaming, terminal, expected):
    requests_sent = []
    provisional = [
        {"type": "response.output_text.delta", "delta": "Pending"},
        {"type": "response.output_item.done", "item": FUNCTION_CALL},
    ]

    def post(**kwargs):
        requests_sent.append(kwargs)
        yield from sse(*provisional)
        if terminal:
            yield from sse(terminal)

    monkeypatch.setattr("llm.chatgpt_plan_brain.post_stream_lines", post)
    registry = ToolRegistry()
    calls = []
    registry.register("echo", lambda request: calls.append(request) or ToolResult.success({}))
    arguments = {
        "brain": ChatGPTPlanBrain(cast(ChatGPTPlan, FakePlan()), PRINCIPAL, CONFIG),
        "gateway": ToolGateway(registry),
        "messages": MESSAGES,
        "tools": registry.list_tool_specs(),
        "config": CONFIG,
    }
    loop = AgentToolLoop()
    if streaming:
        events = list(loop.run_stream_events(**arguments))
        assert isinstance(events[-1], Done) and events[-1].finish_reason == "error"
        assert [event.code for event in events if isinstance(event, Error)] == [expected]
        assert "Private upstream error detail" not in repr(events)
    else:
        with pytest.raises(ToolLoopExecutionError) as error:
            loop.run(**arguments)
        assert error.value.cause.code == expected
        assert "Private upstream error detail" not in str(error.value.cause)
        if expected == "subscription_sharing_usage_limit_exceeded":
            assert error.value.cause.status == 429
    assert len(requests_sent) == 1
    assert calls == []


@pytest.mark.parametrize("streaming", [False, True])
def test_cancellation_closes_stream_and_discards_pending_tools(monkeypatch, streaming):
    token = asyncio.Event()
    closed = []

    def post(**kwargs):
        assert kwargs["cancellation_token"] is token
        try:
            yield from sse({"type": "response.output_item.done", "item": FUNCTION_CALL})
            token.set()
            yield from sse(completed(FUNCTION_CALL))
        finally:
            closed.append(True)

    monkeypatch.setattr("llm.chatgpt_plan_brain.post_stream_lines", post)
    registry = ToolRegistry()
    calls = []
    registry.register("echo", lambda request: calls.append(request) or ToolResult.success({}))
    arguments = {
        "brain": ChatGPTPlanBrain(cast(ChatGPTPlan, FakePlan()), PRINCIPAL, CONFIG),
        "gateway": ToolGateway(registry),
        "messages": MESSAGES,
        "tools": registry.list_tool_specs(),
        "config": CONFIG,
        "cancellation_token": token,
    }
    loop = AgentToolLoop()
    if streaming:
        events = list(loop.run_stream_events(**arguments))
        assert isinstance(events[-1], Done) and events[-1].finish_reason == "cancelled"
        assert not any(isinstance(event, Error) for event in events)
    else:
        assert loop.run(**arguments).cancelled
    assert calls == []
    assert closed == [True]


@pytest.mark.parametrize("streaming", [False, True])
def test_revoked_grant_blocks_an_existing_approval_continuation(monkeypatch, streaming):
    plan = FakePlan()
    requests_sent = []

    def post(**kwargs):
        requests_sent.append(kwargs)
        yield from sse(completed(FUNCTION_CALL))

    monkeypatch.setattr("llm.chatgpt_plan_brain.post_stream_lines", post)
    registry = ToolRegistry()
    calls = []
    registry.register(
        "echo",
        lambda request: calls.append(request) or ToolResult.success({}),
        capability="read",
        risk_classes=["network"],
    )
    gateway = ToolGateway(
        registry,
        approval_context=ApprovalContext(
            safe_mode=True, session_id="session", approved_categories=set()
        ),
    )
    arguments = {
        "brain": ChatGPTPlanBrain(cast(ChatGPTPlan, plan), PRINCIPAL, CONFIG),
        "gateway": gateway,
        "messages": MESSAGES,
        "tools": registry.list_tool_specs(),
        "config": CONFIG,
    }
    loop = AgentToolLoop()
    with pytest.raises(ApprovalRequired) as pending:
        if streaming:
            list(loop.run_stream_events(**arguments))
        else:
            loop.run(**arguments)
    assert pending.value.continuation is not None
    plan.active = False
    with pytest.raises(ToolLoopExecutionError) as error:
        loop.run(**arguments, continuation=pending.value.continuation)
    assert error.value.cause.code == "chatgpt_plan_revoked"
    assert calls == []
    assert len(requests_sent) == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_revocation_after_provider_completion_blocks_dispatch(monkeypatch, streaming):
    plan = FakePlan()

    def post(**_kwargs):
        try:
            yield from sse(completed(FUNCTION_CALL))
        finally:
            plan.active = False

    monkeypatch.setattr("llm.chatgpt_plan_brain.post_stream_lines", post)
    registry = ToolRegistry()
    calls = []
    registry.register("echo", lambda request: calls.append(request) or ToolResult.success({}))
    arguments = {
        "brain": ChatGPTPlanBrain(cast(ChatGPTPlan, plan), PRINCIPAL, CONFIG),
        "gateway": ToolGateway(registry),
        "messages": MESSAGES,
        "tools": registry.list_tool_specs(),
        "config": CONFIG,
    }
    loop = AgentToolLoop()
    with pytest.raises(ToolLoopExecutionError) as error:
        if streaming:
            list(loop.run_stream_events(**arguments))
        else:
            loop.run(**arguments)
    assert error.value.cause.code == "chatgpt_plan_revoked"
    assert calls == []


@pytest.mark.parametrize(
    ("status", "body", "shape", "expected"),
    [
        (
            429,
            {
                "error": {
                    "code": "subscription_sharing_usage_limit_exceeded",
                    "message": "Private diagnostic text",
                }
            },
            "structured",
            "subscription_sharing_usage_limit_exceeded",
        ),
        (401, {"detail": "Private diagnostic text"}, "detail", "chatgpt_plan_admission"),
    ],
)
def test_http_admission_preserves_evidence_omitting_diagnostics(
    monkeypatch, status, body, shape, expected
):
    class AdmissionFailure(FakeHttpResponse):
        async def readexactly(self, count):
            payload = json.dumps(self.payload).encode()
            raise asyncio.IncompleteReadError(payload, count)

    response = AdmissionFailure(body, status=status, headers={"x-request-id": "req_synthetic"})
    sent = []

    async def request(_session, method, url, **kwargs):
        assert method == "POST"
        assert url == "https://api.openai.com/v1/responses"
        assert kwargs["proxy"] is None
        assert not kwargs["allow_redirects"]
        sent.append(url)
        return response

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:9999")
    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)
    brain = ChatGPTPlanBrain(cast(ChatGPTPlan, FakePlan()), PRINCIPAL, CONFIG)
    with pytest.raises(ChatGPTPlanError) as error:
        brain.generate(MESSAGES)
    assert error.value.code == expected
    assert error.value.status == status
    assert error.value.upstream_status == status
    assert error.value.body_shape == shape
    assert error.value.request_id == "req_synthetic"
    assert "Private diagnostic text" not in str(error.value)
    assert response.closed
    assert len(sent) == 1


def test_stream_cancellation_closes_waiting_aiohttp_response(monkeypatch):
    upstream = HoldingProvider(monkeypatch, "body", streaming=True)
    monkeypatch.setattr(
        "llm.chatgpt_plan_brain._RESPONSES_URL", "https://provider.test/v1/responses"
    )
    brain = ChatGPTPlanBrain(cast(ChatGPTPlan, FakePlan()), PRINCIPAL, CONFIG)
    token = asyncio.Event()

    def collect():
        return list(brain.generate_stream_events(MESSAGES, cancellation_token=token))

    with ThreadPoolExecutor(max_workers=1) as thread:
        events = thread.submit(collect)
        try:
            assert upstream.started.wait(2)
        finally:
            cancel_generation(token)
        assert events.result(timeout=2) == [Done(finish_reason="cancelled")]
    assert upstream.cleaned.is_set()
    assert len(upstream.responses) == 1 and upstream.responses[0].closed


def test_stream_framing_requires_complete_multiline_sse_record(monkeypatch):
    response_data = completed(assistant("строка\u2028продолжение"))["response"]
    json_data = json.dumps(response_data, ensure_ascii=False).encode()
    chunks = [
        b'data: {"type":"response.completed",\r',
        b'\ndata: "response": ' + json_data + b"}\r",
        b"\n\r",
        b"\n",
    ]

    class StreamingResponse(FakeHttpResponse):
        async def _lines(self):
            for chunk in chunks:
                yield chunk

    async def request(_session, _method, _url, **_kwargs):
        return StreamingResponse()

    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)
    brain = ChatGPTPlanBrain(cast(ChatGPTPlan, FakePlan()), PRINCIPAL, CONFIG)
    assert brain.generate(MESSAGES).text == "строка\u2028продолжение"
