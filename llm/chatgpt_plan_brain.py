from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Generator, Iterator
from contextlib import closing

import requests

from llm.brain_base import Brain
from llm.cancellation import GenerationCancelled, cancellation_requested
from llm.chatgpt_plan import ChatGPTPlan, ChatGPTPlanError
from llm.provider_http import post_stream_lines
from llm.stream_model import Done, Error, StreamEvent, TextDelta, ToolCallCompleted, Usage
from llm.types import LLMResult, ModelConfig, ToolCall, ToolSpec, is_chatgpt_plan_provider
from llm.usage import parse_reported_usage
from shared.models import JSONValue, LLMMessage

_NAMESPACE = "slavikai"
_RESPONSES_URL = "https://api.openai.com/v1/responses"
_MAX_EVENT_SIZE = 4 * 1024 * 1024
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,100}")

_ERRORS: dict[str, tuple[int, str]] = {
    "subscription_sharing_user_not_eligible": (
        403,
        "Этот аккаунт или workspace не допускает ChatGPT plan.",
    ),
    "subscription_sharing_usage_limit_exceeded": (
        429,
        "Достигнут лимит использования ChatGPT plan.",
    ),
    "subscription_sharing_usage_unavailable": (
        503,
        "Проверка лимита ChatGPT plan временно недоступна.",
    ),
    "subscription_sharing_unsupported_capability": (
        400,
        "Модель или возможность недоступна через ChatGPT plan.",
    ),
    "subscription_sharing_route_not_supported": (403, "OpenAI отклонил маршрут ChatGPT plan."),
    "subscription_sharing_invalid_user": (401, "OpenAI не подтвердил аккаунт ChatGPT plan."),
    "chatpass_v2_scope_not_authorized": (403, "У аккаунта отсутствует разрешение на ChatGPT plan."),
    "chatpass_v2_invalid_authorization_context": (403, "OpenAI отклонил разрешение ChatGPT plan."),
    "subscription_sharing_user_unavailable": (503, "Аккаунт ChatGPT plan временно недоступен."),
}
_ADMISSION_ERRORS = {
    401: "OpenAI отклонил авторизацию ChatGPT plan.",
    403: "OpenAI отклонил доступ к ChatGPT plan.",
    429: "OpenAI ограничил частоту запросов ChatGPT plan.",
    503: "Маршрут ChatGPT plan временно недоступен.",
}


def _request_payload(
    messages: list[LLMMessage], config: ModelConfig, tools: list[ToolSpec] | None
) -> dict[str, JSONValue]:
    items: list[JSONValue] = []
    for message in messages:
        if message.role == "assistant" and message.responses_items:
            items.extend(message.responses_items)
        elif message.role == "tool" and message.tool_call_id:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
        elif message.role != "tool" and not message.tool_calls:
            role = "developer" if message.role == "system" else message.role
            items.append({"role": role, "content": message.content})
        else:
            raise ChatGPTPlanError(
                "chatgpt_plan_history",
                "История инструментов несовместима с маршрутом ChatGPT plan.",
            )
    result: dict[str, JSONValue] = {
        "model": config.model,
        "input": items,
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
    }
    if config.system_prompt:
        result["instructions"] = config.system_prompt
    if tools:
        result["tools"] = [
            {
                "type": "namespace",
                "name": _NAMESPACE,
                "description": "Инструменты SlavikAI с локальной проверкой полномочий.",
                "tools": [
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_schema
                        or {"type": "object", "properties": {}},
                        "strict": False,
                    }
                    for tool in tools
                ],
            }
        ]
    return result


def _protocol_error() -> ChatGPTPlanError:
    return ChatGPTPlanError(
        "chatgpt_plan_protocol", "OpenAI вернул некорректный ответ ChatGPT plan."
    )


def _interrupted() -> ChatGPTPlanError:
    return ChatGPTPlanError(
        "chatgpt_plan_interrupted", "Поток ChatGPT plan завершился без итогового события."
    )


def _provider_error(
    error: JSONValue,
    status: int | None = None,
    *,
    request_id: str | None = None,
    body_shape: str | None = None,
) -> ChatGPTPlanError:
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and (known := _ERRORS.get(code)) is not None:
        documented_status, message = known
        return ChatGPTPlanError(
            code,
            message,
            status=status or documented_status,
            request_id=request_id,
            body_shape=body_shape,
        )
    if status is not None and status in _ADMISSION_ERRORS:
        return ChatGPTPlanError(
            "chatgpt_plan_admission",
            _ADMISSION_ERRORS[status],
            status=status,
            request_id=request_id,
            body_shape=body_shape,
        )
    return ChatGPTPlanError(
        "chatgpt_plan_upstream",
        "Ошибка OpenAI при запросе ChatGPT plan.",
        status=status or 502,
        request_id=request_id,
        body_shape=body_shape,
    )


def _http_error(response: requests.Response | None) -> ChatGPTPlanError:
    if response is None:
        return _provider_error(None)
    header = response.headers.get("x-request-id", "")
    request_id = header if _REQUEST_ID.fullmatch(header) else None
    try:
        body = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        shape = "structured"
    elif isinstance(body, dict) and "detail" in body:
        shape = "detail"
    else:
        shape = "unknown"
    status = response.status_code
    failure = _provider_error(
        error, status if status >= 400 else None, request_id=request_id, body_shape=shape
    )
    failure.upstream_status = status
    return failure


def _response_events(lines: Iterator[str]) -> Iterator[dict[str, JSONValue]]:
    data: list[str] = []
    size = 0
    for line in lines:
        line = line.rstrip("\r\n")
        if line.startswith("data:"):
            value = line.removeprefix("data:").removeprefix(" ")
            size += len(value) + 1
            if size > _MAX_EVENT_SIZE:
                raise _protocol_error()
            data.append(value)
        elif not line and data:
            event = json.loads("\n".join(data))
            data.clear()
            size = 0
            if not isinstance(event, dict):
                raise _protocol_error()
            yield event
    if data:
        raise _interrupted()


def _message_text(item: dict[str, JSONValue]) -> str:
    content = item.get("content")
    if (
        item.get("role") != "assistant"
        or item.get("status", "completed") != "completed"
        or not isinstance(content, list)
    ):
        raise _protocol_error()
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            raise _protocol_error()
        kind = part.get("type")
        text = part.get("text") if kind == "output_text" else part.get("refusal")
        if kind not in {"output_text", "refusal"} or not isinstance(text, str):
            raise _protocol_error()
        parts.append(text)
    return "".join(parts)


def _function_call(item: dict[str, JSONValue]) -> ToolCall:
    name = item.get("name")
    call_id = item.get("call_id")
    raw = item.get("arguments")
    if (
        not isinstance(call_id, str)
        or not call_id
        or not isinstance(name, str)
        or not name
        or not isinstance(raw, str)
        or item.get("namespace") != _NAMESPACE
        or item.get("status", "completed") != "completed"
    ):
        raise _protocol_error()
    arguments = json.loads(raw)
    if not isinstance(arguments, dict):
        raise _protocol_error()
    return ToolCall(call_id, name, arguments, raw)


def _completed(response: JSONValue, tools: list[ToolSpec] | None) -> LLMResult:
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise _protocol_error()
    output = response.get("output")
    if not isinstance(output, list):
        raise _protocol_error()
    items: list[dict[str, JSONValue]] = []
    texts: list[str] = []
    calls: list[ToolCall] = []
    seen: set[str] = set()
    for item in output:
        if not isinstance(item, dict):
            raise _protocol_error()
        kind = item.get("type")
        keys: tuple[str, ...]
        if kind == "message":
            texts.append(_message_text(item))
            keys = ("id", "status", "type", "role", "content", "phase")
        elif kind == "function_call":
            call = _function_call(item)
            if not tools or call.id in seen:
                raise _protocol_error()
            seen.add(call.id)
            calls.append(call)
            keys = ("id", "status", "type", "namespace", "call_id", "name", "arguments")
        elif kind == "reasoning":
            if not isinstance(item.get("encrypted_content"), str):
                continue
            keys = ("id", "type", "encrypted_content")
        else:
            raise _protocol_error()
        replay = {key: item[key] for key in keys if key in item}
        if kind == "reasoning":
            replay["summary"] = []
        items.append(replay)

    if not texts and not calls:
        raise _protocol_error()
    usage = None
    raw_usage = response.get("usage")
    if isinstance(raw_usage, dict):
        usage = parse_reported_usage(
            {
                "prompt_tokens": raw_usage.get("input_tokens"),
                "completion_tokens": raw_usage.get("output_tokens"),
                "total_tokens": raw_usage.get("total_tokens"),
            }
        )
    return LLMResult(
        text="".join(texts),
        usage=usage,
        tool_calls=calls,
        responses_items=tuple(items),
    )


def _check_cancellation(token: asyncio.Event | None) -> None:
    if cancellation_requested(token):
        raise GenerationCancelled("Запрос ChatGPT plan отменён.")


class ChatGPTPlanBrain(Brain):
    supports_native_tools = True
    supports_streaming_tools = True

    def __init__(self, plan: ChatGPTPlan, principal: str, config: ModelConfig) -> None:
        self._plan = plan
        self._principal = principal
        self.config = config

    def _active_config(self, config: ModelConfig | None) -> ModelConfig:
        active = config or self.config
        if (
            not is_chatgpt_plan_provider(active.provider)
            or active.provider != self.config.provider
            or active.model != self.config.model
        ):
            raise ChatGPTPlanError(
                "chatgpt_plan_route", "Маршрут ChatGPT plan изменился.", status=409
            )
        return active

    def tool_dispatch_guard(self, config: ModelConfig | None) -> Callable[[], None]:
        def validate() -> None:
            active = self._active_config(config)
            self._plan.validate_grant(self._principal, active.provider)

        return validate

    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        return self.generate_cancellable(messages, config, None, tools)

    def generate_for_run(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None,
        tools: list[ToolSpec],
        cancellation_token: asyncio.Event | None,
    ) -> LLMResult:
        return self.generate_cancellable(messages, config, cancellation_token, tools)

    def generate_cancellable(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None,
        cancellation_token: asyncio.Event | None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        with closing(self._generate(messages, config, tools, cancellation_token)) as stream:
            try:
                while True:
                    next(stream)
            except StopIteration as finished:
                result: LLMResult = finished.value
                return result

    def _generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None,
        tools: list[ToolSpec] | None,
        cancellation_token: asyncio.Event | None,
    ) -> Generator[str, None, LLMResult]:
        _check_cancellation(cancellation_token)
        active = self._active_config(config)
        payload = _request_payload(messages, active, tools)
        token = self._plan.access(self._principal, active.provider)
        _check_cancellation(cancellation_token)
        started = False
        try:
            with closing(
                post_stream_lines(
                    url=_RESPONSES_URL,
                    headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"},
                    payload=payload,
                    timeout=45,
                    cancellation_token=cancellation_token,
                    direct=True,
                    include_error_body=True,
                )
            ) as lines:
                for event in _response_events(lines):
                    started = True
                    _check_cancellation(cancellation_token)
                    kind = event.get("type")
                    if kind in {"response.output_text.delta", "response.refusal.delta"}:
                        delta = event.get("delta")
                        if not isinstance(delta, str):
                            raise _protocol_error()
                        yield delta
                    elif kind == "response.completed":
                        result = _completed(event.get("response"), tools)
                        self._plan.validate_grant(self._principal, active.provider)
                        return result
                    elif kind == "response.incomplete":
                        raise ChatGPTPlanError(
                            "chatgpt_plan_incomplete", "OpenAI не завершил ответ ChatGPT plan."
                        )
                    elif kind == "response.failed":
                        response = event.get("response")
                        raise _provider_error(
                            response.get("error") if isinstance(response, dict) else None
                        )
                    elif kind == "error":
                        raise _provider_error(event)
        except requests.HTTPError as exc:
            _check_cancellation(cancellation_token)
            raise _http_error(exc.response) from None
        except requests.RequestException:
            _check_cancellation(cancellation_token)
            if started:
                raise _interrupted() from None
            raise ChatGPTPlanError(
                "chatgpt_plan_transport", "Соединение с OpenAI для ChatGPT plan недоступно."
            ) from None
        except (ValueError, TypeError):
            raise _protocol_error() from None
        _check_cancellation(cancellation_token)
        raise _interrupted()

    def generate_stream_events(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
        cancellation_token: asyncio.Event | None = None,
    ) -> Iterator[StreamEvent]:
        deltas: list[str] = []
        try:
            with closing(self._generate(messages, config, tools, cancellation_token)) as stream:
                while True:
                    delta = next(stream)
                    deltas.append(delta)
                    yield TextDelta(delta)
        except StopIteration as finished:
            result: LLMResult = finished.value
        except GenerationCancelled:
            yield Done(finish_reason="cancelled")
            return
        except ChatGPTPlanError as exc:
            yield Error(message=str(exc), code=exc.code)
            yield Done(finish_reason="error")
            return

        visible = "".join(deltas)
        if result.text.startswith(visible):
            if remainder := result.text[len(visible) :]:
                yield TextDelta(remainder)
        else:
            yield TextDelta(result.text, mode="replace")
        if result.usage is not None:
            yield Usage(result.usage)
        for call in result.tool_calls:
            yield ToolCallCompleted(call=call)
        yield Done(
            finish_reason="tool_calls" if result.tool_calls else "stop",
            responses_items=result.responses_items,
        )
