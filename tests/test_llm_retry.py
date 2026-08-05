from __future__ import annotations

import logging

import pytest
import requests

from llm.retry import ProviderRequestError, RetryPolicy, visible_provider_error


def _http_error(status_code: int, *, retry_after: str | None = None) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.HTTPError(response=response)


def test_retry_transient_error_with_backoff(caplog: pytest.LogCaptureFixture) -> None:
    attempts = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise requests.ConnectionError("temporary network failure")
        return "ok"

    policy = RetryPolicy(
        max_attempts=3,
        base_delay_seconds=0.1,
        max_delay_seconds=1,
        jitter_ratio=0,
    )
    with caplog.at_level(logging.INFO, logger="SlavikAI.LLMRetry"):
        result = policy.run(operation, provider="test", sleep=sleeps.append)

    assert result == "ok"
    assert attempts == 3
    assert sleeps == pytest.approx([0.1, 0.2])
    assert sum(record.message == "provider_request_attempt" for record in caplog.records) == 3


def test_retry_respects_retry_after() -> None:
    attempts = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _http_error(429, retry_after="2.5")
        return "ok"

    policy = RetryPolicy(max_attempts=2, max_delay_seconds=10, jitter_ratio=0)
    assert policy.run(operation, provider="test", sleep=sleeps.append) == "ok"
    assert sleeps == [2.5]


def test_retry_skips_non_transient_error() -> None:
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0, jitter_ratio=0)
    for status_code in (400, 401, 403):
        attempts = 0

        def operation(status: int = status_code) -> str:
            nonlocal attempts
            attempts += 1
            raise _http_error(status)

        with pytest.raises(ProviderRequestError) as raised:
            policy.run(operation, provider="test")
        assert attempts == 1
        assert raised.value.status_code == status_code
        assert raised.value.retryable is False
        assert raised.value.kind == "model"


def test_retry_exhaustion_preserves_timeout_kind() -> None:
    policy = RetryPolicy(max_attempts=2, base_delay_seconds=0, jitter_ratio=0)

    with pytest.raises(ProviderRequestError) as raised:
        policy.run(
            lambda: (_ for _ in ()).throw(requests.Timeout("slow provider")),
            provider="test",
        )

    assert raised.value.kind == "timeout"
    assert raised.value.code == "provider_timeout"
    assert "Таймаут" in raised.value.user_message


def test_retry_stops_when_cancellation_requested() -> None:
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise requests.ConnectionError("request was cancelled")

    with pytest.raises(requests.ConnectionError, match="cancelled"):
        RetryPolicy().run(
            operation,
            provider="test",
            stop_requested=lambda: True,
        )

    assert attempts == 1


@pytest.mark.parametrize(
    ("error", "expected_code", "expected_kind"),
    [
        (requests.ConnectionError("offline"), "provider_network_error", "network"),
        (requests.Timeout("slow"), "provider_timeout", "timeout"),
        (_http_error(503), "provider_model_error", "model"),
    ],
)
def test_visible_provider_error_classifies_streaming_transport_errors(
    error: requests.RequestException,
    expected_code: str,
    expected_kind: str,
) -> None:
    code, message, kind = visible_provider_error(error)

    assert code == expected_code
    assert kind == expected_kind
    assert message
