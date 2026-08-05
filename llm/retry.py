from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal

import requests

ProviderErrorKind = Literal["network", "timeout", "model"]

logger = logging.getLogger("SlavikAI.LLMRetry")


class ProviderRequestError(RuntimeError):
    def __init__(
        self,
        *,
        provider: str,
        kind: ProviderErrorKind,
        status_code: int | None,
        retryable: bool,
        attempts: int,
    ) -> None:
        self.provider = provider
        self.kind = kind
        self.status_code = status_code
        self.retryable = retryable
        self.attempts = attempts
        super().__init__(self.user_message)

    @property
    def code(self) -> str:
        return {
            "network": "provider_network_error",
            "timeout": "provider_timeout",
            "model": "provider_model_error",
        }[self.kind]

    @property
    def user_message(self) -> str:
        if self.kind == "timeout":
            return f"Таймаут при обращении к провайдеру {self.provider}. Повторите запрос."
        if self.kind == "network":
            return f"Сетевая ошибка при обращении к провайдеру {self.provider}. Повторите запрос."
        if self.status_code is not None:
            return f"Провайдер {self.provider} отклонил запрос (HTTP {self.status_code})."
        return f"Провайдер {self.provider} вернул ошибку модели."


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 8.0
    jitter_ratio: float = 0.2

    def run[T](
        self,
        operation: Callable[[], T],
        *,
        provider: str,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        stop_requested: Callable[[], bool] | None = None,
    ) -> T:
        attempts = max(1, self.max_attempts)
        for attempt in range(1, attempts + 1):
            logger.info(
                "provider_request_attempt",
                extra={"provider": provider, "attempt": attempt, "max_attempts": attempts},
            )
            try:
                return operation()
            except requests.RequestException as exc:
                if stop_requested is not None and stop_requested():
                    logger.info(
                        "provider_request_retry_cancelled",
                        extra={"provider": provider, "attempt": attempt},
                    )
                    raise
                failure = _classify_request_error(exc)
                if not failure.retryable or attempt >= attempts:
                    logger.error(
                        "provider_request_failed",
                        extra={
                            "provider": provider,
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "kind": failure.kind,
                            "status_code": failure.status_code,
                            "retryable": failure.retryable,
                        },
                    )
                    raise ProviderRequestError(
                        provider=provider,
                        kind=failure.kind,
                        status_code=failure.status_code,
                        retryable=failure.retryable,
                        attempts=attempt,
                    ) from exc
                retry_after = _retry_after_seconds(exc, now=now)
                delay = (
                    retry_after
                    if retry_after is not None
                    else self._backoff_seconds(attempt, random_value=random_value)
                )
                delay = min(max(0.0, delay), max(0.0, self.max_delay_seconds))
                logger.warning(
                    "provider_request_retry",
                    extra={
                        "provider": provider,
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "kind": failure.kind,
                        "status_code": failure.status_code,
                        "delay_seconds": delay,
                    },
                )
                sleep(delay)
                if stop_requested is not None and stop_requested():
                    logger.info(
                        "provider_request_retry_cancelled",
                        extra={"provider": provider, "attempt": attempt},
                    )
                    raise
        raise AssertionError("retry loop must return or raise")

    def _backoff_seconds(
        self,
        attempt: int,
        *,
        random_value: Callable[[], float],
    ) -> float:
        base: float = min(
            max(0.0, self.max_delay_seconds),
            max(0.0, self.base_delay_seconds) * (2.0 ** max(0, attempt - 1)),
        )
        ratio = max(0.0, self.jitter_ratio)
        jitter: float = base * ratio * ((2 * min(max(random_value(), 0.0), 1.0)) - 1)
        return max(0.0, base + jitter)


@dataclass(frozen=True, slots=True)
class _RequestFailure:
    kind: ProviderErrorKind
    status_code: int | None
    retryable: bool


def _classify_request_error(exc: requests.RequestException) -> _RequestFailure:
    if isinstance(exc, requests.Timeout):
        return _RequestFailure(kind="timeout", status_code=None, retryable=True)
    if isinstance(exc, requests.ConnectionError):
        return _RequestFailure(kind="network", status_code=None, retryable=True)
    if isinstance(exc, requests.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else None
        return _RequestFailure(
            kind="model",
            status_code=status_code,
            retryable=_is_transient_status(status_code),
        )
    return _RequestFailure(kind="network", status_code=None, retryable=True)


def _is_transient_status(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code in {408, 425, 429} or 500 <= status_code <= 599


def _retry_after_seconds(
    exc: requests.RequestException,
    *,
    now: Callable[[], datetime],
) -> float | None:
    if not isinstance(exc, requests.HTTPError) or exc.response is None:
        return None
    raw = exc.response.headers.get("Retry-After")
    if raw is None or not raw.strip():
        return None
    normalized = raw.strip()
    try:
        return max(0.0, float(normalized))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(normalized)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (parsed - now()).total_seconds())


def visible_provider_error(exc: Exception) -> tuple[str, str, ProviderErrorKind]:
    if isinstance(exc, ProviderRequestError):
        return exc.code, exc.user_message, exc.kind
    if isinstance(exc, requests.RequestException):
        failure = _classify_request_error(exc)
        if failure.kind == "timeout":
            return (
                "provider_timeout",
                "Таймаут при обращении к провайдеру. Повторите запрос.",
                "timeout",
            )
        if failure.kind == "network":
            return (
                "provider_network_error",
                "Сетевая ошибка при обращении к провайдеру. Повторите запрос.",
                "network",
            )
        if failure.status_code is not None:
            return (
                "provider_model_error",
                f"Провайдер отклонил запрос (HTTP {failure.status_code}).",
                "model",
            )
    return "provider_model_error", f"Ошибка модели: {exc}", "model"


def request_with_retry(
    operation: Callable[[], requests.Response],
    *,
    provider: str,
    policy: RetryPolicy | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> requests.Response:
    def checked_request() -> requests.Response:
        response = operation()
        response.raise_for_status()
        return response

    return (policy or RetryPolicy()).run(
        checked_request,
        provider=provider,
        stop_requested=stop_requested,
    )
