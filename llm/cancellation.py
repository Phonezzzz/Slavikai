from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Protocol

from llm.stream_model import Done


class ClosableResource(Protocol):
    def close(self) -> object: ...


class GenerationCancelled(RuntimeError):
    """Внутренний сигнал отмены блокирующего вызова провайдера."""


_EventResources = weakref.WeakKeyDictionary[asyncio.Event, list[ClosableResource]]
_resources: _EventResources = weakref.WeakKeyDictionary()
_resources_lock = threading.RLock()


def cancellation_requested(token: asyncio.Event | None) -> bool:
    return token is not None and token.is_set()


@contextmanager
def bind_cancellation_resource(
    token: asyncio.Event | None,
    resource: ClosableResource,
) -> Iterator[ClosableResource]:
    """Связывает закрываемый provider-resource с токеном на время запроса."""
    if token is not None:
        with _resources_lock:
            _resources.setdefault(token, []).append(resource)
    try:
        yield resource
    finally:
        if token is not None:
            with _resources_lock:
                resources = _resources.get(token)
                if resources is not None:
                    _resources[token] = [item for item in resources if item is not resource]
                    if not _resources[token]:
                        del _resources[token]
        close = getattr(resource, "close", None)
        if callable(close):
            close()


def cancel_generation(token: asyncio.Event) -> list[str]:
    """Устанавливает токен и немедленно закрывает активные provider-resources."""
    token.set()
    with _resources_lock:
        resources = list(_resources.get(token, ()))
    errors: list[str] = []
    for resource in resources:
        try:
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(resource).__name__}: {exc}")
    return errors


def iter_cancellable[T](
    items: Iterable[T],
    *,
    cancellation_token: asyncio.Event | None,
) -> Iterator[T | Done]:
    """Останавливает синхронный поток и нормализует close-error в Done(cancelled)."""
    iterator = iter(items)
    try:
        while True:
            if cancellation_requested(cancellation_token):
                yield Done(finish_reason="cancelled")
                return
            try:
                item = next(iterator)
            except StopIteration:
                return
            except Exception as exc:  # noqa: BLE001
                if cancellation_requested(cancellation_token):
                    yield Done(finish_reason="cancelled")
                    return
                raise exc
            if cancellation_requested(cancellation_token):
                yield Done(finish_reason="cancelled")
                return
            yield item
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
