from __future__ import annotations

import asyncio
import ipaddress
import ssl
import threading
from urllib.parse import urlsplit

import aiohttp
import requests
from requests.utils import DEFAULT_CA_BUNDLE_PATH, select_proxy

from llm.cancellation import GenerationCancelled, bind_cancellation_resource, cancellation_requested
from llm.retry import RetryPolicy
from shared.models import JSONValue


def is_loopback_url(url: str) -> bool:
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def proxies_for_provider_url(url: str) -> dict[str, str] | None:
    # Пустые значения блокируют scheme proxy и ALL_PROXY fallback requests.
    if is_loopback_url(url):
        return {"http": "", "https": "", "all": ""}
    return None


class _ProviderOperation:
    """Один request scope: task cancellation и прерываемое ожидание retry."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.stopped = threading.Event()
        self.task: asyncio.Task[requests.Response] | None = None

    def close(self) -> None:
        self.stopped.set()
        try:
            self.loop.call_soon_threadsafe(self._cancel_task)
        except RuntimeError:
            # Cancel может получить ресурс непосредственно перед закрытием loop.
            if not self.loop.is_closed():
                raise

    def _cancel_task(self) -> None:
        if self.task is not None and not self.task.cancelling():
            self.task.cancel()

    def sleep(self, delay: float) -> None:
        self.stopped.wait(delay)


async def _post(
    url: str, headers: dict[str, str], payload: dict[str, JSONValue], timeout: int
) -> requests.Response:
    # Dedicated resolver принадлежит попытке; blocking getaddrinfo/executor нет.
    resolver = aiohttp.AsyncResolver(timeout=timeout)
    try:
        async with aiohttp.TCPConnector(resolver=resolver) as connector:
            # Сохраняем разрешённую remote proxy policy requests, но не наследуем
            # env повторно в aiohttp и не разрешаем loopback proxy fallback.
            with requests.Session() as settings_session:
                settings = settings_session.merge_environment_settings(
                    url, proxies_for_provider_url(url) or {}, False, True, None
                )
            proxy = select_proxy(url, settings["proxies"]) or None
            verify = settings["verify"]
            ssl_context = ssl.create_default_context(
                cafile=verify if isinstance(verify, str) else DEFAULT_CA_BUNDLE_PATH
            )
            async with aiohttp.ClientSession(
                connector=connector,
                trust_env=False,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    allow_redirects=False,
                    proxy=proxy,
                    ssl=ssl_context,
                ) as upstream:
                    response = requests.Response()
                    response.status_code = upstream.status
                    response.headers.update(upstream.headers)
                    response.url = url
                    # Проверка status до body read: 429/5xx не удерживают retry
                    # на зависшем error body. Context закроет upstream до retry.
                    response.raise_for_status()
                    response._content = await upstream.read()
                    return response
    except TimeoutError as exc:
        raise requests.Timeout("Provider request timed out") from exc
    except aiohttp.ClientError as exc:
        raise requests.ConnectionError("Provider transport failed") from exc
    finally:
        await resolver.close()


def post_plain_completion(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, JSONValue],
    provider: str,
    timeout: int,
    cancellation_token: asyncio.Event,
) -> requests.Response:
    # Вызов выполняется в существующем generation worker, новый worker не нужен.
    with asyncio.Runner() as runner:
        operation = _ProviderOperation(runner.get_loop())
        with bind_cancellation_resource(cancellation_token, operation):

            def attempt() -> requests.Response:
                if cancellation_requested(cancellation_token):
                    raise GenerationCancelled("Provider request cancelled")
                operation.task = operation.loop.create_task(_post(url, headers, payload, timeout))
                try:
                    return operation.loop.run_until_complete(operation.task)
                except asyncio.CancelledError as exc:
                    raise GenerationCancelled("Provider request cancelled") from exc

            try:
                return RetryPolicy().run(
                    attempt,
                    provider=provider,
                    sleep=operation.sleep,
                    stop_requested=cancellation_token.is_set,
                )
            except Exception as exc:
                if cancellation_requested(cancellation_token):
                    raise GenerationCancelled("Provider request cancelled") from exc
                raise
