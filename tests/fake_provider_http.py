"""Детерминированные upstream boundaries без DNS/TCP запросов."""

import asyncio
import json
import socket
import ssl
import threading

import aiohttp
from aiohttp.abc import AbstractResolver


class FakeHttpResponse:
    def __init__(self, payload=None, status=200, headers=None, read=None):
        self.status = status
        self.headers = headers or {}
        self.payload = payload or {"choices": [{"message": {"content": "ok"}}]}
        self._read = read
        self.closed = False

    async def read(self):
        if self._read is not None:
            await self._read()
        return json.dumps(self.payload).encode()

    def release(self):
        self.closed = True

    async def wait_for_close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.release()
        await self.wait_for_close()


class HoldingProvider:
    def __init__(self, monkeypatch, stage):
        self.stage = stage
        self.started = threading.Event()
        self.cleaned = threading.Event()
        self.active = 0
        self.calls = 0
        self.resolvers = []
        self.responses = []
        owner = self

        class Resolver(AbstractResolver):
            def __init__(self, **kwargs):
                self.closed = False
                owner.resolvers.append(self)

            async def resolve(self, host, port=0, family=socket.AF_INET):
                assert host == "provider.test"
                if stage == "dns":
                    await owner.hold()
                return [
                    {
                        "hostname": host,
                        "host": "192.0.2.1",
                        "port": port,
                        "family": socket.AF_INET,
                        "proto": 0,
                        "flags": 0,
                    }
                ]

            async def close(self):
                self.closed = True

        monkeypatch.setattr(aiohttp, "AsyncResolver", Resolver)

        original_connect = aiohttp.TCPConnector._wrap_create_connection

        async def connect(connector, *args, **kwargs):
            if kwargs["req"].url.host != "provider.test":
                return await original_connect(connector, *args, **kwargs)
            if stage == "tls":
                assert isinstance(kwargs["ssl"], ssl.SSLContext)
                assert kwargs["ssl"].verify_mode == ssl.CERT_REQUIRED
            else:
                assert not kwargs.get("ssl")
            await owner.hold()
            raise AssertionError("cancelled connect must not resume")

        monkeypatch.setattr(aiohttp.TCPConnector, "_wrap_create_connection", connect)
        original_request = aiohttp.ClientSession._request

        async def request(session, method, url, **kwargs):
            if "provider.test" not in str(url):
                return await original_request(session, method, url, **kwargs)
            owner.calls += 1
            assert kwargs.get("allow_redirects") is False
            assert kwargs["json"].get("stream") is not True
            if owner.calls == 1:
                if stage in {"dns", "connect", "tls"}:
                    return await original_request(session, method, url, **kwargs)
                if stage == "headers":
                    await owner.hold()
            response = FakeHttpResponse(
                read=owner.hold if stage == "body" and owner.calls == 1 else None
            )
            owner.responses.append(response)
            return response

        monkeypatch.setattr(aiohttp.ClientSession, "_request", request)

    async def hold(self):
        self.active += 1
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.active -= 1
            self.cleaned.set()
